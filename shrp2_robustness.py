# ==================================================================
# Robustness block for the headline result (within-driver attenuation test)  -- review-fix version
#   - outcomes: crashes (primary), crashes + near-crashes (secondary), at-fault crashes (robustness)
#   - kappa_z on the test set D (sigma^2 / residual between-driver variance given the covariates), review fix M1
#   - driver-level bootstrap (B reps) of the whole curve: CI for slope(12)-slope(1); kappa_Y fitted from the crash
#     curve; PAIRED test of kappa_Y = kappa_z (both recomputed in the same resamples), review fixes M3/M5
#   - D_b: the test repeated with D defined by >= 12 baseline trips (903 drivers), review fix D1:
#       (a) random subsets drawn from all rows of those drivers, (b) drawn from the rows that remain after
#           removing the likely event trips (trimmed rows)
#   - exposure: days offset + mileage band covariate (main)  vs  approximate miles offset
#   - negative binomial check of the slope curve (statsmodels)
#   - Buhlmann-Straub kappa with trip-duration weights, and the duration check for time > 80 mph (M8)
#   - event-signature threshold sensitivity (1.05 / 1.08 / 1.12)
#   - Table 6 (de-attenuation) by outcome with the covariate-conditional calibration (M2) and C_z(harmonic mean n) (M6)
# Builds on the data prep of shrp2_prelim.py (exec of its head).
# ==================================================================
import io, contextlib, os, sys, time, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import erf, sqrt

t0 = time.time()
src = open("shrp2_prelim.py").read()
head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
OUT4 = f"{OUT}/robustness"; os.makedirs(OUT4, exist_ok=True)
PUR, ORA, GRY, BLU, GRN = "#534AB7", "#D85A30", "#8A8A8A", "#2F6DB5", "#3B8C5A"
B_BOOT = int(os.environ.get("B_BOOT", 2000)); B_SENS = int(os.environ.get("B_SENS", 500))
N_DRAW, M_GRID, M_MAX = 6, np.array([1, 2, 3, 4, 6, 8, 12]), 12
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s",
        "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
print(f"B_BOOT={B_BOOT}, B_SENS={B_SENS}", flush=True)

# ---------- outcomes and exposures (driver level, in drv order) ----------
y_crash = drv.crashes.values.astype(float)
y_all = (drv.crashes + drv.near).values.astype(float)
y_fault = g["N_at_fault_crash"].max().reindex(drv.index).fillna(0).values.astype(float)
OUTCOMES = {"crashes (primary)": y_crash, "crashes + near-crashes (secondary)": y_all, "at-fault crashes (robustness)": y_fault}
mid = {"less than 5,000 miles": 2500, "5,000 - 10,000 miles": 7500, "10,000 - 15,000 miles": 12500, "15,000 - 20,000 miles": 17500,
       "20,000 - 25,000 miles": 22500, "25,000 - 30,000 miles": 27500, "more than 30,000 miles": 35000}
miles_pa = D["Avg_Annual_Mileage"].map(mid).astype(float); miles_pa = miles_pa.fillna(miles_pa.median()).values
E_miles = (miles_pa * (E) / 10000.0)                      # E is days/365 -> expected miles in study, in 10,000-mile units
Xt_nomileage = Xt[:, [i for i, c in enumerate(Xtrad.columns) if not c.startswith("Avg_Annual_Mileage")]]

# ---------- likely event rows (event-signature index), as in the sensitivity script ----------
SIG = {"Maximum Deceleration": -1, "ABS Activation": 1, "Maximum Acceleration": 1, "Number of Lateral Accels > Threshold": 1,
       "Maximum Lateral Acceleration": 1, "Time Where Headway 0.0 - 0.5 s": 1, "Time Where Headway 0.5 - 1.0 s": 1, "Brake Activations": 1}
Zsig = np.zeros(len(df))
for c, sgn in SIG.items():
    v = df[c].astype(float); Zsig += sgn * ((v - v.mean()) / (v.std() + 1e-9)).fillna(0).values
df["sig_index"] = Zsig / len(SIG)
k_ev = (drv.crashes + drv.near).reindex(df["Participant_ID"]).values
df["rank_in_driver"] = df.groupby("Participant_ID")["sig_index"].rank(ascending=False, method="first")
df["likely_event"] = df["rank_in_driver"] <= k_ev

# ---------- fast Poisson IRLS (ridge on non-intercept, matched to sklearn alpha=1e-3 scaling) ----------
def irls_slope(X, y, E, alpha=1e-3, iters=25):
    n, p = X.shape; X1 = np.column_stack([np.ones(n), X]); lam = alpha * E.sum()
    P = np.eye(p + 1) * lam; P[0, 0] = 0
    beta = np.zeros(p + 1); beta[0] = np.log(y.sum() / E.sum()); logE = np.log(E)
    for _ in range(iters):
        eta = X1 @ beta + logE; mu = np.exp(np.clip(eta, -30, 30))
        z = (eta - logE) + (y - mu) / mu
        A = (X1 * mu[:, None]).T @ X1 + P; b = (X1 * mu[:, None]).T @ z
        new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < 1e-8: beta = new; break
        beta = new
    return beta[-1]

KG = np.exp(np.linspace(np.log(0.05), np.log(300), 300))
def fit_kappa(s, w=None):
    w = np.ones_like(s) if w is None else w; best = None
    for kk in KG:
        f = M_GRID / (M_GRID + kk); bb = (w * s * f).sum() / (w * f ** 2).sum(); chi = (w * (s - bb * f) ** 2).sum()
        if best is None or chi < best[0]: best = (chi, kk, bb)
    return best[1], best[2], best[0]

def two_sided_p(x, B):
    """two-sided bootstrap p-value for 'x = 0' from the bootstrap distribution x, floor 2/(B+1)"""
    p = 2 * min((x <= 0).mean(), (x >= 0).mean()); return min(1.0, max(p, 2.0 / (B + 1)))

def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)] if np.isfinite(a).any() else a
    return np.percentile(a, q) if len(a) else np.nan

# ---------- a "design" = set of drivers + the rows their random subsets are drawn from ----------
def build_design(keep_ids, rows, label):
    """precompute per-indicator draw matrices XM[c][m] -> [nk, N_DRAW] standardised subsample means,
       and per-driver summaries (n_i, mean_i, SS_within_i) aligned to the kept drivers, for kappa and kappa_z."""
    kpos = np.where(drv.index.isin(keep_ids))[0]; nk = len(kpos); rng = np.random.default_rng(5)
    XM, SUMM = {}, {}
    sub_all = rows[rows["Participant_ID"].isin(keep_ids)]
    for c in TOP6:
        sub = sub_all[["Participant_ID", c]]
        grp = sub.groupby("Participant_ID")[c].apply(lambda s: s.values).reindex(keep_ids); maxn = max(len(a) for a in grp)
        pad = np.full((nk, maxn), np.nan)
        for r_, a in enumerate(grp): pad[r_, :len(a)] = a
        mu_c, sd_c = np.nanmean(beh[c]), np.nanstd(beh[c]) + 1e-9           # fixed standardisation (full-sample driver means)
        XM[c] = {}
        for m_ in M_GRID:
            cols = []
            for d in range(N_DRAW):
                U = rng.random(pad.shape); U[np.isnan(pad)] = np.inf
                order = np.argsort(U, axis=1)[:, :m_]
                xm = np.nanmean(np.take_along_axis(pad, order, axis=1), axis=1); xm = np.where(np.isfinite(xm), xm, mu_c)
                cols.append((xm - mu_c) / sd_c)
            XM[c][m_] = np.column_stack(cols)
        n_i = np.sum(~np.isnan(pad), axis=1).astype(float); m_i = np.nanmean(pad, axis=1)
        ssw_i = np.nansum((pad - m_i[:, None]) ** 2, axis=1)
        SUMM[c] = (n_i, m_i, ssw_i)
    print(f"[{label}] drivers: {nk}; rows: {len(sub_all)}; precompute done in {time.time()-t0:.0f}s", flush=True)
    return dict(label=label, kpos=kpos, nk=nk, XM=XM, SUMM=SUMM)

def kappas_on(des, c, idx):
    """marginal kappa and kappa_z on the drivers idx (positions within the design), from the stored summaries"""
    n_i, m_i, ssw_i = des["SUMM"][c]; Zk = Xt[des["kpos"]]
    n, m, ssw, Z = n_i[idx], m_i[idx], ssw_i[idx], Zk[idx]
    ok = n >= 1; n, m, ssw, Z = n[ok], m[ok], ssw[ok], Z[ok]
    k_marg = vc_from_summaries(n, m, ssw)[2]; s2, tz2, k_z, _ = vc_z_from_summaries(n, m, ssw, Z)
    return k_marg, k_z, s2, tz2

def curve(des, c, idx, yv, Ev, Xtv):
    """mean slope over draws for each m, on driver subset idx (positions within the design)"""
    XM = des["XM"][c]
    return np.array([np.mean([irls_slope(np.column_stack([Xtv[idx], XM[m_][idx, d]]), yv[idx], Ev[idx]) for d in range(N_DRAW)]) for m_ in M_GRID])

def run_test(des, outcomes, B, tag, seed=21):
    """full-sample curve + driver bootstrap for every outcome x indicator; paired kappa_Y vs kappa_z test"""
    kpos, nk = des["kpos"], des["nk"]; idx_all = np.arange(nk); rng_bs = np.random.default_rng(seed)
    boot_idx = [rng_bs.integers(0, nk, nk) for _ in range(B)]                 # same resamples for every indicator/outcome
    # kappa and kappa_z on the design's drivers: point estimates and bootstrap replicates (paired with the curves)
    KZ = {}
    for c in TOP6:
        km, kz, s2, tz2 = kappas_on(des, c, idx_all)
        reps = np.array([kappas_on(des, c, bi)[:2] for bi in boot_idx])      # [B, 2]
        KZ[c] = dict(k_marg=km, k_z=kz, sigma2=s2, tau2_z=tz2, k_marg_boot=reps[:, 0], k_z_boot=reps[:, 1])
        print(f"  [{tag}] {c[:32]:32s} kappa(D)={km:.2f} kappa_z(D)={kz:.2f} [{pct(reps[:,1],2.5):.2f},{pct(reps[:,1],97.5):.2f}]  (all-N: kappa={kap[CLEAN.index(c)]:.2f}, kappa_z={kap_z[CLEAN.index(c)]:.2f})", flush=True)
    rows, curves = [], {}
    for oname, yv in outcomes.items():
        yk, Ek, Xk = yv[kpos], E[kpos], Xt[kpos]
        print(f"\n[{tag}] [{oname}] events among kept drivers: {int(yk.sum())}", flush=True)
        for c in TOP6:
            kz = KZ[c]["k_z"]; kzb = KZ[c]["k_z_boot"]
            s_full = curve(des, c, idx_all, yk, Ek, Xk)
            k_free, b_imp, chi = fit_kappa(s_full)
            f_vc = M_GRID / (M_GRID + kz); b_vc = (s_full * f_vc).sum() / (f_vc ** 2).sum()
            S = np.array([curve(des, c, bi, yk, Ek, Xk) for bi in boot_idx])          # [B, 7]
            d12 = S[:, -1] - S[:, 0]; kf = np.array([fit_kappa(s)[0] for s in S])
            d_full = s_full[-1] - s_full[0]; p_flat = two_sided_p(d12, B)
            # paired test of kappa_Y = kappa_z on the log scale (same resamples -> covariance enters)
            ldiff = np.log(kf) - np.log(kzb); ok = np.isfinite(ldiff)
            p_eq = two_sided_p(ldiff[ok], B) if ok.sum() > 10 else np.nan
            ratio, r_lo, r_hi = k_free / kz, np.exp(pct(ldiff[ok], 2.5)), np.exp(pct(ldiff[ok], 97.5))
            corr_lk = np.corrcoef(np.log(kf[ok]), np.log(kzb[ok]))[0, 1] if ok.sum() > 10 else np.nan
            rows.append(dict(design=tag, outcome=oname, indicator=c, n_drivers=nk, events=int(yk.sum()),
                             kappa_behaviour_allN=kap[CLEAN.index(c)], kappa_z_allN=kap_z[CLEAN.index(c)],
                             kappa_D=KZ[c]["k_marg"], kappa_z=kz, kappa_z_lo=pct(kzb, 2.5), kappa_z_hi=pct(kzb, 97.5),
                             kappa_crash=k_free, kappa_crash_lo=pct(kf, 2.5), kappa_crash_hi=pct(kf, 97.5),
                             ratio_kY_kz=ratio, ratio_lo=r_lo, ratio_hi=r_hi, p_kappa_equal=p_eq, corr_logk_boot=corr_lk,
                             slope_m1=s_full[0], slope_m12=s_full[-1], delta_12_1=d_full,
                             delta_lo=pct(d12, 2.5), delta_hi=pct(d12, 97.5), p_no_attenuation=p_flat, n_reversals=int((np.sign(d12) != np.sign(d_full)).sum()),
                             ratio_m12_m1=s_full[-1] / s_full[0] if s_full[0] != 0 else np.nan,
                             predicted_ratio=(12 / (12 + kz)) / (1 / (1 + kz)), beta_implied=b_vc, B=B))
            curves[(oname, c)] = (s_full, S, b_vc, kz)
            print(f"  {c[:32]:32s} s1={s_full[0]:+.3f} s12={s_full[-1]:+.3f} Δ CI [{pct(d12,2.5):+.3f},{pct(d12,97.5):+.3f}] p_flat={p_flat:.4f} "
                  f"| κ_z={kz:.1f} κ_Y={k_free:.1f} [{pct(kf,2.5):.1f},{pct(kf,97.5):.1f}] ratio={ratio:.2f} [{r_lo:.2f},{r_hi:.2f}] p_eq={p_eq:.2f} corr={corr_lk:.2f}", flush=True)
    return pd.DataFrame(rows), curves, KZ

# ================= main design: D = drivers with >= 12 rows, draws from all rows =================
keep_main = drv.index[drv.n_rows >= M_MAX]
des_main = build_design(keep_main, df, "D: >=12 rows, all rows")
res, curves, KZ_main = run_test(des_main, OUTCOMES, B_BOOT, "main")
res.to_csv(f"{OUT4}/tableB1_bootstrap_by_outcome.csv", index=False)
np.savez(f"{OUT4}/curves.npz", **{f"{o}||{c}||{k}": v for (o, c), (sf, S, b, kz) in curves.items() for k, v in [("s", sf), ("S", S), ("b", np.array([b])), ("kz", np.array([kz]))]})
print(f"main design done in {time.time()-t0:.0f}s", flush=True)

# ================= D_b designs: drivers with >= 12 baseline trips (rows - crashes - near-crashes) =================
keep_b = drv.index[drv.n_base_raw >= M_MAX]
des_b_all = build_design(keep_b, df, "D_b: >=12 baseline trips, all rows")
res_b_all, curves_b_all, _ = run_test(des_b_all, {"crashes (primary)": y_crash}, B_SENS, "D_b all rows", seed=22)
des_b_trim = build_design(keep_b, df[~df.likely_event], "D_b: >=12 baseline trips, trimmed rows")
res_b_trim, curves_b_trim, _ = run_test(des_b_trim, {"crashes (primary)": y_crash}, B_SENS, "D_b trimmed", seed=23)
pd.concat([res_b_all, res_b_trim]).to_csv(f"{OUT4}/tableB6_Db_designs.csv", index=False)
np.savez(f"{OUT4}/curves_Db.npz", **{f"{tag}||{c}||{k}": v for tag, cv in [("all", curves_b_all), ("trim", curves_b_trim)] for (o, c), (sf, S, b, kz) in cv.items() for k, v in [("s", sf), ("S", S), ("b", np.array([b])), ("kz", np.array([kz]))]})
print(f"D_b designs done in {time.time()-t0:.0f}s", flush=True)

# ---------- exposure robustness (primary outcome): miles offset, no mileage dummies ----------
print("\n[exposure] approximate-miles offset (mileage band midpoint x days), mileage dummies removed", flush=True)
kpos, nk = des_main["kpos"], des_main["nk"]; idx_all = np.arange(nk)
rows_e = []
for c in TOP6:
    s_days = curves[("crashes (primary)", c)][0]
    s_miles = curve(des_main, c, idx_all, y_crash[kpos], E_miles[kpos], Xt_nomileage[kpos])
    k_m, _, _ = fit_kappa(s_miles)
    rows_e.append(dict(indicator=c, kappa_z=KZ_main[c]["k_z"], kappa_crash_days_offset=fit_kappa(s_days)[0], kappa_crash_miles_offset=k_m,
                       s1_days=s_days[0], s12_days=s_days[-1], s1_miles=s_miles[0], s12_miles=s_miles[-1]))
    print(f"  {c[:32]:32s} days: s1={s_days[0]:+.3f} s12={s_days[-1]:+.3f} κ={rows_e[-1]['kappa_crash_days_offset']:.1f} | miles: s1={s_miles[0]:+.3f} s12={s_miles[-1]:+.3f} κ={k_m:.1f}", flush=True)
pd.DataFrame(rows_e).to_csv(f"{OUT4}/tableB2_exposure.csv", index=False)

# ---------- negative binomial check (primary outcome): slopes at m = 1, 4, 12 ----------
print("\n[negative binomial] NB2 vs Poisson slopes (primary outcome), mean over draws", flush=True)
rows_nb = []
yk, Ek, Xk = y_crash[kpos], E[kpos], Xt[kpos]
for c in TOP6:
    rec = dict(indicator=c)
    for m_ in [1, 4, 12]:
        sp, sn, alphas = [], [], []
        for d in range(N_DRAW):
            X1 = sm.add_constant(np.column_stack([Xk, des_main["XM"][c][m_][:, d]]))
            try:
                po = sm.GLM(yk, X1, family=sm.families.Poisson(), offset=np.log(Ek)).fit()
                mu = po.fittedvalues; a_hat = max(1e-3, ((yk - mu) ** 2 - mu).sum() / (mu ** 2).sum())
                nbm = sm.GLM(yk, X1, family=sm.families.NegativeBinomial(alpha=a_hat), offset=np.log(Ek)).fit()
                sp.append(po.params[-1]); sn.append(nbm.params[-1]); alphas.append(a_hat)
            except Exception as ex:
                pass
        rec[f"poisson_m{m_}"] = np.mean(sp); rec[f"nb_m{m_}"] = np.mean(sn); rec["nb_alpha"] = np.mean(alphas)
    rows_nb.append(rec); print(f"  {c[:32]:32s} " + "  ".join(f"m={m_}: P={rec[f'poisson_m{m_}']:+.3f} NB={rec[f'nb_m{m_}']:+.3f}" for m_ in [1, 4, 12]) + f"  α={rec['nb_alpha']:.2f}", flush=True)
pd.DataFrame(rows_nb).to_csv(f"{OUT4}/tableB3_negbin.csv", index=False)
X1 = sm.add_constant(Xk); po = sm.GLM(yk, X1, family=sm.families.Poisson(), offset=np.log(Ek)).fit()
phi = (po.resid_pearson ** 2).sum() / po.df_resid; print(f"  Pearson dispersion of the SPF (kept drivers): {phi:.2f}", flush=True)

# ---------- Buhlmann-Straub kappa with trip-duration weights, and the duration check ----------
print("\n[Buhlmann-Straub] kappa with trip-duration weights (minutes) vs trip-count kappa", flush=True)
rows_bs = []
dur = df["Trip Duration"].clip(lower=0.5)
for c in CLEAN:
    sub = df[["Participant_ID", c]].copy(); sub["w"] = dur.values; sub = sub.dropna()
    sub["wx"] = sub.w * sub[c]
    gg = sub.groupby("Participant_ID"); n_i = gg.size().values.astype(float); w_i = gg["w"].sum().values; xbar_i = (gg["wx"].sum() / gg["w"].sum()).values
    sub = sub.join(pd.Series(xbar_i, index=gg.size().index, name="xb"), on="Participant_ID")
    ssw = (sub.w * (sub[c] - sub.xb) ** 2).groupby(sub["Participant_ID"]).sum().values
    multi = n_i >= 2; sigma2 = ssw[multi].sum() / (n_i[multi] - 1).sum()
    W = w_i.sum(); xbar = (w_i * xbar_i).sum() / W; I = len(w_i)
    tau2 = ((w_i * (xbar_i - xbar) ** 2).sum() - (I - 1) * sigma2) / (W - (w_i ** 2).sum() / W)
    k_bs = sigma2 / tau2 if tau2 > 0 else np.inf                                  # in minutes of driving
    med_dur = float(np.nanmedian(dur))
    # duration check (M8): within-driver association between trip duration and the indicator
    dd = df[["Participant_ID", "Trip Duration", c]].dropna()
    dm = dd.groupby("Participant_ID")[[c, "Trip Duration"]].transform(lambda s: s - s.mean())
    r_within = dm.corr().iloc[0, 1]
    rows_bs.append(dict(indicator=c, kappa_trips=kap[CLEAN.index(c)], kappa_BS_minutes=k_bs, kappa_BS_in_trips_at_median_duration=k_bs / med_dur,
                        median_trip_minutes=med_dur, within_driver_corr_with_duration=r_within))
bs = pd.DataFrame(rows_bs).sort_values("kappa_trips"); bs.to_csv(f"{OUT4}/tableB4_buhlmann_straub.csv", index=False)
print(bs.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
sp_ = bs[["kappa_trips", "kappa_BS_in_trips_at_median_duration"]].replace(np.inf, np.nan).dropna()
print(f"  Spearman(trip-count kappa, BS kappa in trips) = {sp_.corr(method='spearman').iloc[0,1]:.3f}; median ratio = {np.median(sp_.iloc[:,1]/sp_.iloc[:,0]):.2f}")

# ---------- event-signature threshold sensitivity ----------
print("\n[screen threshold] indicators classified clean at each enrichment threshold (trip distance excluded)", flush=True)
thr_rows = []
for thr in [1.05, 1.08, 1.12]:
    clean_thr = [v for v in t1[(t1.enrichment <= thr) & (t1.status != "mass point (unusable)")].variable if v != "Trip Distance Origin to Destination"]
    moved = sorted(set(clean_thr) ^ set(CLEAN)); thr_rows.append(dict(threshold=thr, n_clean=len(clean_thr), differs=";".join(moved)))
    print(f"  threshold {thr}: {len(clean_thr)} clean; differs from main set by: {moved if moved else 'none'}")
pd.DataFrame(thr_rows).to_csv(f"{OUT4}/tableB7_threshold.csv", index=False)

# ---------- Table 6 by outcome: naive vs calibrated (covariate-conditional, with offset term), C_z at the harmonic-mean evidence ----------
def rc_fit_irls(Xcal, j, yv, Ev, iters=3):
    off = np.zeros(len(yv)); b = np.nan
    for _ in range(iters):
        b = irls_slope(np.column_stack([Xt, Xcal[:, j]]), yv, Ev * np.exp(off)); off = 0.5 * b ** 2 * tau2z_std[j] * (1 - Rz[:, j])
    return b, float(np.abs(off).max())
rows4 = []
for oname, yv in [("crashes (primary)", y_crash), ("crashes + near-crashes (secondary)", y_all)]:
    for c in TOP6:
        j = CLEAN.index(c)
        b_naive = irls_slope(np.column_stack([Xt, Xb[:, j]]), yv, E)
        b_rc_marg = irls_slope(np.column_stack([Xt, Xb_rc[:, j]]), yv, E)
        b_rc, off_max = rc_fit_irls(Xb_rcz, j, yv, E)
        hn = 1 / np.mean(1 / nk_mat[:, j]); Cz_hn = hn / (hn + kap_z[j])
        rows4.append(dict(outcome=oname, indicator=c, kappa=kap[j], kappa_z=kap_z[j], harmonic_mean_n=hn, credibility_z_at_harmonic_mean_n=Cz_hn,
                          mean_credibility_z=Rz[:, j].mean(), slope_naive=b_naive, slope_rc_marginal=b_rc_marg, slope_regression_calibrated=b_rc,
                          max_offset_term=off_max, understatement_pct=100 * (1 - b_naive / b_rc) if b_rc != 0 else np.nan,
                          predicted_understatement_pct=100 * (1 - Cz_hn)))
t4b = pd.DataFrame(rows4); t4b.to_csv(f"{OUT4}/tableB5_deattenuation_by_outcome.csv", index=False)
print("\n[Table 6 by outcome]"); print(t4b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------- design facts for the text ----------
facts = dict(n_drivers=int(len(drv)), n_rows=int(len(df)), rows_median=float(drv.n_rows.median()), rows_q25=float(drv.n_rows.quantile(.25)), rows_q75=float(drv.n_rows.quantile(.75)), rows_max=int(drv.n_rows.max()),
             base_median=float(drv.n_base_raw.median()), base_q25=float(drv.n_base_raw.quantile(.25)), base_q75=float(drv.n_base_raw.quantile(.75)), base_max=int(drv.n_base_raw.max()),
             drivers_more_events_than_rows=int((drv.n_base_raw < 0).sum()), drivers_events_eq_rows=int((drv.n_base_raw == 0).sum()),
             D_drivers=int(des_main["nk"]), D_crashes=int(y_crash[des_main["kpos"]].sum()), D_all_events=int(y_all[des_main["kpos"]].sum()), D_atfault=int(y_fault[des_main["kpos"]].sum()),
             Db_drivers=int(des_b_all["nk"]), Db_crashes=int(y_crash[des_b_all["kpos"]].sum()), Db_trimmed_rows=int(len(df[~df.likely_event][df[~df.likely_event]["Participant_ID"].isin(keep_b)])),
             event_rows_share_all=float((drv.crashes + drv.near).sum() / drv.n_rows.sum()), event_rows_share_D=float(y_all[des_main["kpos"]].sum() / drv.n_rows.values[des_main["kpos"]].sum()),
             pearson_dispersion_SPF_D=float(phi), B_BOOT=B_BOOT, B_SENS=B_SENS, median_trip_minutes=float(np.nanmedian(dur)))
json.dump(facts, open(f"{OUT4}/facts.json", "w"), indent=1); print(facts)

# ---------- figure: collapse with bootstrap bands, primary and secondary (x = m / kappa_z on D) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, oname in zip(axes, ["crashes (primary)", "crashes + near-crashes (secondary)"]):
    for c, mk, col in zip(TOP6, "osD^v<", [ORA, BLU, PUR, GRN, GRY, "#B5651D"]):
        s_full, S, b_vc, kz = curves[(oname, c)]
        d12 = S[:, -1] - S[:, 0]
        if abs(b_vc) < 1e-6 or pct(d12, 2.5) < 0 < pct(d12, 97.5):
            ax.plot([], [], mk, color=col, label=f"{c}: no detectable association with this outcome (omitted)"); continue
        lo, hi = np.percentile(S, 2.5, axis=0) / b_vc, np.percentile(S, 97.5, axis=0) / b_vc
        ax.errorbar(M_GRID / kz, s_full / b_vc, yerr=[np.clip(s_full / b_vc - np.minimum(lo, hi), 0, None), np.clip(np.maximum(lo, hi) - s_full / b_vc, 0, None)],
                    fmt=mk, ms=5, color=col, capsize=3, lw=1, alpha=.9, label=f"{c} (κ_z={kz:.1f})")
    xx = np.logspace(-1.4, 1.3, 300); ax.plot(xx, xx / (1 + xx), "k--", lw=1.5, label="C(x) = x/(1+x)")
    ax.set_xscale("log"); ax.set_xlim(0.04, 20); ax.set_ylim(-0.1, 1.4); ax.set_title(f"{oname}: {int(OUTCOMES[oname][kpos].sum())} events, {nk} drivers", fontsize=10)
    ax.set_xlabel("x = m / κ_z (log)"); ax.legend(fontsize=6.5, loc="upper left")
axes[0].set_ylabel("observed slope ÷ full-information slope")
fig.tight_layout(); fig.savefig(f"{OUT4}/figB_bootstrap_collapse.png", dpi=160); fig.savefig(f"{OUT4}/figB_bootstrap_collapse.pdf"); plt.close(fig)
print(f"\ndone in {time.time()-t0:.0f}s")
