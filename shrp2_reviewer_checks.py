# ==================================================================
# Reviewer checks (Nov 2026):
#   R1  crashes excluding severity IV (tyre strikes), decoded from the driver-level
#       Max_Severity / Mean_Severity summaries (coding provisional: 0 = IV, 3/4 = III/II)
#       -> Table 4 analogue (raw vs calibrated slopes), Table 5 analogue (linear models),
#          within-driver test (Table 3 analogue, B_SENS resamples)
#   R2  slopes at 1 and 12 trips on D_b with and without the likely event trips,
#       PAIRED driver bootstrap (same resamples for both designs)
#   R3  between-driver credibility parameter kappa_B: all-driver crash model with slope
#       b * n_i/(n_i + kappa_B) on the raw standardised mean, profile over kappa, driver
#       bootstrap paired with kappa_z
#   R4  moved to shrp2_r7_final_checks.py (crash rate by fifth of the behavioural scores)
# Builds on the data prep of shrp2_prelim.py (exec of its head) and the design
# machinery of shrp2_robustness.py.
# ==================================================================
import io, contextlib, os, sys, time, json
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor

t0 = time.time()
os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read()
head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
OUTR = f"{OUT}/reviewer"; os.makedirs(OUTR, exist_ok=True)
B_SENS = int(os.environ.get("B_SENS", 500)); B_KB = int(os.environ.get("B_KB", 500))
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s",
        "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
M_GRID, M_MAX, N_DRAW = np.array([1, 2, 3, 4, 6, 8, 12]), 12, 6
def log(*a):
    print(*a, flush=True)

def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m

def irls(X, y, E, alpha=1e-3, iters=30):
    """ridge Poisson IRLS matched to sklearn's alpha scaling; returns (beta, deviance)"""
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
    mu = np.exp(np.clip(X1 @ beta + logE, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
    return beta, dev

def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; return np.percentile(a, q) if len(a) else np.nan
def two_sided_p(x, B):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    p = 2 * min((x <= 0).mean(), (x >= 0).mean()); return min(1.0, max(p, 2.0 / (B + 1)))

y_crash = drv.crashes.values.astype(float)
J = {c: CLEAN.index(c) for c in CLEAN}

# ==================================================================
# R1. crashes excluding severity IV: decode the driver-level severity summaries
# ==================================================================
log("\n=== R1: decode severity summaries ===")
sev_cols = ["Participant_ID", "N_events_total", "N_crashes", "N_near_crashes", "Max_Severity", "Mean_Severity"]
sv = df[sev_cols].sort_values("N_events_total").groupby("Participant_ID").tail(1).set_index("Participant_ID").reindex(drv.index)
sv["sumsev"] = sv.Mean_Severity * sv.N_events_total
sv["S_c"] = sv.sumsev - sv.N_near_crashes                      # near-crash code = 1 each, baseline rows 0
sv["crashes"] = drv.crashes
mismatch = int((sv.N_crashes != sv.crashes).sum())
log(f"drivers whose severity record carries a different crash count than the max used in the paper: {mismatch}")
def non_iv(Sc, k):
    """number of crashes with a non-zero code, given the crash-code sum Sc and k crashes (codes 0, 3, 4; 1 tolerated)"""
    if k == 0: return 0, False
    Sc = max(Sc, 0.0); cands = [k1 for k1 in range(k + 1) if 3 * k1 - 0.6 <= Sc <= 4 * k1 + 0.6]
    if not cands:                                             # a code-1 crash or a non-unit near-crash code: nearest
        return int(min(k, round(Sc / 3.5))), True
    return (min(cands, key=lambda k1: abs(Sc - 3.5 * k1)), len(cands) > 1)
dec = [non_iv(r.S_c, int(r.crashes)) for r in sv.itertuples()]
sv["crashes_non_iv"] = [d[0] for d in dec]; sv["ambiguous"] = [d[1] for d in dec]
sv["crashes_iv"] = sv.crashes - sv.crashes_non_iv
y_sev = sv.crashes_non_iv.values.astype(float)
log(f"crashes: {int(drv.crashes.sum())}; decoded severity-IV (code 0): {int(sv.crashes_iv.sum())} ({100*sv.crashes_iv.sum()/drv.crashes.sum():.0f}%); "
    f"remaining: {int(y_sev.sum())}; drivers with an ambiguous decomposition: {int(sv.ambiguous.sum())} of {int((sv.crashes>0).sum())} with crashes")
sv[["crashes", "crashes_non_iv", "crashes_iv", "ambiguous", "Max_Severity", "Mean_Severity", "N_events_total", "N_near_crashes"]].to_csv(f"{OUTR}/severity_decode.csv")

# ---- R1a: Table 4 analogue (raw vs covariate-conditional regression calibration), both outcomes
def rc_fit(Xcal, j, yv, Ev, iters=3):
    off = np.zeros(len(yv)); b = np.nan
    for _ in range(iters):
        m = pois_fit(np.column_stack([Xt, Xcal[:, j]]), yv, Ev * np.exp(off)); b = m.coef_[-1]
        off = 0.5 * b ** 2 * tau2z_std[j] * (1 - Rz[:, j])
    return b
rows = []
for c in CLEAN:
    j = J[c]; rec = dict(indicator=c, kappa_z=kap_z[j], n_harmonic=1 / np.mean(1 / nk_mat[:, j]))
    for tag, yv in [("all", y_crash), ("exIV", y_sev)]:
        b_raw = pois_fit(np.column_stack([Xt, Xb[:, j]]), yv, E).coef_[-1]
        b_rc = rc_fit(Xb_rcz, j, yv, E)
        # bootstrap SE of the raw slope (driver resampling, 200 reps) for the sign question
        rng = np.random.default_rng(3); bs = []
        for b_ in range(200):
            ii = rng.integers(0, len(yv), len(yv)); bs.append(irls(np.column_stack([Xt[ii], Xb[ii, j]]), yv[ii], E[ii])[0][-1])
        rec.update({f"raw_{tag}": b_raw, f"raw_se_{tag}": float(np.std(bs)), f"rc_{tag}": b_rc,
                    f"under_{tag}": 100 * (1 - b_raw / b_rc) if b_rc != 0 else np.nan})
    rows.append(rec)
t4s = pd.DataFrame(rows); t4s.to_csv(f"{OUTR}/tableR1_deattenuation_by_severity.csv", index=False)
log("\nR1a: raw and calibrated slopes (per SD), all crashes vs crashes excluding severity IV")
log(t4s[["indicator", "raw_all", "raw_se_all", "rc_all", "under_all", "raw_exIV", "raw_se_exIV", "rc_exIV", "under_exIV"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- R1b: Table 5 analogue, linear models only, five 75/25 splits, both outcomes
def dev_mean(eta, yv, Ev):
    mu = Ev * np.exp(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(2 * (np.where(yv > 0, yv * np.log(yv / mu), 0) - (yv - mu))))
recs = []
for tag, yv in [("all", y_crash), ("exIV", y_sev)]:
    for split in range(5):
        rng = np.random.default_rng(100 + split); idx = rng.permutation(len(yv)); nte = len(yv) // 4
        te, tr = idx[:nte], idx[nte:]
        spf = pois_fit(Xt[tr], yv[tr], E[tr], alpha=1e-2)
        lin = pois_fit(np.column_stack([Xt[tr], Xb[tr]]), yv[tr], E[tr], alpha=1e-2)
        rc = pois_fit(np.column_stack([Xt[tr], Xb_rcz[tr]]), yv[tr], E[tr], alpha=1e-2)
        preds = {"baseline": spf.intercept_ + Xt[te] @ spf.coef_,
                 "raw means": lin.intercept_ + np.column_stack([Xt[te], Xb[te]]) @ lin.coef_,
                 "credibility-weighted": rc.intercept_ + np.column_stack([Xt[te], Xb_rcz[te]]) @ rc.coef_}
        for name, eta in preds.items():
            recs.append(dict(outcome=tag, split=split, model=name, dev=dev_mean(eta, yv[te], E[te]),
                             **{f"dev_{s}": dev_mean(eta[strat[te] == si], yv[te][strat[te] == si], E[te][strat[te] == si]) for si, s in enumerate(["low", "mid", "high"])}))
t5s = pd.DataFrame(recs).groupby(["outcome", "model"], sort=False).mean().drop(columns="split")
t5s.to_csv(f"{OUTR}/tableR1_models_by_severity.csv")
log("\nR1b: test deviance (mean over 5 splits) by evidence tertile")
log(t5s.to_string(float_format=lambda v: f"{v:.4f}"))
for tag in ["all", "exIV"]:
    b_ = t5s.loc[(tag, "baseline")]
    for mname in ["raw means", "credibility-weighted"]:
        r_ = t5s.loc[(tag, mname)]
        log(f"  [{tag}] {mname:22s} gain over baseline: all {100*(1-r_.dev/b_.dev):+.1f}%  low {100*(1-r_.dev_low/b_.dev_low):+.1f}%  mid {100*(1-r_.dev_mid/b_.dev_mid):+.1f}%  high {100*(1-r_.dev_high/b_.dev_high):+.1f}%")

# ==================================================================
# design machinery (from shrp2_robustness.py) for R1c and R2
# ==================================================================
SIG = {"Maximum Deceleration": -1, "ABS Activation": 1, "Maximum Acceleration": 1, "Number of Lateral Accels > Threshold": 1,
       "Maximum Lateral Acceleration": 1, "Time Where Headway 0.0 - 0.5 s": 1, "Time Where Headway 0.5 - 1.0 s": 1, "Brake Activations": 1}
Zsig = np.zeros(len(df))
for c, sgn in SIG.items():
    v = df[c].astype(float); Zsig += sgn * ((v - v.mean()) / (v.std() + 1e-9)).fillna(0).values
df["sig_index"] = Zsig / len(SIG)
k_ev = (drv.crashes + drv.near).reindex(df["Participant_ID"]).values
df["rank_in_driver"] = df.groupby("Participant_ID")["sig_index"].rank(ascending=False, method="first")
df["likely_event"] = df["rank_in_driver"] <= k_ev

def build_design(keep_ids, rows, label, inds=TOP6, seed=5):
    kpos = np.where(drv.index.isin(keep_ids))[0]; nk = len(kpos); rng = np.random.default_rng(seed)
    XM, SUMM = {}, {}
    sub_all = rows[rows["Participant_ID"].isin(keep_ids)]
    for c in inds:
        sub = sub_all[["Participant_ID", c]]
        grp = sub.groupby("Participant_ID")[c].apply(lambda s: s.values).reindex(keep_ids); maxn = max(len(a) for a in grp)
        pad = np.full((nk, maxn), np.nan)
        for r_, a in enumerate(grp): pad[r_, :len(a)] = a
        mu_c, sd_c = np.nanmean(beh[c]), np.nanstd(beh[c]) + 1e-9
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
    log(f"[{label}] drivers: {nk}; rows: {len(sub_all)}; precompute done at {time.time()-t0:.0f}s")
    return dict(label=label, kpos=kpos, nk=nk, XM=XM, SUMM=SUMM)

def kappas_on(des, c, idx):
    n_i, m_i, ssw_i = des["SUMM"][c]; Zk = Xt[des["kpos"]]
    n, m, ssw, Z = n_i[idx], m_i[idx], ssw_i[idx], Zk[idx]
    ok = n >= 1; n, m, ssw, Z = n[ok], m[ok], ssw[ok], Z[ok]
    return vc_z_from_summaries(n, m, ssw, Z)[2]

def slope_m(des, c, m_, idx, yv, Ev, Xtv):
    XM = des["XM"][c][m_]
    return np.mean([irls(np.column_stack([Xtv[idx], XM[idx, d]]), yv[idx], Ev[idx])[0][-1] for d in range(N_DRAW)])

def curve(des, c, idx, yv, Ev, Xtv):
    return np.array([slope_m(des, c, m_, idx, yv, Ev, Xtv) for m_ in M_GRID])

KG = np.exp(np.linspace(np.log(0.05), np.log(300), 300))
def fit_kappa(s):
    best = None
    for kk in KG:
        f = M_GRID / (M_GRID + kk); bb = (s * f).sum() / (f ** 2).sum(); chi = ((s - bb * f) ** 2).sum()
        if best is None or chi < best[0]: best = (chi, kk, bb)
    return best[1], best[2]

# ==================================================================
# R1c. within-driver test for crashes excluding severity IV (Table 3 analogue), B_SENS resamples
# ==================================================================
log("\n=== R1c: within-driver test, crashes excluding severity IV ===")
keep_main = drv.index[drv.n_rows >= M_MAX]
des_main = build_design(keep_main, df, "D: >=12 rows, all rows")
kpos, nk = des_main["kpos"], des_main["nk"]; idx_all = np.arange(nk)
rng_bs = np.random.default_rng(21); boot_idx = [rng_bs.integers(0, nk, nk) for _ in range(B_SENS)]
yk, Ek, Xk = y_sev[kpos], E[kpos], Xt[kpos]
log(f"events among the {nk} drivers of D: {int(yk.sum())} non-IV crashes (all crashes: {int(y_crash[kpos].sum())})")
rows = []
for c in TOP6:
    kz = kappas_on(des_main, c, idx_all); kzb = np.array([kappas_on(des_main, c, bi) for bi in boot_idx])
    s_full = curve(des_main, c, idx_all, yk, Ek, Xk); k_free, b_imp = fit_kappa(s_full)
    S = np.array([curve(des_main, c, bi, yk, Ek, Xk) for bi in boot_idx])
    d12 = S[:, -1] - S[:, 0]; kf = np.array([fit_kappa(s)[0] for s in S])
    ldiff = np.log(kf) - np.log(kzb)
    rows.append(dict(indicator=c, n_drivers=nk, events=int(yk.sum()), slope_m1=s_full[0], slope_m12=s_full[-1],
                     delta_12_1=s_full[-1] - s_full[0], delta_lo=pct(d12, 2.5), delta_hi=pct(d12, 97.5), p_no_attenuation=two_sided_p(d12, B_SENS),
                     kappa_z=kz, kappa_crash=k_free, kappa_crash_lo=pct(kf, 2.5), kappa_crash_hi=pct(kf, 97.5),
                     ratio_kY_kz=k_free / kz, ratio_lo=np.exp(pct(ldiff, 2.5)), ratio_hi=np.exp(pct(ldiff, 97.5)), p_kappa_equal=two_sided_p(ldiff, B_SENS), B=B_SENS))
    r = rows[-1]
    log(f"  {c[:32]:32s} s1={r['slope_m1']:+.3f} s12={r['slope_m12']:+.3f} Δ=[{r['delta_lo']:+.3f},{r['delta_hi']:+.3f}] p_flat={r['p_no_attenuation']:.3f} | κ_z={kz:.1f} κ_Y={k_free:.1f} [{r['kappa_crash_lo']:.1f},{r['kappa_crash_hi']:.1f}] ratio={r['ratio_kY_kz']:.2f} [{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] p_eq={r['p_kappa_equal']:.2f}")
pd.DataFrame(rows).to_csv(f"{OUTR}/tableR1_within_exIV.csv", index=False)
log(f"R1c done at {time.time()-t0:.0f}s")

# ==================================================================
# R2. D_b: slopes at 1 and 12 trips with and without the likely event trips, PAIRED bootstrap
# ==================================================================
log("\n=== R2: D_b, all rows vs likely event trips removed, paired bootstrap ===")
keep_b = drv.index[drv.n_base_raw >= M_MAX]
des_all = build_design(keep_b, df, "D_b all rows", seed=5)
des_trim = build_design(keep_b, df[~df.likely_event], "D_b trimmed", seed=5)
kb, nkb = des_all["kpos"], des_all["nk"]; idx_b = np.arange(nkb)
rng_bs = np.random.default_rng(22); boot_b = [rng_bs.integers(0, nkb, nkb) for _ in range(B_SENS)]
yb, Eb, Xb_t = y_crash[kb], E[kb], Xt[kb]
rows = []
for c in TOP6:
    rec = dict(indicator=c, n_drivers=nkb, crashes=int(yb.sum()))
    for m_ in [1, 12]:
        a_full = slope_m(des_all, c, m_, idx_b, yb, Eb, Xb_t); t_full = slope_m(des_trim, c, m_, idx_b, yb, Eb, Xb_t)
        diff = np.array([slope_m(des_trim, c, m_, bi, yb, Eb, Xb_t) - slope_m(des_all, c, m_, bi, yb, Eb, Xb_t) for bi in boot_b])
        rec.update({f"s{m_}_all": a_full, f"s{m_}_trim": t_full, f"s{m_}_diff": t_full - a_full,
                    f"s{m_}_diff_lo": pct(diff, 2.5), f"s{m_}_diff_hi": pct(diff, 97.5), f"s{m_}_p": two_sided_p(diff, B_SENS)})
    rows.append(rec)
    log(f"  {c[:32]:32s} m=12: all {rec['s12_all']:+.3f} trimmed {rec['s12_trim']:+.3f} diff {rec['s12_diff']:+.3f} [{rec['s12_diff_lo']:+.3f},{rec['s12_diff_hi']:+.3f}] p={rec['s12_p']:.2f} | m=1: {rec['s1_all']:+.3f} vs {rec['s1_trim']:+.3f}")
pd.DataFrame(rows).to_csv(f"{OUTR}/tableR2_event_trips_paired.csv", index=False)
log(f"R2 done at {time.time()-t0:.0f}s")

# ==================================================================
# R3. between-driver credibility parameter kappa_B (all 3,110 drivers)
# ==================================================================
log("\n=== R3: between-driver kappa_B, profile over kappa, driver bootstrap paired with kappa_z ===")
KGB = np.exp(np.linspace(np.log(0.05), np.log(300), 45))
NBASE_SCALE = (drv.n_base.values / drv.n_rows.values)         # share of a driver's rows that are baseline trips
EVID = {"rows": nk_mat, "baseline": np.clip(nk_mat * NBASE_SCALE[:, None], 1, None)}
def profile_kappa(j, idx, yv, ev):
    """deviance profile of the all-driver model with slope b*C_kappa(n_i) on the raw standardised mean"""
    n_ = ev[idx, j]; x_ = Xb[idx, j]; Xt_ = Xt[idx]; y_ = yv[idx]; E_ = E[idx]
    devs, bs = [], []
    for kk in KGB:
        w = n_ / (n_ + kk) * x_
        beta, dv = irls(np.column_stack([Xt_, w]), y_, E_); devs.append(dv); bs.append(beta[-1])
    devs = np.array(devs); i = int(np.argmin(devs))
    return KGB[i], bs[i], devs[i], devs[0], devs[-1]
rows = []; profiles = {}
rng_bs = np.random.default_rng(31); N = len(y_crash); boot_all = [rng_bs.integers(0, N, N) for _ in range(B_KB)]
for evname, ev in EVID.items():
  log(f"-- evidence definition: {evname}")
  for c in TOP6:
      j = J[c]
      n_i, m_i, ssw_i, ids_i = driver_summaries(c); pos_i = pos_of[ids_i].values
      full_n = np.zeros(N); full_m = np.zeros(N); full_s = np.zeros(N); full_n[pos_i] = n_i; full_m[pos_i] = m_i; full_s[pos_i] = ssw_i
      def kz_on(idx):
          ok = full_n[idx] >= 1; ii = idx[ok]
          return vc_z_from_summaries(full_n[ii], full_m[ii], full_s[ii], Xt[ii])[2]
      kB, bB, dev_best, dev_k0, dev_kmax = profile_kappa(j, np.arange(N), y_crash, ev)
      kz_full = kz_on(np.arange(N))
      reps = []
      for bi in boot_all:
          kb_, bb_, d_, d0_, dm_ = profile_kappa(j, bi, y_crash, ev); reps.append((kb_, kz_on(bi)))
      reps = np.array(reps); ldiff = np.log(reps[:, 0]) - np.log(reps[:, 1])
      rows.append(dict(evidence=evname, indicator=c, kappa_z_allN=kap_z[j], kappa_z_recomputed=kz_full, kappa_B=kB, kappa_B_lo=pct(reps[:, 0], 2.5), kappa_B_hi=pct(reps[:, 0], 97.5),
                       share_boot_at_lower_bound=float(np.mean(reps[:, 0] <= KGB[0] * 1.001)), share_boot_at_upper_bound=float(np.mean(reps[:, 0] >= KGB[-1] * 0.999)),
                       ratio_kB_kz=kB / kz_full, ratio_lo=np.exp(pct(ldiff, 2.5)), ratio_hi=np.exp(pct(ldiff, 97.5)), p_kappa_equal=two_sided_p(ldiff, B_KB),
                       slope_b_at_kB=bB, dev_gain_vs_no_attenuation=dev_k0 - dev_best, dev_gain_vs_kmax=dev_kmax - dev_best, B=B_KB))
      r = rows[-1]
      log(f"  {c[:32]:32s} κ_z={kz_full:.1f}  κ_B={kB:.1f} [{r['kappa_B_lo']:.1f},{r['kappa_B_hi']:.1f}] (boot at bounds: {r['share_boot_at_lower_bound']:.2f} low, {r['share_boot_at_upper_bound']:.2f} high)  ratio={r['ratio_kB_kz']:.2f} [{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] p_eq={r['p_kappa_equal']:.2f}  Δdev vs κ→0: {r['dev_gain_vs_no_attenuation']:.2f}, vs κ=300: {r['dev_gain_vs_kmax']:.2f}")
pd.DataFrame(rows).to_csv(f"{OUTR}/tableR3_kappa_between.csv", index=False)
log(f"R3 done at {time.time()-t0:.0f}s")

# ==================================================================
# R4 (crash rate by fifth of the two behavioural scores) is now in shrp2_r7_final_checks.py:
# the intervals there come from a bootstrap of drivers, because a driver is in the test set of more than one split.
# ==================================================================
log(f"\nall done at {time.time()-t0:.0f}s")
