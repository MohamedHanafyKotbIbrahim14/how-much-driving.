# ==================================================================
# Final checks (Nov 2026). Builds on the data prep of shrp2_prelim.py (exec of its head).
#   R4   crash rate by fifth of the behavioural score on the test drivers of the five 75/25 splits:
#        score from raw means, from credibility-weighted means (evidence = rows) and from
#        credibility-weighted means with the evidence counted as baseline trips.
#        Intervals and p-values come from a bootstrap of DRIVERS, not of test records: a driver is in
#        the test set of 1.25 splits on average, so the records of a driver are resampled together.
#        (Replaces block R4 of shrp2_reviewer_checks.py, which resampled the records.)
#   R7a  ranking of all drivers (Figure 5) with the weights from baseline trips
#   R7b  order of the three groups of indicators: one bootstrap of the drivers for all sixteen kappas
#   R7c  severity coding: shares of the codes, and share at fault by code, among drivers with one crash
#   R7d  crashes plus near-crashes on D_b: slopes at one and twelve trips with and without the likely
#        event trips, paired bootstrap (the two indicators that change sign between the outcomes)
#   R7e  rows on which each indicator is missing (why the median evidence is below the median rows)
#   R7f  is typical behaviour stable over the study? correlation of the means of each driver's first and last
#        five trips (time order) against two random, disjoint sets of five trips, drivers with at least ten trips
#   R7g  trips to half-credibility with the estimator of the paper (equations sigma, tau) and with the
#        between-driver variance weighted by the drivers' trips (Buhlmann-Gisler eq. 4.27, same within variance)
# ==================================================================
import io, contextlib, os, time, json
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor

t0 = time.time()
os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read()
head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
OUTR = f"{OUT}/reviewer"; os.makedirs(OUTR, exist_ok=True)
B_Q = int(os.environ.get("B_Q", 2000)); B_SENS = int(os.environ.get("B_SENS", 500)); B_GRP = int(os.environ.get("B_GRP", 1000))
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s",
        "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
def log(*a): print(*a, flush=True)
def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m
def pct(a, q):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; return np.percentile(a, q) if len(a) else np.nan
def two_sided_p(x, B):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    p = 2 * min((x <= 0).mean(), (x >= 0).mean()); return min(1.0, max(p, 2.0 / (B + 1)))

y_crash = drv.crashes.values.astype(float); y_all = (drv.crashes + drv.near).values.astype(float); N = len(y_crash); p_t = Xt.shape[1]
# credibility weights with the evidence counted as baseline trips (as in shrp2_r5_baseline_weights.py)
scale_b = (drv.n_base.values / drv.n_rows.values)[:, None]
nk_base = np.clip(nk_mat * scale_b, 1, None); Rz_b = nk_base / (nk_base + kap_z[None, :]); Xb_rcz_b = Mhat + Rz_b * (Xb - Mhat)

# ==================================================================
# R4. crash rate by fifth of the three scores, test drivers of the five splits; driver bootstrap
# ==================================================================
log("=== R4: crash rate by fifth of the raw and the credibility-weighted scores (test drivers), bootstrap of drivers ===")
recs = []
for split in range(5):
    rng = np.random.default_rng(100 + split); idx = rng.permutation(N); nte = N // 4
    te, tr = idx[:nte], idx[nte:]
    b_n = pois_fit(np.column_stack([Xt[tr], Xb[tr]]), y_crash[tr], E[tr], alpha=1e-2).coef_[p_t:]
    b_r = pois_fit(np.column_stack([Xt[tr], Xb_rcz[tr]]), y_crash[tr], E[tr], alpha=1e-2).coef_[p_t:]
    b_b = pois_fit(np.column_stack([Xt[tr], Xb_rcz_b[tr]]), y_crash[tr], E[tr], alpha=1e-2).coef_[p_t:]
    recs.append(pd.DataFrame(dict(split=split, driver=te, y=y_crash[te], E=E[te], tert=strat[te],
                                  s_raw=Xb[te] @ b_n, s_rc=Xb_rcz[te] @ b_r, s_rcb=Xb_rcz_b[te] @ b_b)))
tst = pd.concat(recs, ignore_index=True)
def fifths(s): return pd.qcut(s.rank(method="first"), 5, labels=False).astype(int)
SCORES = {"raw": "s_raw", "rc": "s_rc", "rcb": "s_rcb"}
for k, col in SCORES.items():
    tst[f"q_{k}"] = tst.groupby("split")[col].transform(fifths)
    for t_ in range(3):
        sel = tst.tert == t_; tst.loc[sel, f"qt_{k}"] = tst[sel].groupby("split")[col].transform(fifths)
def rate_table(d, qcol):
    g_ = d.groupby(qcol).agg(crashes=("y", "sum"), years=("E", "sum"), drivers=("y", "size"))
    g_["rate_per_100y"] = 100 * g_.crashes / g_.years; g_["share_of_crashes"] = g_.crashes / g_.crashes.sum(); return g_
def driver_boot(d, qa, qb, B, seed=41):
    """bootstrap of drivers for (rate in the highest fifth of score b) - (rate in the highest fifth of score a),
       and for the difference of the ratios highest / lowest fifth; the records of a driver stay together"""
    ids, inv = np.unique(d.driver.values, return_inverse=True); nd = len(ids)
    def per_driver(mask): return np.bincount(inv, weights=np.where(mask, d.y.values, 0.0), minlength=nd), np.bincount(inv, weights=np.where(mask, d.E.values, 0.0), minlength=nd)
    ya5, ea5 = per_driver(d[qa].values == 4); yb5, eb5 = per_driver(d[qb].values == 4)
    ya1, ea1 = per_driver(d[qa].values == 0); yb1, eb1 = per_driver(d[qb].values == 0)
    rng = np.random.default_rng(seed); dtop, dratio = np.empty(B), np.empty(B)
    for b_ in range(B):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
        ra5, rb5 = 100 * (w @ ya5) / (w @ ea5), 100 * (w @ yb5) / (w @ eb5); ra1, rb1 = 100 * (w @ ya1) / (w @ ea1), 100 * (w @ yb1) / (w @ eb1)
        dtop[b_] = rb5 - ra5; dratio[b_] = rb5 / rb1 - ra5 / ra1
    return dtop, dratio, nd
out, summ_rows = {}, []
SCOPES = [("all drivers", tst, "q")] + [(f"evidence tertile {['low', 'mid', 'high'][t_]}", tst[tst.tert == t_], "qt") for t_ in range(3)]
for scope, d, qp in SCOPES:
    T_ = {k: rate_table(d, f"{qp}_{k}") for k in SCORES}
    tab = pd.DataFrame({"fifth": ["1 (lowest)", "2", "3", "4", "5 (highest)"],
                        "raw: crashes/100 driver-years": T_["raw"].rate_per_100y.values, "weighted: crashes/100 driver-years": T_["rc"].rate_per_100y.values,
                        "weighted (baseline trips): crashes/100 driver-years": T_["rcb"].rate_per_100y.values,
                        "raw: share of crashes": T_["raw"].share_of_crashes.values, "weighted: share of crashes": T_["rc"].share_of_crashes.values,
                        "weighted (baseline trips): share of crashes": T_["rcb"].share_of_crashes.values})
    dtop, dratio, nd = driver_boot(d, f"{qp}_raw", f"{qp}_rc", B_Q); dtop_b, dratio_b, _ = driver_boot(d, f"{qp}_raw", f"{qp}_rcb", B_Q)
    top = lambda k: T_[k].rate_per_100y.iloc[-1]; bot = lambda k: T_[k].rate_per_100y.iloc[0]; shr = lambda k: T_[k].share_of_crashes.iloc[-1]
    s_ = dict(scope=scope, drivers=int(len(d)), distinct_drivers=int(nd), records_per_driver=len(d) / nd, crashes=int(d.y.sum()),
              top_rate_raw=top("raw"), top_rate_rc=top("rc"), top_diff=top("rc") - top("raw"), top_diff_lo=pct(dtop, 2.5), top_diff_hi=pct(dtop, 97.5), top_diff_p=two_sided_p(dtop, B_Q),
              ratio_top_bottom_raw=top("raw") / bot("raw"), ratio_top_bottom_rc=top("rc") / bot("rc"), ratio_diff_lo=pct(dratio, 2.5), ratio_diff_hi=pct(dratio, 97.5), ratio_diff_p=two_sided_p(dratio, B_Q),
              top_share_raw=shr("raw"), top_share_rc=shr("rc"),
              top_rate_rcb=top("rcb"), top_diff_b=top("rcb") - top("raw"), top_diff_b_lo=pct(dtop_b, 2.5), top_diff_b_hi=pct(dtop_b, 97.5), top_diff_b_p=two_sided_p(dtop_b, B_Q),
              ratio_top_bottom_rcb=top("rcb") / bot("rcb"), top_share_rcb=shr("rcb"), B=B_Q)
    sp = [(rate_table(d[d.split == k_], f"{qp}_raw").rate_per_100y.iloc[-1], rate_table(d[d.split == k_], f"{qp}_rc").rate_per_100y.iloc[-1]) for k_ in range(5)]
    s_["top_rate_raw_sd_splits"] = float(np.std([a for a, b in sp], ddof=1)); s_["top_rate_rc_sd_splits"] = float(np.std([b for a, b in sp], ddof=1))
    out[scope] = tab; summ_rows.append(s_)
    log(f"\n[{scope}] {s_['drivers']} test records of {nd} drivers, {s_['crashes']} crashes"); log(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log(f"  highest fifth, rows as evidence:      raw {s_['top_rate_raw']:.1f} vs weighted {s_['top_rate_rc']:.1f} (diff {s_['top_diff']:+.1f} [{s_['top_diff_lo']:+.1f}, {s_['top_diff_hi']:+.1f}], p = {s_['top_diff_p']:.3f}); share {100*s_['top_share_raw']:.1f}% vs {100*s_['top_share_rc']:.1f}%")
    log(f"  highest fifth, baseline trips:        raw {s_['top_rate_raw']:.1f} vs weighted {s_['top_rate_rcb']:.1f} (diff {s_['top_diff_b']:+.1f} [{s_['top_diff_b_lo']:+.1f}, {s_['top_diff_b_hi']:+.1f}], p = {s_['top_diff_b_p']:.3f}); share {100*s_['top_share_rcb']:.1f}%")
pd.concat([t.assign(scope=s) for s, t in out.items()]).to_csv(f"{OUTR}/tableR4_quintile_rates.csv", index=False)
pd.DataFrame(summ_rows).to_csv(f"{OUTR}/tableR4_quintile_summary.csv", index=False)
tst.to_csv(f"{OUTR}/test_scores.csv", index=False)
log(f"R4 done at {time.time()-t0:.0f}s")

# ==================================================================
# R7a. ranking of all drivers (Figure 5): weights from rows and weights from baseline trips
# ==================================================================
log("\n=== R7a: drivers who change fifth, weights from rows and from baseline trips ===")
b_n = pois_fit(np.column_stack([Xt, Xb]), y, E, alpha=1e-2).coef_[p_t:]
terc = np.digitize(nb, np.quantile(nb, [1 / 3, 2 / 3])); q_n = pd.qcut(Xb @ b_n, 5, labels=False).astype(int)
rows = []
for name, Xc in [("rows", Xb_rcz), ("baseline trips", Xb_rcz_b)]:
    b_r = pois_fit(np.column_stack([Xt, Xc]), y, E, alpha=1e-2).coef_[p_t:]; q_r = pd.qcut(Xc @ b_r, 5, labels=False).astype(int)
    for t_, nm in enumerate(["low", "mid", "high"]):
        rows.append(dict(evidence=name, evidence_tercile=nm, share_changing_fifth=float(np.mean((q_n != q_r)[terc == t_])),
                         share_of_raw_top_leaving=float(np.mean(q_r[(q_n == 4) & (terc == t_)] != 4)), share_of_raw_bottom_leaving=float(np.mean(q_r[(q_n == 0) & (terc == t_)] != 0)),
                         share_of_raw_top_fifth=float(np.mean(terc[q_n == 4] == t_)), share_of_weighted_top_fifth=float(np.mean(terc[q_r == 4] == t_)),
                         share_of_raw_bottom_fifth=float(np.mean(terc[q_n == 0] == t_)), share_of_weighted_bottom_fifth=float(np.mean(terc[q_r == 0] == t_))))
r7a = pd.DataFrame(rows); r7a.to_csv(f"{OUTR}/tableR7_ranking_baseline_weights.csv", index=False); log(r7a.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ==================================================================
# R7b. order of the three groups: one bootstrap of the drivers for all sixteen indicators
# ==================================================================
log("\n=== R7b: order of the three groups of indicators, joint bootstrap of the drivers ===")
G2 = ["Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 2.5 - 3.0 s"]; G3 = ["Time at > 80 mph"]; G1 = [c for c in CLEAN if c not in G2 + G3]
FN, FM, FS = {}, {}, {}
for c in CLEAN:
    n_i, m_i, ssw_i, ids_i = driver_summaries(c); pos_i = pos_of[ids_i].values
    FN[c] = np.zeros(N); FM[c] = np.zeros(N); FS[c] = np.zeros(N); FN[c][pos_i] = n_i; FM[c][pos_i] = m_i; FS[c][pos_i] = ssw_i
def kappa_on(c, idx):
    ok = FN[c][idx] >= 1; ii = idx[ok]; return vc_from_summaries(FN[c][ii], FM[c][ii], FS[c][ii])[2]
rng = np.random.default_rng(12); KB_ = np.empty((B_GRP, len(CLEAN)))
for b_ in range(B_GRP):
    idx = rng.integers(0, N, N); KB_[b_] = [kappa_on(c, idx) for c in CLEAN]
j1, j2, j3 = [CLEAN.index(c) for c in G1], [CLEAN.index(c) for c in G2], [CLEAN.index(c) for c in G3]
o12 = float(np.mean(KB_[:, j1].max(axis=1) < KB_[:, j2].min(axis=1))); o23 = float(np.mean(KB_[:, j2].max(axis=1) < KB_[:, j3].min(axis=1)))
k_full = np.array([kappa_on(c, np.arange(N)) for c in CLEAN])
grp = dict(B=B_GRP, share_first_below_second=o12, share_second_below_third=o23, n_inf_rare=int(np.sum(~np.isfinite(KB_[:, j3[0]]))),
           first_max=float(k_full[j1].max()), second_min=float(k_full[j2].min()), second_max=float(k_full[j2].max()), third=float(k_full[j3[0]]),
           rare_lo=float(np.percentile(np.where(np.isfinite(KB_[:, j3[0]]), KB_[:, j3[0]], 1e12), 2.5)))
json.dump(grp, open(f"{OUTR}/group_order.json", "w"), indent=1); log(grp)

# ==================================================================
# R7c. severity coding: codes among the drivers with one crash, and share at fault by code
# ==================================================================
log("\n=== R7c: severity codes of the drivers with one crash ===")
sev_cols = ["Participant_ID", "N_events_total", "N_crashes", "N_near_crashes", "Max_Severity", "Mean_Severity"]
sv = df[sev_cols].sort_values("N_events_total").groupby("Participant_ID").tail(1).set_index("Participant_ID").reindex(drv.index)
sv["S_c"] = sv.Mean_Severity * sv.N_events_total - sv.N_near_crashes; sv["crashes"] = drv.crashes; sv["atfault"] = drv.atfault.fillna(0)
one = sv[sv.crashes == 1]; code = {k: (one.S_c - k).abs() < 0.5 for k in (0, 3, 4)}
sevc = dict(drivers_one_crash=int(len(one)), **{f"share_code{k}": float(code[k].mean()) for k in code}, share_other=float(1 - sum(code[k].mean() for k in code)),
            **{f"atfault_code{k}": float((one.atfault[code[k]] >= 1).mean()) for k in code}, **{f"n_code{k}": int(code[k].sum()) for k in code})
json.dump(sevc, open(f"{OUTR}/severity_codes.json", "w"), indent=1); log(sevc)

# ==================================================================
# R7d. crashes plus near-crashes on D_b, with and without the likely event trips (paired bootstrap)
# ==================================================================
log("\n=== R7d: crashes plus near-crashes on D_b, all rows vs likely event trips removed ===")
M_GRID, M_MAX, N_DRAW = np.array([1, 2, 3, 4, 6, 8, 12]), 12, 6
SIG = {"Maximum Deceleration": -1, "ABS Activation": 1, "Maximum Acceleration": 1, "Number of Lateral Accels > Threshold": 1,
       "Maximum Lateral Acceleration": 1, "Time Where Headway 0.0 - 0.5 s": 1, "Time Where Headway 0.5 - 1.0 s": 1, "Brake Activations": 1}
Zsig = np.zeros(len(df))
for c, sgn in SIG.items():
    v = df[c].astype(float); Zsig += sgn * ((v - v.mean()) / (v.std() + 1e-9)).fillna(0).values
df["sig_index"] = Zsig / len(SIG)
k_ev = (drv.crashes + drv.near).reindex(df["Participant_ID"]).values
df["rank_in_driver"] = df.groupby("Participant_ID")["sig_index"].rank(ascending=False, method="first"); df["likely_event"] = df["rank_in_driver"] <= k_ev
def irls(X, y, E, alpha=1e-3, iters=30):
    n, p = X.shape; X1 = np.column_stack([np.ones(n), X]); lam = alpha * E.sum(); P = np.eye(p + 1) * lam; P[0, 0] = 0
    beta = np.zeros(p + 1); beta[0] = np.log(y.sum() / E.sum()); logE = np.log(E)
    for _ in range(iters):
        eta = X1 @ beta + logE; mu = np.exp(np.clip(eta, -30, 30)); z = (eta - logE) + (y - mu) / mu
        A = (X1 * mu[:, None]).T @ X1 + P; b = (X1 * mu[:, None]).T @ z; new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < 1e-8: beta = new; break
        beta = new
    return beta
def build_design(keep_ids, rows, seed=5):
    kpos = np.where(drv.index.isin(keep_ids))[0]; nk = len(kpos); rng = np.random.default_rng(seed); XM = {}
    sub_all = rows[rows["Participant_ID"].isin(keep_ids)]
    for c in TOP6:
        grp_ = sub_all[["Participant_ID", c]].groupby("Participant_ID")[c].apply(lambda s: s.values).reindex(keep_ids); maxn = max(len(a) for a in grp_)
        pad = np.full((nk, maxn), np.nan)
        for r_, a in enumerate(grp_): pad[r_, :len(a)] = a
        mu_c, sd_c = np.nanmean(beh[c]), np.nanstd(beh[c]) + 1e-9; XM[c] = {}
        for m_ in M_GRID:
            cols = []
            for d in range(N_DRAW):
                U = rng.random(pad.shape); U[np.isnan(pad)] = np.inf; order = np.argsort(U, axis=1)[:, :m_]
                xm = np.nanmean(np.take_along_axis(pad, order, axis=1), axis=1); xm = np.where(np.isfinite(xm), xm, mu_c); cols.append((xm - mu_c) / sd_c)
            XM[c][m_] = np.column_stack(cols)
    return dict(kpos=kpos, nk=nk, XM=XM)
def slope_m(des, c, m_, idx, yv, Ev, Xtv):
    XM = des["XM"][c][m_]; return np.mean([irls(np.column_stack([Xtv[idx], XM[idx, d]]), yv[idx], Ev[idx])[-1] for d in range(N_DRAW)])
keep_b = drv.index[drv.n_base_raw >= M_MAX]; des_all = build_design(keep_b, df); des_trim = build_design(keep_b, df[~df.likely_event])
kb, nkb = des_all["kpos"], des_all["nk"]; idx_b = np.arange(nkb); rng_bs = np.random.default_rng(22); boot_b = [rng_bs.integers(0, nkb, nkb) for _ in range(B_SENS)]
yb, Eb, Xb_t = y_all[kb], E[kb], Xt[kb]; rows = []
for c in TOP6:
    rec = dict(indicator=c, n_drivers=nkb, events=int(yb.sum()))
    for m_ in [1, 12]:
        a_full = slope_m(des_all, c, m_, idx_b, yb, Eb, Xb_t); t_full = slope_m(des_trim, c, m_, idx_b, yb, Eb, Xb_t)
        A_ = np.array([slope_m(des_all, c, m_, bi, yb, Eb, Xb_t) for bi in boot_b]); T_b = np.array([slope_m(des_trim, c, m_, bi, yb, Eb, Xb_t) for bi in boot_b]); diff = T_b - A_
        rec.update({f"s{m_}_all": a_full, f"s{m_}_all_lo": pct(A_, 2.5), f"s{m_}_all_hi": pct(A_, 97.5), f"s{m_}_trim": t_full, f"s{m_}_trim_lo": pct(T_b, 2.5), f"s{m_}_trim_hi": pct(T_b, 97.5),
                    f"s{m_}_diff": t_full - a_full, f"s{m_}_diff_lo": pct(diff, 2.5), f"s{m_}_diff_hi": pct(diff, 97.5), f"s{m_}_p": two_sided_p(diff, B_SENS)})
    rows.append(rec)
    log(f"  {c[:32]:32s} m=12: all {rec['s12_all']:+.3f} [{rec['s12_all_lo']:+.3f}, {rec['s12_all_hi']:+.3f}]  trimmed {rec['s12_trim']:+.3f} [{rec['s12_trim_lo']:+.3f}, {rec['s12_trim_hi']:+.3f}]  diff {rec['s12_diff']:+.3f} [{rec['s12_diff_lo']:+.3f}, {rec['s12_diff_hi']:+.3f}] p = {rec['s12_p']:.3f}   [{time.time()-t0:.0f}s]")
pd.DataFrame(rows).to_csv(f"{OUTR}/tableR7_event_trips_secondary.csv", index=False)

# ==================================================================
# R7e. rows on which each indicator is missing
# ==================================================================
miss = {c: float(df[c].isna().mean()) for c in CLEAN}; HEADW = [c for c in CLEAN if "Headway" in c or c == "Time With Lead Vehicle"]
ev = dict(missing_share=miss, missing_min_other=float(min(v for c, v in miss.items() if c not in HEADW)), missing_max_other=float(max(v for c, v in miss.items() if c not in HEADW)),
          missing_headway=float(max(miss[c] for c in HEADW)), median_evidence={c: float(np.median(nk_df[c][nk_df[c] >= 1])) for c in CLEAN}, median_rows=float(np.median(nb)),
          baseline_rows=int(drv.n_rows.sum() - (drv.crashes + drv.near).sum()))
json.dump(ev, open(f"{OUTR}/evidence_missing.json", "w"), indent=1); log("\nR7e:", {k: v for k, v in ev.items() if k not in ("missing_share", "median_evidence")})
# ==================================================================
# R7f  stability of typical behaviour over the study (assumption M1)
# ==================================================================
M_DR, B_DR = 5, 1000
rng_dr = np.random.default_rng(7); rows_dr = []
for c in CLEAN:
    d = df[["Participant_ID", "Trip Day Number in Study", "Trip ID", c]].dropna(subset=[c]).sort_values(["Participant_ID", "Trip Day Number in Study", "Trip ID"])
    g = d.groupby("Participant_ID")[c].apply(np.asarray); g = g[g.apply(len) >= 2 * M_DR]
    A = np.array([v[:M_DR].mean() for v in g]); Z = np.array([v[-M_DR:].mean() for v in g]); R1, R2 = [], []
    for v in g:
        p_ = rng_dr.permutation(len(v)); R1.append(v[p_[:M_DR]].mean()); R2.append(v[p_[M_DR:2 * M_DR]].mean())
    R1, R2 = np.array(R1), np.array(R2); nd = len(g)
    cor = lambda x, y: np.corrcoef(x, y)[0, 1]
    bd = []
    for _ in range(B_DR):
        i = rng_dr.integers(0, nd, nd); bd.append(cor(A[i], Z[i]) - cor(R1[i], R2[i]))
    bd = np.array(bd)
    rows_dr.append(dict(indicator=c, drivers=nd, random_halves=cor(R1, R2), time_halves=cor(A, Z), diff=cor(A, Z) - cor(R1, R2),
                        lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), B=B_DR))
dr = pd.DataFrame(rows_dr); dr.to_csv(f"{OUTR}/tableR7_drift.csv", index=False)
log("\nR7f drift:", int((dr.hi < 0).sum()), "of", len(dr), "indicators with a lower time-ordered correlation; median difference", round(float(dr["diff"].median()), 3))

# ==================================================================
# R7g  the estimator of kappa: unweighted (paper) against weighted between-driver variance
# ==================================================================
def kappa_weighted(n, m, ssw):
    multi = n >= 2; s2 = ssw[multi].sum() / (n[multi] - 1).sum(); N = n.sum(); K = len(n); xb = (n * m).sum() / N
    c_ = (K - 1) / K / ((n / N) * (1 - n / N)).sum(); t2 = c_ * (K / (K - 1) * ((n / N) * (m - xb) ** 2).sum() - K * s2 / N)
    return s2 / t2 if t2 > 0 else np.inf
rng_k = np.random.default_rng(5); rows_k = []
for c in CLEAN:
    n, m, ssw, _ = driver_summaries(c)
    bw = []
    for b in range(500):
        i = rng_k.integers(0, len(n), len(n)); bw.append(kappa_weighted(n[i], m[i], ssw[i]))
    bw = np.array(bw)
    rows_k.append(dict(indicator=c, paper=vc_from_summaries(n, m, ssw)[2], weighted=kappa_weighted(n, m, ssw), w_lo=np.percentile(bw, 2.5), w_hi=np.percentile(bw, 97.5),
                       paper_without_one_trip_drivers=vc_from_summaries(n[n >= 2], m[n >= 2], ssw[n >= 2])[2], one_trip_drivers=int((n == 1).sum())))
ke = pd.DataFrame(rows_k); ke.to_csv(f"{OUTR}/tableR7_kappa_estimators.csv", index=False)
n, m, ssw, _ = driver_summaries("Time at > 80 mph"); sel = n >= 2; wv = ssw[sel] / (n[sel] - 1)
th = pd.qcut(n[sel], 3, labels=["few", "middle", "many"], duplicates="drop"); wm = pd.Series(wv).groupby(np.asarray(th)).mean()
kr = ke.set_index("indicator").loc["Time at > 80 mph"]
json.dump(dict(paper=float(kr.paper), weighted=float(kr.weighted), w_lo=float(kr.w_lo), w_hi=float(kr.w_hi), paper_without_one_trip_drivers=float(kr.paper_without_one_trip_drivers),
               one_trip_drivers=int(kr.one_trip_drivers), within_variance_by_third={k: float(v) for k, v in wm.items()},
               within_variance_ratio_many_few=float(wm["many"] / wm["few"])), open(f"{OUTR}/kappa_rare_estimators.json", "w"), indent=1)
log("R7g estimators:", ke[["indicator", "paper", "weighted", "w_lo", "w_hi"]].round(1).to_string(index=False))

log(f"\nall done at {time.time()-t0:.0f}s")
