# ==================================================================
# PRELIMINARY results on SHRP2_merged_P4.csv for the paper
# "How much driving is enough? Credibility-weighted behavioural
#  risk from naturalistic data"
#
# Honest scope: in this file every row is an SHRP2 event record
# (crash / near-crash / baseline) and the per-row event type is NOT
# available, so behavioural means include event trips. The EVIDENCE
# for indicator k of driver i is n_ik = the number of rows on which k
# is observed, i.e. the number of trips actually averaged (this is
# what kappa, C(n) and the calibration use; review fix D1). The
# baseline count n_base = (rows - crashes - near)_+ is kept only as a
# descriptive quantity and to define the sensitivity set D_b.
# To limit contamination, only indicators WITHOUT an event signature
# (data-driven test, Table 1) are used as behaviour. Exposure offset
# = days in study (true observation window), never sampled miles.
# Review fixes M1/M2: kappa_z = sigma^2 / tau_z^2 with tau_z^2 the
# residual between-driver variance given the covariates z, and the
# covariate-conditional regression calibration.
#
# Deliverables produced:
#   Table 1  event-signature screen of trip-summary variables
#   Table 2  reliability table: kappa_j per clean indicator (+ bootstrap CI)
#   Fig 1    attenuation test: behaviour-crash slope by evidence stratum (between drivers)
#   Fig 1b   within-driver subsampling test: slope vs m rows sampled per driver
#   Table 5  kappa cross-check: variance components vs free fit to the crash-slope curve
#   Table 4  naive vs regression-calibrated effect sizes
#   Table 3  model comparison (5 splits) by evidence stratum + falsification
#   Fig 2    credibility curves and deviance by stratum
# ==================================================================
import os, warnings, copy, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
warnings.filterwarnings("ignore")
torch.set_num_threads(4); torch.manual_seed(7); np.random.seed(7)
OUT = "prelim_out"; os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("SHRP2_merged_P4.csv", low_memory=False)
N_ROWS_ALL = len(df)                                            # rows before removing copies of a trip
df = df.drop_duplicates(["Participant_ID", "Trip ID"]).reset_index(drop=True)   # one row per trip (copies are identical)
N_TRIP_COPIES, N_TRIPS = N_ROWS_ALL - len(df), len(df)
# Rows with a mean trip speed of exactly 0 (952 rows, 391 drivers) have a normal duration, brake activations and
# road-type shares: the vehicle moved but the speed channel is missing. The three speed-based variables are
# treated as missing on these rows (the row is kept for the other indicators).
_zero_speed = df["Mean Speed"] == 0
N_ZERO_SPEED_ROWS, N_ZERO_SPEED_DRIVERS = int(_zero_speed.sum()), int(df.loc[_zero_speed, "Participant_ID"].nunique())
df.loc[_zero_speed, ["Mean Speed", "Time at 70-80 mph", "Time at > 80 mph"]] = np.nan
for c in [c for c in df.columns if c.startswith("Minimum TTC")]:
    df.loc[df[c] >= 1e4, c] = 10.0; df[c] = df[c].clip(upper=10.0)
lead = df["Time With Lead Vehicle"].replace(0, np.nan)
for c in [c for c in df.columns if c.startswith("Time Where Headway")]:
    df[c] = (df[c] / lead).clip(0, 1)
dur = df["Trip Duration"].replace(0, np.nan)
for c in ["Time at 70-80 mph", "Time at > 80 mph", "Time With Lead Vehicle"]:
    df[c] = (df[c] / dur).clip(0, 1)
df["Trip Duration"] = df["Trip Duration"] / 60000.0         # ms -> minutes (scale-free for kappa and z-scores)

TRIP_VARS = ["Trip Duration", "Mean Speed", "Maximum Acceleration", "Maximum Deceleration",
             "Maximum Lateral Acceleration", "Number of Longitudinal Accels > Threshold",
             "Number of Longitudinal Decels > Threshold", "Number of Lateral Accels > Threshold",
             "Brake Activations", "Trip Distance Origin to Destination", "ABS Activation",
             "Time at 70-80 mph", "Time at > 80 mph", "Time With Lead Vehicle",
             "Time Where Headway 0.0 - 0.5 s", "Time Where Headway 0.5 - 1.0 s",
             "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 2.5 - 3.0 s",
             "Time Where Headway 3.0 - 3.5 s", "Time Where Headway > 3.5 s",
             "Minimum TTC Lead 0-10 mph", "Minimum TTC Lead 10-20 mph", "Minimum TTC Lead 20-30 mph",
             "Minimum TTC Lead 30-40 mph", "Minimum TTC Lead 40-50 mph", "Minimum TTC Lead >= 50 mph",
             "% Urb Frwy", "% Urb 2 Ln", "% Rur Frwy", "% Rur 2 Ln",
             "% Spd Lim 35 or Less", "% Spd Lim 40-50", "% Spd Lim 55-65", "% Spd Lim 70 or Greater"]

g = df.groupby("Participant_ID")
drv = pd.DataFrame({"n_rows": g.size(), "crashes": g["N_crashes"].max(), "near": g["N_near_crashes"].max(),
                    "atfault": g["N_at_fault_crash"].max(),
                    "days": g["Trip Day Number in Study"].max() - g["Trip Day Number in Study"].min() + 1})
drv["n_base_raw"] = drv.n_rows - drv.crashes - drv.near                 # can be <= 0 for 16 drivers (more events than rows)
drv["n_base"] = drv.n_base_raw.clip(lower=1)
drv["days"] = drv["days"].fillna(drv["days"].median()).clip(lower=30)
event_driver = (drv.crashes + drv.near) > 0
df["event_driver"] = df["Participant_ID"].map(event_driver)

# ------------------------------------------------------------------
# TABLE 1: event-signature screen. Extreme 3% rows (risky tail) --
# are they over-represented among event drivers beyond their row share?
# ------------------------------------------------------------------
RISKY_LOW = {"Minimum TTC Lead 0-10 mph", "Minimum TTC Lead 10-20 mph", "Minimum TTC Lead 20-30 mph",
             "Minimum TTC Lead 30-40 mph", "Minimum TTC Lead 40-50 mph", "Minimum TTC Lead >= 50 mph",
             "Maximum Deceleration"}
row_share = df["event_driver"].mean(); rows1 = []
for c in TRIP_VARS:
    s = df[c]
    low = c in RISKY_LOW or (s.min() < 0 and "Decel" in c)
    q = s.quantile(0.03) if low else s.quantile(0.97)
    ext = (s <= q) if low else (s >= q)
    if ext.sum() < 50: continue
    enr = df.loc[ext, "event_driver"].mean() / row_share
    rows1.append(dict(variable=c, extreme_rows=int(ext.sum()), event_driver_share=df.loc[ext, "event_driver"].mean(),
                      enrichment=enr, degenerate=bool(ext.mean() > 0.10)))
t1 = pd.DataFrame(rows1).sort_values("enrichment", ascending=False)
t1["event_signature"] = t1["enrichment"] > 1.08
t1["status"] = np.where(t1.event_signature, "event signature", np.where(t1.degenerate, "mass point (unusable)", "clean"))   # ratio first: a ratio above 1.08 is an event signature even on a wide tail (author, 12 Nov 2026)
t1.to_csv(f"{OUT}/table1_event_signature_screen.csv", index=False)
CLEAN = [v for v in t1.loc[t1.status == "clean", "variable"] if v not in ("Trip Distance Origin to Destination",)]
print("TABLE 1 — event-signature screen (row share of event drivers = %.3f)" % row_share)
print(t1[["variable", "extreme_rows", "event_driver_share", "enrichment", "status"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nclean behavioural indicators ({len(CLEAN)}): {CLEAN}\n")

# ------------------------------------------------------------------
# Driver-level design: traditional factors (SPF) + clean behaviour means
# ------------------------------------------------------------------
first = g[["Gender", "Age_Group", "Years_Driving", "Num_Violations", "Num_Crashes_Prior", "Avg_Annual_Mileage"]].first()
beh = g[CLEAN].mean()
nk_df = g[CLEAN].count()                                     # n_ik: trips on which indicator k is observed for driver i
D = drv.join(first).join(beh)
D["Years_Driving"] = D["Years_Driving"].fillna(D["Years_Driving"].median())
Xtrad = pd.get_dummies(D[["Gender", "Age_Group", "Num_Violations", "Num_Crashes_Prior", "Avg_Annual_Mileage"]].astype(str),
                       drop_first=True).astype(float)
Xtrad["Years_Driving"] = (D["Years_Driving"] - D["Years_Driving"].mean()) / D["Years_Driving"].std()
Xt = Xtrad.values; Xb = beh.fillna(beh.median()).values
Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)
y = D["crashes"].values.astype(float); E = (D["days"].values / 365.0)
nb = D["n_rows"].values.astype(float)                        # evidence of a driver = recorded trips (rows) in the file
nk_mat = nk_df.reindex(D.index).values.astype(float).clip(min=1)   # per-indicator evidence n_ik

# ------------------------------------------------------------------
# TABLE 2: reliability table (variance components) on clean indicators
#   sigma2 = pooled within-driver variance;  tau2 = Var(means) - sigma2*E[1/n]
#   kappa = sigma2/tau2  (marginal);  kappa_z = sigma2/tau_z2 with tau_z2 the
#   residual between-driver variance given the covariates z (review fix M1)
# ------------------------------------------------------------------
def driver_summaries(c, sub=None):
    """per-driver (n_i, mean_i, within-SS_i, driver index) for indicator c -- all that the estimator needs"""
    gg = (df if sub is None else sub).groupby("Participant_ID")[c]
    n = gg.count().values.astype(float); m = gg.mean().values
    ssw = (gg.var(ddof=0) * gg.count()).values          # sum of squared deviations within driver
    ok = n >= 1
    ids = gg.count().index[ok]
    return n[ok], m[ok], np.nan_to_num(ssw[ok]), ids

def vc_from_summaries(n, m, ssw, idx=None):
    """marginal variance components; tau2 <= 0 -> kappa = inf (credibility 0), as in Buhlmann-Gisler eq. (4.27)"""
    if idx is not None: n, m, ssw = n[idx], m[idx], ssw[idx]
    multi = n >= 2
    sigma2 = ssw[multi].sum() / (n[multi] - 1).sum()
    tau2 = m.var(ddof=1) - sigma2 * (1.0 / n).mean()
    return sigma2, tau2, (sigma2 / tau2 if tau2 > 0 else np.inf)

def vc_z_from_summaries(n, m, ssw, Z, idx=None):
    """covariate-conditional components: OLS of driver means on [1, Z]; residuals e, leverages h;
       tau_z2 = (sum e^2 - sigma2 * sum (1-h_ii)/n_i) / (N - p)   (unbiased under M1-M3 and linear E[theta|z])"""
    if idx is not None: n, m, ssw, Z = n[idx], m[idx], ssw[idx], Z[idx]
    multi = n >= 2
    sigma2 = ssw[multi].sum() / (n[multi] - 1).sum()
    Z1 = np.column_stack([np.ones(len(n)), Z])
    G = np.linalg.pinv(Z1.T @ Z1); H_diag = np.einsum("ij,jk,ik->i", Z1, G, Z1)
    coef = G @ (Z1.T @ m); e = m - Z1 @ coef; p = int(round(H_diag.sum()))
    tau_z2 = (np.sum(e ** 2) - sigma2 * np.sum((1 - H_diag) / n)) / (len(n) - p)
    return sigma2, tau_z2, (sigma2 / tau_z2 if tau_z2 > 0 else np.inf), coef

def pct_with_inf(a, q):
    """order-statistic percentiles that tolerate +inf replicates (tau2 <= 0)"""
    a = np.sort(np.asarray(a, float)); k = int(np.clip(round(q / 100 * (len(a) - 1)), 0, len(a) - 1)); return a[k]

pos_of = pd.Series(np.arange(len(D)), index=D.index)         # driver id -> row in D / Xt
n_med = float(np.median(nb)); rng_b = np.random.default_rng(11)
rows2 = []
for c in CLEAN:
    n_i, m_i, ssw_i, ids_i = driver_summaries(c); Z_i = Xt[pos_of[ids_i].values]
    s2, t2_, k = vc_from_summaries(n_i, m_i, ssw_i)
    s2z, tz2, kz, _ = vc_z_from_summaries(n_i, m_i, ssw_i, Z_i)
    boots, bootz = [], []                                    # cluster (driver) bootstrap, 500 reps, same resamples
    for b in range(500):
        idx = rng_b.integers(0, len(n_i), len(n_i))
        boots.append(vc_from_summaries(n_i, m_i, ssw_i, idx)[2]); bootz.append(vc_z_from_summaries(n_i, m_i, ssw_i, Z_i, idx)[2])
    lo, hi = pct_with_inf(boots, 2.5), pct_with_inf(boots, 97.5); loz, hiz = pct_with_inf(bootz, 2.5), pct_with_inf(bootz, 97.5)
    n_ref = 10.0                                             # reference evidence: median number of trips on which an indicator is observed
    rows2.append(dict(indicator=c, sigma2_within=s2, tau2_between=t2_, kappa=k, kappa_lo=lo, kappa_hi=hi, n_inf_boot=int(np.sum(~np.isfinite(boots))),
                      tau2_z=tz2, kappa_z=kz, kappa_z_lo=loz, kappa_z_hi=hiz, n_inf_boot_z=int(np.sum(~np.isfinite(bootz))),
                      median_n_ik=float(np.median(n_i)), harmonic_mean_n_ik=float(1 / np.mean(1 / n_i)),
                      epochs_to_half_credibility=k, reliability_at_median_n=n_ref / (n_ref + k) if np.isfinite(k) else 0,
                      reliability_z_at_median_n=n_ref / (n_ref + kz) if np.isfinite(kz) else 0))
t2 = pd.DataFrame(rows2).sort_values("kappa")
t2.to_csv(f"{OUT}/table2_reliability_kappa.csv", index=False)
print("TABLE 2 — reliability of behavioural indicators (median recorded trips per driver = %.0f)" % n_med)
print(t2[["indicator", "kappa", "kappa_lo", "kappa_hi", "kappa_z", "kappa_z_lo", "kappa_z_hi", "median_n_ik", "reliability_at_median_n"]]
      .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ------------------------------------------------------------------
# credibility weights and calibrated covariates (per-indicator evidence n_ik)
#   Xb_rc  : marginal calibration  mu + C_ik (xbar - mu)          (kept for comparison)
#   Xb_rcz : covariate-conditional calibration  m_ik + C_z,ik (xbar - m_ik),  m_ik = OLS fit of xbar on z  (review fix M2)
# ------------------------------------------------------------------
kap = t2.set_index("indicator").loc[CLEAN, "kappa"].values; kap = np.where(np.isfinite(kap), kap, 1e6)
kap_z = t2.set_index("indicator").loc[CLEAN, "kappa_z"].values; kap_z = np.where(np.isfinite(kap_z), kap_z, 1e6)
tau2z_std = t2.set_index("indicator").loc[CLEAN, "tau2_z"].values / (beh.std(ddof=0).reindex(CLEAN).values ** 2 + 1e-18)   # tau_z^2 in standardised units
R = nk_mat / (nk_mat + kap[None, :])                         # marginal credibility of each driver mean
Rz = nk_mat / (nk_mat + kap_z[None, :])                      # covariate-conditional credibility
Xb_rc = R * Xb                                               # marginal calibration (standardised means, mu = 0)
_Z1 = np.column_stack([np.ones(len(Xt)), Xt]); _Hz = _Z1 @ np.linalg.pinv(_Z1.T @ _Z1) @ _Z1.T
Mhat = _Hz @ Xb                                              # covariate-predicted driver means (standardised units)
Xb_rcz = Mhat + Rz * (Xb - Mhat)                             # covariate-conditional calibration
tert = np.quantile(nb, [1 / 3, 2 / 3]); strat = np.digitize(nb, tert)   # 0 low, 1 mid, 2 high evidence (recorded trips)
print(f"\nevidence strata (recorded trips): low <= {tert[0]:.0f}, mid, high > {tert[1]:.0f}; "
      f"stratum means n = {[round(nb[strat==s].mean(),1) for s in range(3)]}")

# ------------------------------------------------------------------
json.dump({"zero_speed_rows": N_ZERO_SPEED_ROWS, "zero_speed_drivers": N_ZERO_SPEED_DRIVERS}, open(f"{OUT}/zero_speed.json", "w"))
_tl, _tm, _th = (nb[strat == s_] for s_ in range(3))                  # the three evidence thirds (recorded trips)
json.dump({"n_rows_all": int(N_ROWS_ALL), "n_trip_copies": int(N_TRIP_COPIES), "n_trips": int(N_TRIPS), "n_drivers": int(len(drv)),
           "trips_no_event_drivers": int(drv.loc[(drv.crashes + drv.near) == 0, "n_rows"].sum()),
           "tert_low_max": int(_tl.max()), "tert_mid_min": int(_tm.min()), "tert_mid_max": int(_tm.max()), "tert_high_min": int(_th.min())},
          open(f"{OUT}/trip_facts.json", "w"), indent=1)

# FIG 1 / attenuation test: per-feature slope by evidence stratum,
# Poisson GLM with offset log(days) and traditional controls.
# Theory: slope_s ~ beta * n_s/(n_s+kappa_j)  -> rises with evidence.
# ------------------------------------------------------------------
def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m

# rank clean features by association strength (full-sample slope)
assoc = []
for j, c in enumerate(CLEAN):
    m = pois_fit(np.column_stack([Xt, Xb[:, j]]), y, E); assoc.append((abs(m.coef_[-1]), c, j))
assoc.sort(reverse=True); top = assoc[:6]
rows_att = []
for _, c, j in top:
    for s in range(3):
        sel = strat == s
        m = pois_fit(np.column_stack([Xt[sel], Xb[sel, j]]), y[sel], E[sel])
        # SE via observed information approx: bootstrap 60 reps
        bs = []
        for b in range(60):
            ii = np.random.choice(np.where(sel)[0], sel.sum(), replace=True)
            bs.append(pois_fit(np.column_stack([Xt[ii], Xb[ii, j]]), y[ii], E[ii]).coef_[-1])
        n_s = nk_mat[sel, j].mean(); k = kap_z[j]
        rows_att.append(dict(indicator=c, stratum=["low", "mid", "high"][s], mean_n=n_s, harmonic_mean_n=1 / np.mean(1 / nk_mat[sel, j]), slope=m.coef_[-1],
                             se=np.std(bs), predicted_reliability=n_s / (n_s + k), kappa_z=k))
att = pd.DataFrame(rows_att); att.to_csv(f"{OUT}/table_attenuation_slopes.csv", index=False)
print("\nATTENUATION TEST — slope of log crash rate on indicator (z-score), by evidence stratum")
print(att.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (_, c, j) in zip(axes.ravel(), top):
    sub = att[att.indicator == c]
    ax.errorbar(sub.mean_n, sub.slope, yerr=1.96 * sub.se, fmt="o-", color="#534AB7", capsize=4, label="estimated slope")
    bh = sub.slope.iloc[-1] / max(sub.predicted_reliability.iloc[-1], 1e-6)   # implied true slope from high stratum
    ns = np.linspace(1, nb.max(), 100); ax.plot(ns, bh * ns / (ns + kap_z[j]), "--", color="#D85A30",
                                                label=f"theory: β·n/(n+κ_z), κ_z={kap_z[j]:.1f}")
    ax.axhline(0, lw=0.8, color="gray"); ax.set_xscale("log")
    ax.set_title(c, fontsize=10); ax.set_xlabel("evidence n (recorded trips, log)"); ax.set_ylabel("slope")
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_attenuation_test.png", dpi=150); fig.savefig(f"{OUT}/fig1_attenuation_test.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# FIG 1b / within-driver subsampling test (driver composition held fixed).
#   Keep drivers with >= M_MAX rows. For each m in M_GRID draw m rows per
#   driver at random, average them, fit the same Poisson GLM, repeat
#   N_DRAW times and average the slope. Measurement-error theory predicts
#   E[slope_m] = beta * m/(m+kappa_j) with kappa_j taken from Table 2
#   (estimated from the behaviour data alone, never from crashes).
#   We also fit kappa freely from the crash slopes (1-parameter profile)
#   and compare it with the variance-components kappa_j.
# ------------------------------------------------------------------
M_GRID, M_MAX, N_DRAW = [1, 2, 3, 4, 6, 8, 12], 12, 25
keep_ids = drv.index[drv.n_rows >= M_MAX]
kpos = np.where(drv.index.isin(keep_ids))[0]                 # D/Xt/y/E keep drv's row order
print(f"\nWITHIN-DRIVER SUBSAMPLING TEST: {len(keep_ids)} drivers with >= {M_MAX} rows, "
      f"{int(y[kpos].sum())} crashes among them")
# kappa_z on the test set D (the quantity Proposition 2 refers to, review fix M1)
kap_z_D = {}
for _, c, j in top:
    n_i, m_i, ssw_i, ids_i = driver_summaries(c, df[df["Participant_ID"].isin(keep_ids)])
    kap_z_D[c] = vc_z_from_summaries(n_i, m_i, ssw_i, Xt[pos_of[ids_i].values])[2]
print("kappa_z on D:", {c[:24]: round(v, 2) for c, v in kap_z_D.items()})

def pois_slope_se(X, y, E, alpha=1e-3):
    """slope of last column + Fisher SE (Poisson, offset log E)"""
    m = pois_fit(X, y, E, alpha); mu = E * np.exp(m.intercept_ + X @ m.coef_)
    X1 = np.column_stack([np.ones(len(X)), X]); info = (X1 * mu[:, None]).T @ X1      # Fisher information (counts)
    try: se = np.sqrt(np.linalg.inv(info)[-1, -1])
    except np.linalg.LinAlgError: se = np.nan
    return m.coef_[-1], se

rng_w = np.random.default_rng(5); rows_w = []; kappa_cmp = []
for _, c, j in top:
    # padded matrix [driver, row] of the indicator for kept drivers
    sub = df[df["Participant_ID"].isin(keep_ids)][["Participant_ID", c]]
    grp = sub.groupby("Participant_ID")[c].apply(lambda s: s.values)
    grp = grp.reindex(keep_ids); maxn = max(len(a) for a in grp)
    pad = np.full((len(keep_ids), maxn), np.nan)
    for r, a in enumerate(grp): pad[r, :len(a)] = a
    mu_c, sd_c = np.nanmean(beh[c]), np.nanstd(beh[c]) + 1e-9          # FIXED scaling (full-sample driver means)
    for m_ in M_GRID:
        sl, ses = [], []
        for d in range(N_DRAW):
            U = rng_w.random(pad.shape); U[np.isnan(pad)] = np.inf
            order = np.argsort(U, axis=1)[:, :m_]
            xm = np.nanmean(np.take_along_axis(pad, order, axis=1), axis=1)
            xm = np.where(np.isfinite(xm), xm, mu_c); z = (xm - mu_c) / sd_c
            b, se = pois_slope_se(np.column_stack([Xt[kpos], z]), y[kpos], E[kpos]); sl.append(b); ses.append(se)
        rows_w.append(dict(indicator=c, m=m_, slope=np.mean(sl), mc_sd=np.std(sl), se=np.nanmean(ses),
                           predicted_reliability=m_ / (m_ + kap_z_D[c])))
    w = pd.DataFrame([r for r in rows_w if r["indicator"] == c]); mm = w.m.values; s_ = w.slope.values; wt = 1 / (w.se.values ** 2)
    # one-parameter fit (beta) with kappa fixed at the variance-components value
    f_vc = mm / (mm + kap_z_D[c]); b_vc = (wt * s_ * f_vc).sum() / (wt * f_vc ** 2).sum(); chi_vc = (wt * (s_ - b_vc * f_vc) ** 2).sum()
    # flat alternative (no attenuation)
    b_flat = (wt * s_).sum() / wt.sum(); chi_flat = (wt * (s_ - b_flat) ** 2).sum()
    # free-kappa profile on a log grid
    kg = np.exp(np.linspace(np.log(0.05), np.log(200), 400)); prof = []
    for kk in kg:
        f = mm / (mm + kk); bb = (wt * s_ * f).sum() / (wt * f ** 2).sum(); prof.append((wt * (s_ - bb * f) ** 2).sum())
    prof = np.array(prof); k_free = kg[prof.argmin()]; inside = kg[prof <= prof.min() + 3.84]
    kappa_cmp.append(dict(indicator=c, kappa_vc=kap_z_D[c], kappa_marginal_allN=kap[j], kappa_from_crash_slopes=k_free,
                          k_free_lo=inside.min(), k_free_hi=inside.max(), chi2_fixed_kappa=chi_vc, chi2_flat=chi_flat,
                          beta_implied=b_vc, slope_m1=s_[0], slope_m12=s_[-1]))
att_w = pd.DataFrame(rows_w); att_w.to_csv(f"{OUT}/table_attenuation_within.csv", index=False)
kcmp = pd.DataFrame(kappa_cmp); kcmp.to_csv(f"{OUT}/table5_kappa_crosscheck.csv", index=False)
print("\nWITHIN-DRIVER TEST — mean slope (per SD of full-sample driver mean) vs number of sampled rows m")
print(att_w.pivot(index="indicator", columns="m", values="slope").loc[[c for _, c, _ in top]]
      .to_string(float_format=lambda v: f"{v:.3f}"))
print("\nTABLE 5 — kappa cross-check: variance components (behaviour only) vs free fit to crash-slope curve")
print(kcmp.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (_, c, j), kc in zip(axes.ravel(), top, kappa_cmp):
    w = att_w[att_w.indicator == c]
    ax.errorbar(w.m, w.slope, yerr=1.96 * w.se, fmt="o", color="#534AB7", capsize=4, label="mean slope over 25 draws")
    ms = np.linspace(1, 12, 100)
    ax.plot(ms, kc["beta_implied"] * ms / (ms + kap_z_D[c]), "--", color="#D85A30", label=f"theory: β·m/(m+κ_z), κ_z={kap_z_D[c]:.1f}")
    ax.axhline((w.slope / w.se ** 2).sum() / (1 / w.se ** 2).sum(), color="gray", lw=0.8, ls=":", label="no attenuation (flat)")
    ax.axhline(0, lw=0.6, color="black")
    ax.set_title(f"{c}\nκ from crash curve = {kc['kappa_from_crash_slopes']:.1f} [{kc['k_free_lo']:.1f}, {kc['k_free_hi']:.1f}]", fontsize=9)
    ax.set_xlabel("rows sampled per driver, m"); ax.set_ylabel("slope"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1b_attenuation_within.png", dpi=150); fig.savefig(f"{OUT}/fig1b_attenuation_within.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# TABLE 4: naive vs regression-calibrated effect sizes (pooled sample)
# ------------------------------------------------------------------
def rc_fit(Xcal, j, yv, Ev, iters=3):
    """regression calibration with the second-order offset 0.5*beta^2*tau_z^2*(1-C_z(n_i)) (Corollary 2), iterated"""
    off = np.zeros(len(yv)); b = np.nan
    for _ in range(iters):
        m = pois_fit(np.column_stack([Xt, Xcal[:, j]]), yv, Ev * np.exp(off)); b = m.coef_[-1]
        off = 0.5 * b ** 2 * tau2z_std[j] * (1 - Rz[:, j])
    return b, float(np.abs(off).max())
rows4 = []
for _, c, j in assoc[:8]:
    b_naive = pois_fit(np.column_stack([Xt, Xb[:, j]]), y, E).coef_[-1]
    b_rc_marg = pois_fit(np.column_stack([Xt, Xb_rc[:, j]]), y, E).coef_[-1]
    b_rc, off_max = rc_fit(Xb_rcz, j, y, E)
    hn = 1 / np.mean(1 / nk_mat[:, j])
    rows4.append(dict(indicator=c, kappa=kap[j], kappa_z=kap_z[j], mean_reliability=Rz[:, j].mean(), harmonic_mean_n=hn,
                      credibility_at_harmonic_mean_n=hn / (hn + kap_z[j]), slope_naive=b_naive, slope_rc_marginal=b_rc_marg,
                      slope_regression_calibrated=b_rc, max_offset_term=off_max,
                      understatement_pct=100 * (1 - b_naive / b_rc) if b_rc != 0 else np.nan,
                      predicted_understatement_pct=100 * kap_z[j] / (hn + kap_z[j])))
t4 = pd.DataFrame(rows4); t4.to_csv(f"{OUT}/table4_deattenuation.csv", index=False)
print("\nTABLE 4 — naive vs regression-calibrated effect sizes (log crash rate per SD)")
print(t4.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ------------------------------------------------------------------
# TABLE 3: model comparison, 5 repeated splits, by evidence stratum
#   SPF (traditional GLM) | SPF + linear behaviour | SPF + linear RC behaviour
#   | gated bounded correction (full trust) | + credibility C(n) | RC inputs
# ------------------------------------------------------------------
HMAX, LAM_O, LAM_S, STAGE1 = 1.0, 1.0, 0.01, 0.15
def pois_nll(eta, y, E): return (E * torch.exp(eta) - y * eta).mean()
def pois_dev(eta, y, E):
    mu = E * torch.exp(eta); eps = 1e-12
    return (2 * (y * torch.log((y + eps) / mu) - (y - mu))).mean().item()
def mlp(p, h=32): return nn.Sequential(nn.Linear(p, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
class GRN(nn.Module):
    def __init__(self, p):
        super().__init__(); self.h = mlp(p); self.gate = mlp(p, 16)
        with torch.no_grad():
            self.h[-1].weight.mul_(0.1); self.h[-1].bias.zero_(); self.gate[-1].bias.fill_(-3.0); self.gate[-1].weight.mul_(0.1)
    def forward(self, X, eta0, C):
        A = torch.sigmoid(self.gate(X)).squeeze(); h = HMAX * torch.tanh(self.h(X)).squeeze()
        return eta0 + C * A * h, A, h
def fit_grn(X, eta0, C, Xd, y, E, epochs=600, seed=0, patience=30):
    torch.manual_seed(seed); m = GRN(X.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=1e-4)
    n = len(X); p = torch.randperm(n); vi, fi = p[: n // 10], p[n // 10:]
    with torch.no_grad(): w = (E * torch.exp(eta0)); w = w / w.mean()
    best, state, wait, n1 = np.inf, None, 0, int(STAGE1 * epochs)
    for ep in range(epochs):
        opt.zero_grad()
        A = torch.sigmoid(m.gate(X[fi])).squeeze(); h = HMAX * torch.tanh(m.h(X[fi])).squeeze()
        r = C[fi] * h if ep < n1 else C[fi] * A * h
        pen = 0.0 if ep < n1 else LAM_S * A.mean()
        ortho = ((Xd[fi].T @ (w[fi] * r)) / len(fi)).pow(2).sum()
        (pois_nll(eta0[fi] + r, y[fi], E[fi]) + pen + LAM_O * ortho).backward(); opt.step()
        if ep >= n1:
            with torch.no_grad(): vl = pois_nll(m(X[vi], eta0[vi], C[vi])[0], y[vi], E[vi]).item()
            if vl < best - 1e-6: best, state, wait = vl, copy.deepcopy(m.state_dict()), 0
            else:
                wait += 1
                if wait >= patience: break
    if state: m.load_state_dict(state)
    return m

T = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
kap_pool = float(np.median(kap_z[np.isfinite(kap_z) & (kap_z < 1e5)]))
recs = []
for split in range(5):
    rng = np.random.default_rng(100 + split); idx = rng.permutation(len(y)); nte = len(y) // 4
    te, tr = idx[:nte], idx[nte:]
    spf = pois_fit(Xt[tr], y[tr], E[tr], alpha=1e-2)
    eta0_tr, eta0_te = spf.intercept_ + Xt[tr] @ spf.coef_, spf.intercept_ + Xt[te] @ spf.coef_
    lin = pois_fit(np.column_stack([Xt[tr], Xb[tr]]), y[tr], E[tr], alpha=1e-2)
    lin_rc = pois_fit(np.column_stack([Xt[tr], Xb_rcz[tr]]), y[tr], E[tr], alpha=1e-2)
    Xd_tr = T(np.column_stack([np.ones(len(tr)), Xt[tr]]))
    ones_tr, ones_te = torch.ones(len(tr)), torch.ones(len(te))
    C_tr, C_te = T(nb[tr] / (nb[tr] + kap_pool)), T(nb[te] / (nb[te] + kap_pool))
    m_ft = fit_grn(T(Xb[tr]), T(eta0_tr), ones_tr, Xd_tr, T(y[tr]), T(E[tr]), seed=split)
    m_cr = fit_grn(T(Xb[tr]), T(eta0_tr), C_tr, Xd_tr, T(y[tr]), T(E[tr]), seed=split)
    m_rc = fit_grn(T(Xb_rcz[tr]), T(eta0_tr), ones_tr, Xd_tr, T(y[tr]), T(E[tr]), seed=split)
    perm = rng.permutation(len(tr)); m_perm = fit_grn(T(Xb[tr][perm]), T(eta0_tr), ones_tr, Xd_tr, T(y[tr]), T(E[tr]), seed=split)
    with torch.no_grad():
        preds = {"SPF (traditional GLM)": eta0_te,
                 "SPF + linear behaviour": lin.intercept_ + np.column_stack([Xt[te], Xb[te]]) @ lin.coef_,
                 "SPF + linear RC behaviour": lin_rc.intercept_ + np.column_stack([Xt[te], Xb_rcz[te]]) @ lin_rc.coef_,
                 "gated correction, full trust": m_ft(T(Xb[te]), T(eta0_te), ones_te)[0].numpy(),
                 "gated correction + credibility": m_cr(T(Xb[te]), T(eta0_te), C_te)[0].numpy(),
                 "gated correction, RC inputs": m_rc(T(Xb_rcz[te]), T(eta0_te), ones_te)[0].numpy(),
                 "falsification (shuffled behaviour)": m_perm(T(Xb[te][rng.permutation(len(te))]), T(eta0_te), ones_te)[0].numpy()}
        gate_ft = m_ft(T(Xb[te]), T(eta0_te), ones_te)[1].mean().item()
        gate_perm = m_perm(T(Xb[te][rng.permutation(len(te))]), T(eta0_te), ones_te)[1].mean().item()
    for name, eta in preds.items():
        # portfolio calibration on train total
        et, _ = (eta, None)
        d_all = pois_dev(T(eta), T(y[te]), T(E[te]))
        d_s = [pois_dev(T(eta[strat[te] == s]), T(y[te][strat[te] == s]), T(E[te][strat[te] == s])) for s in range(3)]
        recs.append(dict(split=split, model=name, dev=d_all, dev_low=d_s[0], dev_mid=d_s[1], dev_high=d_s[2]))
    recs.append(dict(split=split, model="[gate mean: real / shuffled]", dev=gate_ft, dev_low=gate_perm, dev_mid=np.nan, dev_high=np.nan))
res = pd.DataFrame(recs)
summ = res.groupby("model", sort=False).agg(dev=("dev", "mean"), dev_sd=("dev", "std"), low=("dev_low", "mean"),
                                            mid=("dev_mid", "mean"), high=("dev_high", "mean"))
summ.to_csv(f"{OUT}/table3_model_comparison.csv")
print(f"\nTABLE 3 — test Poisson deviance (5 splits), clean behaviour only, offset = study days, kappa_pool = {kap_pool:.1f}")
print(summ.to_string(float_format=lambda v: f"{v:.4f}"))

# FIG 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ns = np.linspace(1, 60, 200)
for _, r in t2.head(6).iterrows():
    ax[0].plot(ns, ns / (ns + r.kappa), label=f"{r.indicator[:28]} (κ={r.kappa:.1f})")
ax[0].axvline(n_med, ls=":", c="gray"); ax[0].text(n_med + 1, 0.32, f"median driver (n={n_med:.0f})", fontsize=8, c="gray")
ax[0].set_xlabel("evidence n (recorded trips)"); ax[0].set_ylabel("reliability n/(n+κ)"); ax[0].set_title("Credibility of behavioural indicators")
ax[0].legend(fontsize=7)
models = ["SPF + linear behaviour", "SPF + linear RC behaviour", "gated correction, full trust",
          "gated correction + credibility", "gated correction, RC inputs", "falsification (shuffled behaviour)"]
xs = np.arange(3); w = 0.13; base = summ.loc["SPF (traditional GLM)"]
for k, mname in enumerate(models):
    r = summ.loc[mname]
    ax[1].bar(xs + (k - 2.5) * w, [100 * (1 - r.low / base.low), 100 * (1 - r.mid / base.mid), 100 * (1 - r.high / base.high)], w, label=mname)
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xticks(xs); ax[1].set_xticklabels(["low evidence", "mid", "high evidence"]); ax[1].set_ylabel("deviance improvement over SPF (%)")
ax[1].set_title("Gain over the traditional GLM by evidence stratum (5 splits)"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_credibility_and_strata.png", dpi=150); fig.savefig(f"{OUT}/fig2_credibility_and_strata.pdf"); plt.close(fig)
print(f"\nfigures: {OUT}/fig1_attenuation_test.png, {OUT}/fig2_credibility_and_strata.png")
