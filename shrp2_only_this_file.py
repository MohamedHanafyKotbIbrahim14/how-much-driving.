# ==================================================================
# "This is the only data I have": sensitivity analyses that make the
# event-linked SHRP2 merged file sufficient on its own.
#
#   R1  kappa_j from (a) all rows, (b) the 920 event-free drivers whose
#       rows are 100% baseline trips, (c) all drivers after trimming the
#       k = crashes + near-crashes rows per driver that look most like
#       event trips, (d) placebo: trimming k random rows per driver.
#   R2  within-driver subsampling (attenuation) test re-run on the
#       trimmed rows, with kappa cross-check.
#   R3  clean vs event-signature indicators: mean difference between the
#       "likely event rows" and the other rows of the same drivers.
# ==================================================================
import io, contextlib, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor

src = open("shrp2_prelim.py").read()
head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
OUT3 = f"{OUT}/only_this_file"; os.makedirs(OUT3, exist_ok=True)
PUR, ORA, GRY, BLU, GRN = "#534AB7", "#D85A30", "#8A8A8A", "#2F6DB5", "#3B8C5A"

def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m

# ---------- event-signature index per row and "likely event rows" ----------
SIG = {"Maximum Deceleration": -1, "ABS Activation": 1, "Maximum Acceleration": 1, "Number of Lateral Accels > Threshold": 1,
       "Maximum Lateral Acceleration": 1, "Time Where Headway 0.0 - 0.5 s": 1, "Time Where Headway 0.5 - 1.0 s": 1, "Brake Activations": 1}
Z = np.zeros(len(df))
for c, sgn in SIG.items():
    v = df[c].astype(float); Z += sgn * ((v - v.mean()) / (v.std() + 1e-9)).fillna(0).values
df["sig_index"] = Z / len(SIG)
k_ev = (drv.crashes + drv.near).reindex(df["Participant_ID"]).values      # events per driver, on each row
df["rank_in_driver"] = df.groupby("Participant_ID")["sig_index"].rank(ascending=False, method="first")
df["likely_event"] = df["rank_in_driver"] <= k_ev
rng = np.random.default_rng(3)
df["rand_rank"] = df.groupby("Participant_ID")["sig_index"].transform(lambda s: rng.permutation(len(s)) + 1)
df["random_trim"] = df["rand_rank"] <= k_ev
print(f"rows flagged as likely event trips: {df.likely_event.sum()} (= sum of crashes+near-crashes {int((drv.crashes+drv.near).sum())})")

# ---------- R3: do clean indicators differ between likely-event rows and other rows? ----------
ev_drivers = df["Participant_ID"].isin(drv.index[(drv.crashes + drv.near) > 0])
rows3 = []
for c in CLEAN + list(SIG):
    a = df.loc[ev_drivers & df.likely_event, c]; b = df.loc[ev_drivers & ~df.likely_event, c]
    sd = df[c].std(); d = (a.mean() - b.mean()) / sd
    rows3.append(dict(indicator=c, status="clean" if c in CLEAN else "event signature", likely_event_rows=len(a), other_rows=len(b),
                      standardised_difference=d))
r3 = pd.DataFrame(rows3).sort_values("standardised_difference", key=np.abs, ascending=False)
r3.to_csv(f"{OUT3}/tableR3_event_rows_vs_other_rows.csv", index=False)
print("\nR3 — likely event rows vs other rows of the same drivers (difference in SD units)")
print(r3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---------- R1: kappa under four row sets ----------
def kappa_on(sub, c):
    gg = sub.groupby("Participant_ID")[c]
    n = gg.count().values.astype(float); m = gg.mean().values; ssw = np.nan_to_num((gg.var(ddof=0) * gg.count()).values)
    ok = n >= 1; return vc_from_summaries(n[ok], m[ok], ssw[ok])[2]
sets = {"all rows": df, "event-free drivers (all-baseline rows)": df[~ev_drivers],
        "trimmed: likely event rows removed": df[~df.likely_event], "placebo: random rows removed": df[~df.random_trim]}
rows1 = []
for c in CLEAN:
    rec = dict(indicator=c)
    for nm, sub in sets.items(): rec[nm] = kappa_on(sub, c)
    rows1.append(rec)
r1 = pd.DataFrame(rows1).sort_values("all rows"); r1.to_csv(f"{OUT3}/tableR1_kappa_sensitivity.csv", index=False)
print("\nR1 — kappa_j (trips to half-credibility) under four row sets")
print(r1.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
sp = r1[["all rows", "event-free drivers (all-baseline rows)"]].replace(np.inf, np.nan).dropna()
print(f"\nSpearman(all rows, event-free drivers) = {sp.corr(method='spearman').iloc[0,1]:.3f}; "
      f"median ratio event-free/all = {np.median(sp.iloc[:,1]/sp.iloc[:,0]):.2f}")
spt = r1[["all rows", "trimmed: likely event rows removed"]].replace(np.inf, np.nan).dropna()
print(f"median ratio trimmed/all = {np.median(spt.iloc[:,1]/spt.iloc[:,0]):.2f}")

# ---------- R2: within-driver subsampling test on trimmed rows ----------
dft = df[~df.likely_event]
beh_t = dft.groupby("Participant_ID")[CLEAN].mean().reindex(drv.index)
kap_t = r1.set_index("indicator").loc[CLEAN, "trimmed: likely event rows removed"].values
M_GRID, M_MAX, N_DRAW = [1, 2, 3, 4, 6, 8, 12], 12, 25
cnt_t = dft.groupby("Participant_ID").size().reindex(drv.index).fillna(0)
keep_ids = drv.index[cnt_t >= M_MAX]; kpos = np.where(drv.index.isin(keep_ids))[0]
print(f"\nR2 — subsampling test on trimmed rows: {len(keep_ids)} drivers with >= {M_MAX} trimmed rows, {int(y[kpos].sum())} crashes")

def pois_slope_se(X, y, E, alpha=1e-3):
    m = pois_fit(X, y, E, alpha); mu = E * np.exp(m.intercept_ + X @ m.coef_)
    X1 = np.column_stack([np.ones(len(X)), X]); info = (X1 * mu[:, None]).T @ X1
    try: se = np.sqrt(np.linalg.inv(info)[-1, -1])
    except np.linalg.LinAlgError: se = np.nan
    return m.coef_[-1], se

TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s",
        "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
rng_w = np.random.default_rng(5); rows_w = []; kcmp = []
for c in TOP6:
    j = CLEAN.index(c)
    sub = dft[dft["Participant_ID"].isin(keep_ids)][["Participant_ID", c]]
    grp = sub.groupby("Participant_ID")[c].apply(lambda s: s.values).reindex(keep_ids); maxn = max(len(a) for a in grp)
    pad = np.full((len(keep_ids), maxn), np.nan)
    for r_, a in enumerate(grp): pad[r_, :len(a)] = a
    mu_c, sd_c = np.nanmean(beh_t[c]), np.nanstd(beh_t[c]) + 1e-9
    for m_ in M_GRID:
        sl, ses = [], []
        for d in range(N_DRAW):
            U = rng_w.random(pad.shape); U[np.isnan(pad)] = np.inf
            order = np.argsort(U, axis=1)[:, :m_]
            xm = np.nanmean(np.take_along_axis(pad, order, axis=1), axis=1); xm = np.where(np.isfinite(xm), xm, mu_c)
            b, se = pois_slope_se(np.column_stack([Xt[kpos], (xm - mu_c) / sd_c]), y[kpos], E[kpos]); sl.append(b); ses.append(se)
        rows_w.append(dict(indicator=c, m=m_, slope=np.mean(sl), se=np.nanmean(ses)))
    w = pd.DataFrame([r for r in rows_w if r["indicator"] == c]); mm, s_, wt = w.m.values, w.slope.values, 1 / w.se.values ** 2
    f_vc = mm / (mm + kap_t[j]); b_vc = (wt * s_ * f_vc).sum() / (wt * f_vc ** 2).sum(); chi_vc = (wt * (s_ - b_vc * f_vc) ** 2).sum()
    b_flat = (wt * s_).sum() / wt.sum(); chi_flat = (wt * (s_ - b_flat) ** 2).sum()
    kg = np.exp(np.linspace(np.log(0.05), np.log(200), 400)); prof = []
    for kk in kg:
        f = mm / (mm + kk); bb = (wt * s_ * f).sum() / (wt * f ** 2).sum(); prof.append((wt * (s_ - bb * f) ** 2).sum())
    prof = np.array(prof); inside = kg[prof <= prof.min() + 3.84]
    kcmp.append(dict(indicator=c, kappa_vc_trimmed=kap_t[j], kappa_from_crash_slopes=kg[prof.argmin()], k_lo=inside.min(), k_hi=inside.max(),
                     chi2_fixed_kappa=chi_vc, chi2_flat=chi_flat, beta_implied=b_vc, slope_m1=s_[0], slope_m12=s_[-1]))
att_t = pd.DataFrame(rows_w); att_t.to_csv(f"{OUT3}/tableR2_attenuation_trimmed.csv", index=False)
kc_t = pd.DataFrame(kcmp); kc_t.to_csv(f"{OUT3}/tableR2_kappa_crosscheck_trimmed.csv", index=False)
print(kc_t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# figure: trimmed data collapse next to the original one
kc0 = pd.read_csv(f"{OUT}/table5_kappa_crosscheck.csv"); att0 = pd.read_csv(f"{OUT}/table_attenuation_within.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (att, kc, ttl) in zip(axes, [(att0, kc0, "all rows (main analysis)"), (att_t, kc_t, "likely event trips removed (sensitivity)")]):
    for (ind, grp), mk, col in zip(att.groupby("indicator", sort=False), "osD^v<", [ORA, BLU, PUR, GRN, GRY, "#B5651D"]):
        r = kc[kc.indicator == ind].iloc[0]; kv = r.get("kappa_vc", r.get("kappa_vc_trimmed"))
        ax.errorbar(grp.m / kv, grp.slope / r.beta_implied, yerr=1.96 * grp.se / abs(r.beta_implied), fmt=mk, ms=5, color=col, capsize=3, lw=1, alpha=.9, label=f"{ind} (κ={kv:.1f})")
    xx = np.logspace(-1.4, 1.3, 300); ax.plot(xx, xx / (1 + xx), "k--", lw=1.5, label="C(x) = x/(1+x)")
    ax.set_xscale("log"); ax.set_xlim(0.04, 20); ax.set_ylim(-0.05, 1.3); ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("x = m / κ_j (log)"); ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("observed slope ÷ full-information slope")
fig.tight_layout(); fig.savefig(f"{OUT3}/figR2_collapse_trimmed.png", dpi=160); fig.savefig(f"{OUT3}/figR2_collapse_trimmed.pdf"); plt.close(fig)
print("done")
