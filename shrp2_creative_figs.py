# ==================================================================
# Creative / headline figures for
# "How much driving is enough? Credibility-weighted behavioural risk
#  from naturalistic data"  -- built on the data prep of shrp2_prelim.py
# (run shrp2_prelim.py first: it writes the tables these figures read).
#
#   Fig A  universal credibility curve: six indicators collapse onto C = x/(1+x)
#   Fig B  funnel plots: raw driver means vs evidence, and after credibility weighting
#   Fig C  how much driving is enough: trips to half-credibility + share of drivers who have it
#   Fig D  watching credibility build up: three drivers' running mean vs credibility estimate
#   Fig E  who gets flagged: naive vs credibility-weighted risk quintiles (alluvial)
# ==================================================================
import io, contextlib, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle, Patch
from sklearn.linear_model import PoissonRegressor

src = open("shrp2_prelim.py").read()
head = src.split("# FIG 1 / attenuation test")[0]            # data prep, Table 1, Table 2, design matrices
with contextlib.redirect_stdout(io.StringIO()):
    exec(head)
OUT2 = f"{OUT}/creative"; os.makedirs(OUT2, exist_ok=True)
PUR, ORA, GRY, BLU, GRN = "#534AB7", "#D85A30", "#8A8A8A", "#2F6DB5", "#3B8C5A"

def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m

t2i = t2.set_index("indicator")
nice = {"Mean Speed": "Mean speed", "% Spd Lim 55-65": "Share of trip on 55–65 mph roads",
        "Time Where Headway 2.0 - 2.5 s": "Share of car-following at 2.0–2.5 s headway",
        "Time Where Headway > 3.5 s": "Share of car-following at > 3.5 s headway",
        "Time Where Headway 2.5 - 3.0 s": "Share of car-following at 2.5–3.0 s headway",
        "Time Where Headway 3.0 - 3.5 s": "Share of car-following at 3.0–3.5 s headway",
        "Time at > 80 mph": "Share of trip time above 80 mph", "Time at 70-80 mph": "Share of trip time at 70–80 mph",
        "Time With Lead Vehicle": "Share of trip time car-following", "Trip Duration": "Trip duration",
        "% Rur 2 Ln": "Share on rural 2-lane roads", "% Urb 2 Ln": "Share on urban 2-lane roads", "% Rur Frwy": "Share on rural freeways",
        "% Spd Lim 35 or Less": "Share on ≤35 mph roads", "% Spd Lim 40-50": "Share on 40–50 mph roads",
        "% Spd Lim 70 or Greater": "Share on ≥70 mph roads"}
fam = lambda c: ("road type" if c.startswith("% Urb") or c.startswith("% Rur") else "speed limit" if c.startswith("% Spd") else
                 "headway" if "Headway" in c or "Lead" in c else "speed" if "mph" in c or "Speed" in c else "trip")
FAMC = {"road type": GRN, "speed limit": BLU, "headway": PUR, "speed": ORA, "trip": GRY}

# ------------------------------------------------------------------
# FIG A: universal credibility curve (data collapse)
# ------------------------------------------------------------------
att_w = pd.read_csv(f"{OUT}/table_attenuation_within.csv"); kc = pd.read_csv(f"{OUT}/table5_kappa_crosscheck.csv")
fig, ax = plt.subplots(figsize=(8, 5.2))
for (ind, grp), mk, col in zip(att_w.groupby("indicator", sort=False), "osD^v<", [ORA, BLU, PUR, GRN, GRY, "#B5651D"]):
    r = kc[kc.indicator == ind].iloc[0]
    ax.errorbar(grp.m / r.kappa_vc, grp.slope / r.beta_implied, yerr=1.96 * grp.se / abs(r.beta_implied), fmt=mk, ms=6,
                color=col, capsize=3, lw=1, alpha=.9, label=f"{nice.get(ind, ind)}  (κ = {r.kappa_vc:.1f})")
xx = np.logspace(-1.4, 1.3, 300); ax.plot(xx, xx / (1 + xx), "k--", lw=1.6, label="credibility law  C(x) = x / (1 + x)")
ax.axhline(1, color="gray", lw=.6, ls=":"); ax.axhline(.5, color="gray", lw=.6, ls=":"); ax.axvline(1, color="gray", lw=.6, ls=":")
ax.text(1.05, 0.08, "x = 1: evidence equals κ\n→ half credibility", fontsize=8, color="gray")
ax.set_xscale("log"); ax.set_xlim(0.04, 20); ax.set_ylim(-0.05, 1.25)
ax.set_xlabel("behavioural evidence relative to its half-credibility point,  x = m / κ_j   (log scale)")
ax.set_ylabel("observed crash–behaviour slope ÷ full-information slope")
ax.set_title("Six behaviours, one law: how the crash association grows with evidence\n"
             "(κ_j measured from behaviour data alone; only the vertical scale of each indicator is fitted)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT2}/figA_universal_credibility_curve.png", dpi=170); fig.savefig(f"{OUT2}/figA_universal_credibility_curve.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# FIG B: funnel plots -- raw driver means vs evidence, then credibility-weighted
# ------------------------------------------------------------------
FUN = ["Mean Speed", "Time With Lead Vehicle", "Time Where Headway 2.0 - 2.5 s"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col")
for col_i, c in enumerate(FUN):
    n_i, m_i, ssw_i, _ = driver_summaries(c); s2, tau2, k = t2i.loc[c, ["sigma2_within", "tau2_between", "kappa"]]
    mu = m_i.mean(); Rn = n_i / (n_i + k); x_shr = mu + Rn * (m_i - mu)
    ns = np.linspace(1, n_i.max(), 300); R_ns = ns / (ns + k)
    for row, (vals, band, ttl) in enumerate([(m_i, np.sqrt(tau2 + s2 / ns), "raw driver mean"),
                                             (x_shr, np.sqrt(R_ns * tau2), "credibility-weighted  μ + C(n)(x̄ − μ)")]):
        ax = axes[row, col_i]
        ax.scatter(n_i + np.random.uniform(-.25, .25, len(n_i)), vals, s=6, alpha=.25, color=PUR if row else GRY, lw=0)
        ax.plot(ns, mu + 1.96 * band, color=ORA, lw=1.6); ax.plot(ns, mu - 1.96 * band, color=ORA, lw=1.6)
        ax.axhline(mu + 1.96 * np.sqrt(tau2), color="k", lw=.8, ls=":"); ax.axhline(mu - 1.96 * np.sqrt(tau2), color="k", lw=.8, ls=":")
        ax.axhline(mu, color="k", lw=.6); ax.set_xscale("log")
        ax.set_title(f"{nice[c]}\n{ttl}", fontsize=9)
        lo, hi = np.nanpercentile(m_i, [0.5, 99.5]); ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        if row == 1: ax.set_xlabel("evidence: trips sampled for the driver, n (log)")
        if col_i == 0: ax.set_ylabel("driver-level value")
    # premise check inset: binned Var(x̄ | n) against tau2 + sigma2/n
    ins = axes[0, col_i].inset_axes([0.60, 0.62, 0.37, 0.34])
    bins = [1, 2, 3, 4, 5, 7, 10, 15, 25, 1000]; bx, bv = [], []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        sel = (n_i >= lo_b) & (n_i < hi_b)
        if sel.sum() >= 15: bx.append(n_i[sel].mean()); bv.append(m_i[sel].var(ddof=1))
    ins.plot(bx, bv, "o", color=PUR, ms=4); ins.plot(ns, tau2 + s2 / ns, color=ORA, lw=1.3); ins.axhline(tau2, color="k", lw=.6, ls=":")
    ins.set_xscale("log"); ins.set_title("premise: Var(x̄ | n) = τ² + σ²/n", fontsize=7); ins.tick_params(labelsize=6)
axes[0, 0].text(0.02, 0.04, "orange: 95% funnel τ² + σ²/n;  dotted: ±1.96 τ (true between-driver spread)", transform=axes[0, 0].transAxes, fontsize=7)
axes[1, 0].text(0.02, 0.04, "orange: 95% band of the credibility estimate, C(n)·τ²", transform=axes[1, 0].transAxes, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT2}/figB_funnel_plots.png", dpi=170); fig.savefig(f"{OUT2}/figB_funnel_plots.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# FIG C: how much driving is enough?
# ------------------------------------------------------------------
tt = t2.sort_values("kappa").reset_index(drop=True); K_ = len(tt)
q25, q50, q75 = np.percentile(nb, [25, 50, 75])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
for i, r in tt.iterrows():
    col = FAMC[fam(r.indicator)]
    a1.plot([r.kappa_lo, r.kappa_hi], [i, i], color=col, lw=2, alpha=.5); a1.plot(r.kappa, i, "o", color=col, ms=7)
a1.set_yticks(range(K_)); a1.set_yticklabels([nice.get(c, c) for c in tt.indicator], fontsize=8.5)
a1.set_xscale("log"); a1.set_xlabel("trips needed for half-credibility, κ_j  (log scale, 95% driver-bootstrap CI)")
a1.axvspan(q25, q75, color="gray", alpha=0.12, lw=0); a1.axvline(q50, color="gray", ls=":", lw=1)
a1.text(q50, K_ - 0.45, f"evidence of a typical driver:\nmedian {q50:.0f} trips, IQR {q25:.0f}–{q75:.0f}", fontsize=7.5, color="dimgray", ha="center", va="top")
a1.set_ylim(-0.7, K_ - 0.3); a1.set_title("How many trips before a behaviour is half-credible?", fontsize=10)
a1.legend(handles=[Patch(color=v, label=k) for k, v in FAMC.items()], fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, frameon=False, title="indicator family", title_fontsize=7.5)
bands = [(0, .5, "#E9E3F5", "< 50%"), (.5, .75, "#B8ACE0", "50–75%"), (.75, .9, "#7F70C8", "75–90%"), (.9, 1.01, "#3D338F", "≥ 90%")]
for i, r in tt.iterrows():
    Rj = nb / (nb + r.kappa); left = 0
    for lo_b, hi_b, col, lab in bands:
        sh = ((Rj >= lo_b) & (Rj < hi_b)).mean(); a2.barh(i, sh, left=left, color=col, edgecolor="white", lw=.5, label=lab if i == 0 else None); left += sh
a2.set_yticks(range(K_)); a2.set_yticklabels([]); a2.set_xlim(0, 1); a2.set_xlabel("share of SHRP2 drivers (n = %d)" % len(nb))
a2.set_title("Share of drivers whose own evidence reaches this credibility", fontsize=10); a2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, title="reliability C(n) of the driver's own mean", title_fontsize=7.5)
a2.set_ylim(-0.7, K_ - 0.3)
fig.tight_layout(); fig.savefig(f"{OUT2}/figC_how_much_driving_is_enough.png", dpi=170); fig.savefig(f"{OUT2}/figC_how_much_driving_is_enough.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# FIG D: watching credibility build up for three drivers (mean speed)
# ------------------------------------------------------------------
c = "Mean Speed"; s2, tau2, k = t2i.loc[c, ["sigma2_within", "tau2_between", "kappa"]]; sig = np.sqrt(s2)
ok = df[df["Trip Day Number in Study"].notna() & df[c].notna()]
cnt = ok.groupby("Participant_ID").size(); cand = cnt[cnt >= 35].index
fm = ok[ok["Participant_ID"].isin(cand)].groupby("Participant_ID")[c].mean().sort_values()
picks = [fm.index[-1], fm.index[len(fm) // 2], fm.index[0]]; mu = beh[c].mean()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, pid, col, lab in zip(axes, picks, [ORA, GRN, BLU], ["A (fast)", "B (average)", "C (slow)"]):
    s = ok[ok["Participant_ID"] == pid].sort_values(["Trip Day Number in Study", "Trip ID"])[c].values
    kk = np.arange(1, len(s) + 1); run = np.cumsum(s) / kk; Rk = kk / (kk + k); est = mu + Rk * (run - mu)
    ax.fill_between(kk, run - 1.96 * sig / np.sqrt(kk), run + 1.96 * sig / np.sqrt(kk), color=GRY, alpha=.15)
    ax.plot(kk, run, color=GRY, lw=1.2, ls="--", label="raw running mean (±1.96 σ/√k)")
    ax.fill_between(kk, est - 1.96 * np.sqrt(Rk * s2 / kk), est + 1.96 * np.sqrt(Rk * s2 / kk), color=col, alpha=.18)
    ax.plot(kk, est, color=col, lw=2.2, label="credibility-weighted estimate (±1.96 posterior sd)")
    ax.axhline(mu, color="k", lw=.8, ls=":", label="population mean"); ax.plot(kk, s, ".", color="k", ms=3, alpha=.35, label="individual trips")
    for kq in [1, 3, 8, 20, len(s)]:
        if kq <= len(s): ax.annotate(f"C={kq/(kq+k):.2f}", (kq, est[kq - 1]), textcoords="offset points", xytext=(0, -14), fontsize=7, color=col, ha="center")
    ax.set_title(f"driver {lab}: {len(s)} trips in chronological order", fontsize=9); ax.set_xlabel("trips observed so far, k")
axes[0].set_ylabel(f"{nice[c]} (as recorded)"); axes[1].legend(fontsize=7, loc="lower right")
fig.suptitle(f"Watching credibility build up: after k trips the driver's mean gets weight C(k) = k/(k+κ), κ = {k:.1f} for mean speed", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT2}/figD_credibility_builds_up.png", dpi=170); fig.savefig(f"{OUT2}/figD_credibility_builds_up.pdf"); plt.close(fig)

# ------------------------------------------------------------------
# FIG E: who gets flagged? naive vs credibility-weighted behavioural risk quintiles
# ------------------------------------------------------------------
p_t = Xt.shape[1]
b_n = pois_fit(np.column_stack([Xt, Xb]), y, E, alpha=1e-2).coef_[p_t:]
b_r = pois_fit(np.column_stack([Xt, Xb_rc]), y, E, alpha=1e-2).coef_[p_t:]
s_naive, s_rc = Xb @ b_n, Xb_rc @ b_r
q_n = pd.qcut(s_naive, 5, labels=False).astype(int); q_r = pd.qcut(s_rc, 5, labels=False).astype(int)
terc = np.digitize(nb, np.quantile(nb, [1 / 3, 2 / 3]))
K = 5; cnts = np.zeros((K, K, 3))
for a, b, t in zip(q_n, q_r, terc): cnts[a, b, t] += 1
moved = pd.Series(q_n != q_r).groupby(terc).mean(); top_keep = [np.mean(q_r[(q_n == 4) & (terc == t)] == 4) for t in range(3)]
pd.DataFrame(dict(evidence_tercile=["low", "mid", "high"], share_changing_quintile=moved.values, share_of_naive_top_quintile_still_top=top_keep)) \
    .to_csv(f"{OUT2}/tableE_reclassification.csv", index=False)
print("reclassification by evidence tercile (low/mid/high): changed quintile =", np.round(moved.values, 3),
      "| naive top-quintile still top =", np.round(top_keep, 3))

gap = 0.012; scale = (1 - gap * (K - 1)) / cnts.sum()
lh, rh = cnts.sum(axis=(1, 2)) * scale, cnts.sum(axis=(0, 2)) * scale
ly0 = np.concatenate([[0], np.cumsum(lh + gap)[:-1]]); ry0 = np.concatenate([[0], np.cumsum(rh + gap)[:-1]])
loff, roff = {}, {}; lc, rc = ly0.copy(), ry0.copy()
for a in range(K):
    for b in range(K):
        for t in range(3): loff[(a, b, t)] = lc[a]; lc[a] += cnts[a, b, t] * scale
for b in range(K):
    for a in range(K):
        for t in range(3): roff[(a, b, t)] = rc[b]; rc[b] += cnts[a, b, t] * scale
TC = [ORA, "#B9B9B9", PUR]; x0, x1 = 0.16, 0.84
fig, ax = plt.subplots(figsize=(10, 7))
for a in range(K):
    for b in range(K):
        for t in range(3):
            h = cnts[a, b, t] * scale
            if h == 0: continue
            yl, yr = loff[(a, b, t)], roff[(a, b, t)]
            verts = [(x0, yl), (0.5, yl), (0.5, yr), (x1, yr), (x1, yr + h), (0.5, yr + h), (0.5, yl + h), (x0, yl + h), (x0, yl)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), facecolor=TC[t], edgecolor="none", alpha=0.85 if a != b else 0.10))
labs = ["Q1 lowest risk", "Q2", "Q3", "Q4", "Q5 highest risk"]
for a in range(K):
    ax.add_patch(Rectangle((x0 - 0.07, ly0[a]), 0.07, lh[a], color="#333")); ax.text(x0 - 0.08, ly0[a] + lh[a] / 2, labs[a], ha="right", va="center", fontsize=9)
    ax.add_patch(Rectangle((x1, ry0[a]), 0.07, rh[a], color="#333")); ax.text(x1 + 0.08, ry0[a] + rh[a] / 2, labs[a], ha="left", va="center", fontsize=9)
ax.text(x0 - 0.035, 1.03, "naive behavioural score\n(driver means taken at face value)", ha="center", fontsize=9.5, weight="bold")
ax.text(x1 + 0.035, 1.03, "credibility-weighted score\n(means shrunk by C(n) = n/(n+κ_j))", ha="center", fontsize=9.5, weight="bold")
ax.legend(handles=[Patch(color=TC[0], label=f"low-evidence drivers (n ≤ {np.quantile(nb, 1/3):.0f} trips): {moved.values[0]*100:.0f}% change quintile"),
                   Patch(color=TC[1], label=f"mid-evidence: {moved.values[1]*100:.0f}% change quintile"),
                   Patch(color=TC[2], label=f"high-evidence drivers (n > {np.quantile(nb, 2/3):.0f} trips): {moved.values[2]*100:.0f}% change quintile"),
                   Patch(color="#DDD", label="drivers staying in the same quintile (faded)")], fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.08, 1.10); ax.axis("off")
ax.text(0.5, -0.045, f"of drivers flagged in the naive top quintile, those still flagged after credibility weighting: "
        f"{top_keep[0]*100:.0f}% (low evidence)  |  {top_keep[1]*100:.0f}% (mid)  |  {top_keep[2]*100:.0f}% (high evidence)",
        ha="center", fontsize=8.5, color="#333")
fig.tight_layout(); fig.savefig(f"{OUT2}/figE_who_gets_flagged.png", dpi=170); fig.savefig(f"{OUT2}/figE_who_gets_flagged.pdf"); plt.close(fig)
print("done:", os.listdir(OUT2))
