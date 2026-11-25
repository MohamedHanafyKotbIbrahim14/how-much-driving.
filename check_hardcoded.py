# Checks the hand-written statements of the manuscript that depend on the data (run after the pipeline).
import io, contextlib, os, json, re, numpy as np, pandas as pd
os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read(); head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(head)
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
b1 = pd.read_csv("prelim_out/robustness/tableB1_bootstrap_by_outcome.csv"); pr = b1[b1.outcome == b1.outcome.iloc[0]].set_index("indicator")
print("outcome used:", b1.outcome.iloc[0])
# 1. kappa_z on D vs all drivers; variance ratios
keep = drv.index[drv.n_rows >= 12]; kpos = np.where(drv.index.isin(keep))[0]
print("\n[B.2] kappa_z on D vs all; within-driver variance ratio (D/all); between-driver variance ratio (D/all):")
larger = 0; vr = {}
for c in TOP6:
    n, m_, ssw, ids = driver_summaries(c); pos = pos_of[ids].values; inD = np.isin(pos, kpos)
    s2a, t2a, kza, _ = vc_z_from_summaries(n, m_, ssw, Xt[pos]); s2d, t2d, kzd, _ = vc_z_from_summaries(n[inD], m_[inD], ssw[inD], Xt[pos[inD]])
    larger += kzd > kza; vr[c] = dict(kz_all=float(kza), kz_D=float(kzd), sigma2_ratio=float(s2d / s2a), tau2z_ratio=float(t2d / t2a))
    print(f"  {c[:30]:30s} kz_all {kza:.2f}  kz_D {kzd:.2f}  sigma2 D/all {s2d/s2a:.3f}  tau2_z D/all {t2d/t2a:.3f}")
print("  kappa_z larger on D for", larger, "of 6")
json.dump(vr, open("prelim_out/d_variance_ratios.json", "w"), indent=1)
# 2. 'just under a fifth' / 'almost three quarters' for mean speed on D
kz = pr.loc["Mean Speed", "kappa_z"]; print(f"\n[4.3] mean speed kappa_z on D = {kz:.2f}: 1/(1+kz) = {1/(1+kz):.3f}, 12/(12+kz) = {12/(12+kz):.3f}")
# 3. grouping: k-means etc. is computed in make_tables. 4. slopes at m=1 / m=12 for mean speed
print(f"[4.3] mean speed slopes: m=1 {pr.loc['Mean Speed','slope_m1']:+.3f}, m=12 {pr.loc['Mean Speed','slope_m12']:+.3f}; ratio {pr.loc['Mean Speed','ratio_m12_m1']:.2f}, predicted {pr.loc['Mean Speed','predicted_ratio']:.2f}")
# 5. headway growth
print("[4.3] growth factors m12/m1:", {c[:22]: round(pr.loc[c, 'ratio_m12_m1'], 1) for c in TOP6})
# 6. sign of slopes, 5 negative 1 positive?
print("[4.3] signs at m=12:", {c[:22]: np.sign(pr.loc[c, 'slope_m12']) for c in TOP6})
# 7. which indicators have the strongest association (TOP6 still the top six?)
b5 = pd.read_csv("prelim_out/robustness/tableB5_deattenuation_by_outcome.csv"); print("[4.3] indicators in B5:", b5[b5.outcome == b5.outcome.iloc[0]].indicator.tolist())
# 8. zero speed facts
print("[2] zero speed:", json.load(open("prelim_out/zero_speed.json")))
# 9. Table 2 kappa values (all 16)
t2 = pd.read_csv("prelim_out/table2_reliability_kappa.csv").sort_values("kappa"); print("\n[Table 3] kappa by indicator:"); print(t2[["indicator", "kappa", "kappa_lo", "kappa_hi", "kappa_z"]].round(2).to_string(index=False))
