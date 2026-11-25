import io, contextlib, os
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor
os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read(); head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(head)
def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
nbase = drv.n_base.values.astype(float); scale = nbase / drv.n_rows.values
tb = np.quantile(nbase, [1/3, 2/3]); sb = np.digitize(nbase, tb)
print("baseline-trip tertiles: cut points", tb.round(1), "| crashes per driver by tertile (rows-based):", [round(y[strat==s].mean(),2) for s in range(3)], "| (baseline-based):", [round(y[sb==s].mean(),2) for s in range(3)])
rows = []; rng = np.random.default_rng(7)
for c in TOP6:
    j = CLEAN.index(c)
    for name, st in [("rows", strat), ("baseline", sb)]:
        for s in range(3):
            sel = st == s
            m = pois_fit(np.column_stack([Xt[sel], Xb[sel, j]]), y[sel], E[sel]).coef_[-1]
            bs = []
            for b in range(60):
                ii = rng.choice(np.where(sel)[0], sel.sum(), replace=True); bs.append(pois_fit(np.column_stack([Xt[ii], Xb[ii, j]]), y[ii], E[ii]).coef_[-1])
            nk_ev = nk_mat[sel, j] * (scale[sel] if name == "baseline" else 1)
            rows.append(dict(indicator=c, strata=name, tertile=["low","mid","high"][s], hmean_n=1/np.mean(1/np.clip(nk_ev,1,None)), slope=m, se=np.std(bs), crashes=int(y[sel].sum())))
r = pd.DataFrame(rows)
for c in TOP6:
    sub = r[r.indicator==c]
    print(f"\n{c}")
    print(sub.pivot(index="tertile", columns="strata", values=["slope","se","hmean_n","crashes"]).loc[["low","mid","high"]].to_string(float_format=lambda v: f"{v:.3f}"))
r.to_csv("prelim_out/reviewer/tableR6_tertiles_rows_vs_baseline.csv", index=False)
