import io, contextlib, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor
os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read(); head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(head)
def pois_fit(X, y, E, alpha=1e-3):
    m = PoissonRegressor(alpha=alpha, max_iter=3000); m.fit(X, y / E, sample_weight=E); return m
facts = json.load(open("prelim_out/robustness/facts.json"))
print("event rows share (all / D):", facts["event_rows_share_all"], facts["event_rows_share_D"])
print("corr(rows, crashes) =", np.corrcoef(drv.n_rows, drv.crashes)[0,1].round(3), "| corr(baseline rows, crashes) =", np.corrcoef(drv.n_base, drv.crashes)[0,1].round(3))
# credibility weights from baseline-scaled evidence
scale = (drv.n_base.values / drv.n_rows.values)[:, None]
nk_base = np.clip(nk_mat * scale, 1, None)
Rz_b = nk_base / (nk_base + kap_z[None, :]); Xb_rcz_b = Mhat + Rz_b * (Xb - Mhat)
def rc_fit(Xcal, Rw, j, yv, Ev, iters=3):
    off = np.zeros(len(yv)); b = np.nan
    for _ in range(iters):
        m = pois_fit(np.column_stack([Xt, Xcal[:, j]]), yv, Ev * np.exp(off)); b = m.coef_[-1]
        off = 0.5 * b ** 2 * tau2z_std[j] * (1 - Rw[:, j])
    return b
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
rows = []
for c in TOP6:
    j = CLEAN.index(c)
    b_raw = pois_fit(np.column_stack([Xt, Xb[:, j]]), y, E).coef_[-1]
    b_rc = rc_fit(Xb_rcz, Rz, j, y, E); b_rcb = rc_fit(Xb_rcz_b, Rz_b, j, y, E)
    rows.append(dict(indicator=c, raw=b_raw, rc_rows=b_rc, rc_baseline=b_rcb, under_rows=100*(1-b_raw/b_rc), under_baseline=100*(1-b_raw/b_rcb), mean_C_rows=Rz[:, j].mean(), mean_C_base=Rz_b[:, j].mean()))
pd.DataFrame(rows).to_csv("prelim_out/reviewer/tableR5_baseline_weights_deatten.csv", index=False); print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
# model comparison, 5 splits, linear models, both weightings
def dev_mean(eta, yv, Ev):
    mu = Ev * np.exp(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(2 * (np.where(yv > 0, yv * np.log(yv / mu), 0) - (yv - mu))))
recs = []
for split in range(5):
    rng = np.random.default_rng(100 + split); idx = rng.permutation(len(y)); nte = len(y) // 4; te, tr = idx[:nte], idx[nte:]
    spf = pois_fit(Xt[tr], y[tr], E[tr], alpha=1e-2)
    for name, X_ in [("raw means", Xb), ("weighted (rows)", Xb_rcz), ("weighted (baseline rows)", Xb_rcz_b)]:
        m = pois_fit(np.column_stack([Xt[tr], X_[tr]]), y[tr], E[tr], alpha=1e-2); eta = m.intercept_ + np.column_stack([Xt[te], X_[te]]) @ m.coef_
        eta0 = spf.intercept_ + Xt[te] @ spf.coef_
        recs.append(dict(split=split, model=name, dev_all=dev_mean(eta, y[te], E[te]), dev0_all=dev_mean(eta0, y[te], E[te]),
                         **{f"dev_{s}": dev_mean(eta[strat[te]==si], y[te][strat[te]==si], E[te][strat[te]==si]) for si, s in enumerate(["low","mid","high"])},
                         **{f"dev0_{s}": dev_mean(eta0[strat[te]==si], y[te][strat[te]==si], E[te][strat[te]==si]) for si, s in enumerate(["low","mid","high"])},
                         gain_all=100*(1-dev_mean(eta, y[te], E[te])/dev_mean(eta0, y[te], E[te])),
                         **{f"gain_{s}": 100*(1-dev_mean(eta[strat[te]==si], y[te][strat[te]==si], E[te][strat[te]==si])/dev_mean(eta0[strat[te]==si], y[te][strat[te]==si], E[te][strat[te]==si])) for si, s in enumerate(["low","mid","high"])}))
pd.DataFrame(recs).to_csv("prelim_out/reviewer/tableR5_baseline_weights_models_splits.csv", index=False)   # per split, for the s.d. across splits
g_ = pd.DataFrame(recs).groupby("model", sort=False).mean().drop(columns="split")
for s_ in ["all", "low", "mid", "high"]: g_[f"gain_avgdev_{s_}"] = 100 * (1 - g_[f"dev_{s_}"] / g_[f"dev0_{s_}"])
g_.to_csv("prelim_out/reviewer/tableR5_baseline_weights_models.csv"); print(g_.to_string(float_format=lambda v: f"{v:+.1f}"))
