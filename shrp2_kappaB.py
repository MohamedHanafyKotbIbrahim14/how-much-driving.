# Between-driver credibility parameter kappa_B (revised manuscript, equation "between"):
#   log E[Y_i] = log E_i + alpha + z'gamma + g log n_i + b * n_i/(n_i+kappa) * (xbar_i - mu_z,i)
# profile over kappa, driver bootstrap paired with kappa_z. Evidence counted two ways:
#   "baseline": the driver's baseline trips on which the indicator is observed (n_ik * n_base/n_rows), the
#               definition that does not contain the outcome (a crash adds a row);
#   "rows":     all rows, for the record (contaminated: n_i contains Y_i).
import io, contextlib, os, sys, time
import numpy as np, pandas as pd
t0 = time.time(); os.chdir("/home/claude/shrp2")
src = open("shrp2_prelim.py").read(); head = src.split("# FIG 1 / attenuation test")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(head)
OUTR = f"{OUT}/reviewer"; os.makedirs(OUTR, exist_ok=True)
B_KB = int(os.environ.get("B_KB", 500))
TOP6 = ["Mean Speed", "% Spd Lim 55-65", "Time Where Headway > 3.5 s", "Time Where Headway 2.0 - 2.5 s", "Time Where Headway 3.0 - 3.5 s", "Time Where Headway 2.5 - 3.0 s"]
def log(*a): print(*a, flush=True)
def irls(X, y, E, alpha=1e-3, iters=30):
    n, p = X.shape; X1 = np.column_stack([np.ones(n), X]); lam = alpha * E.sum(); P = np.eye(p + 1) * lam; P[0, 0] = 0
    beta = np.zeros(p + 1); beta[0] = np.log(y.sum() / E.sum()); logE = np.log(E)
    for _ in range(iters):
        eta = X1 @ beta + logE; mu = np.exp(np.clip(eta, -30, 30)); z = (eta - logE) + (y - mu) / mu
        A = (X1 * mu[:, None]).T @ X1 + P; b = (X1 * mu[:, None]).T @ z; new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < 1e-8: beta = new; break
        beta = new
    mu = np.exp(np.clip(X1 @ beta + logE, -30, 30))
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
    return beta, dev
def pct(a, q): a = np.asarray(a, float); a = a[np.isfinite(a)]; return np.percentile(a, q) if len(a) else np.nan
def two_sided_p(x, B): x = np.asarray(x, float); x = x[np.isfinite(x)]; p = 2 * min((x <= 0).mean(), (x >= 0).mean()); return min(1.0, max(p, 2.0 / (B + 1)))
J = {c: CLEAN.index(c) for c in CLEAN}; N = len(y)
scale = drv.n_base.values / drv.n_rows.values
EVID = {"baseline": np.clip(nk_mat * scale[:, None], 1, None), "rows": nk_mat}
KGB = np.exp(np.linspace(np.log(0.05), np.log(300), 45))
def profile(j, idx, ev):
    n_ = ev[idx, j]; d_ = (Xb[idx, j] - Mhat[idx, j]); Xt_ = Xt[idx]; y_ = y[idx]; E_ = E[idx]; ln = np.log(n_)
    devs, bs, gs = [], [], []
    for kk in KGB:
        beta, dv = irls(np.column_stack([Xt_, ln, n_ / (n_ + kk) * d_]), y_, E_); devs.append(dv); bs.append(beta[-1]); gs.append(beta[-2])
    devs = np.array(devs); i = int(np.argmin(devs)); return KGB[i], bs[i], gs[i], devs[i], devs[0], devs[-1]
rng = np.random.default_rng(31); boots = [rng.integers(0, N, N) for _ in range(B_KB)]
rows = []
for evname, ev in EVID.items():
    log(f"-- evidence: {evname}")
    for c in TOP6:
        j = J[c]
        n_i, m_i, ssw_i, ids_i = driver_summaries(c); pos_i = pos_of[ids_i].values
        fn = np.zeros(N); fm = np.zeros(N); fs = np.zeros(N); fn[pos_i] = n_i; fm[pos_i] = m_i; fs[pos_i] = ssw_i
        def kz_on(idx):
            ok = fn[idx] >= 1; ii = idx[ok]; return vc_z_from_summaries(fn[ii], fm[ii], fs[ii], Xt[ii])[2]
        kB, bB, gB, dbest, d0, dmax = profile(j, np.arange(N), ev); kz = kz_on(np.arange(N))
        reps = np.array([(profile(j, bi, ev)[0], kz_on(bi)) for bi in boots]); ld = np.log(reps[:, 0]) - np.log(reps[:, 1])
        r = dict(evidence=evname, indicator=c, kappa_z=kz, kappa_B=kB, kappa_B_lo=pct(reps[:, 0], 2.5), kappa_B_hi=pct(reps[:, 0], 97.5),
                 share_at_lower_bound=float(np.mean(reps[:, 0] <= KGB[0] * 1.001)), share_at_upper_bound=float(np.mean(reps[:, 0] >= KGB[-1] * 0.999)),
                 ratio_kB_kz=kB / kz, ratio_lo=np.exp(pct(ld, 2.5)), ratio_hi=np.exp(pct(ld, 97.5)), p_equal=two_sided_p(ld, B_KB),
                 b_at_kB=bB, g_logn=gB, dev_gain_vs_kappa0=d0 - dbest, dev_gain_vs_kappa300=dmax - dbest, B=B_KB)
        rows.append(r)
        log(f"  {c[:30]:30s} κ_z={kz:.1f} κ_B={kB:.1f} [{r['kappa_B_lo']:.1f},{r['kappa_B_hi']:.1f}] at-bounds {r['share_at_lower_bound']:.2f}/{r['share_at_upper_bound']:.2f} ratio={r['ratio_kB_kz']:.2f} [{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}] p={r['p_equal']:.2f} g={gB:+.2f} Δdev(κ→0)={r['dev_gain_vs_kappa0']:.1f} Δdev(κ=300)={r['dev_gain_vs_kappa300']:.1f}  [{time.time()-t0:.0f}s]")
pd.DataFrame(rows).to_csv(f"{OUTR}/tableR3b_kappa_between_glogn.csv", index=False)
log(f"done at {time.time()-t0:.0f}s")
