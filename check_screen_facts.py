# -*- coding: utf-8 -*-
"""Facts for the worked example of the screen in Section 2.3 (round 13). Same preprocessing as shrp2_prelim.py.
Writes prelim_out/reviewer/screen_facts.json, read by make_tables.py. Add to the pipeline at the rerun.
usage: python3 -I check_screen_facts.py <SHRP2_merged_P4.csv> <table1_event_signature_screen.csv> <out.json>"""
import sys, json, numpy as np, pandas as pd
src, t1_p, out_p = sys.argv[1:4]
df = pd.read_csv(src, low_memory=False)
df = df.drop_duplicates(["Participant_ID", "Trip ID"]).reset_index(drop=True)   # one row per trip, as in shrp2_prelim.py
z = df["Mean Speed"] == 0
df.loc[z, ["Mean Speed", "Time at 70-80 mph", "Time at > 80 mph"]] = np.nan
for c in [c for c in df.columns if c.startswith("Minimum TTC")]:
    df.loc[df[c] >= 1e4, c] = 10.0; df[c] = df[c].clip(upper=10.0)
lead = df["Time With Lead Vehicle"].replace(0, np.nan)
for c in [c for c in df.columns if c.startswith("Time Where Headway")]:
    df[c] = (df[c] / lead).clip(0, 1)
dur = df["Trip Duration"].replace(0, np.nan)
for c in ["Time at 70-80 mph", "Time at > 80 mph", "Time With Lead Vehicle"]:
    df[c] = (df[c] / dur).clip(0, 1)
ids, d_idx = np.unique(df["Participant_ID"].values, return_inverse=True)
g = df.groupby("Participant_ID"); ev = ((g["N_crashes"].max() + g["N_near_crashes"].max()) > 0).reindex(ids).values
n_i = np.bincount(d_idx, minlength=len(ids)).astype(float)
acc = df["Number of Longitudinal Accels > Threshold"].dropna()
t1 = pd.read_csv(t1_p)
usable = t1.loc[~t1.degenerate, "variable"].tolist()            # variables with a usable tail
low_set = {"Maximum Deceleration"} | {c for c in df.columns if c.startswith("Minimum TTC")}
rng = np.random.default_rng(11); B = 2000
perms = np.array([rng.permutation(ev) for _ in range(B)])        # which drivers count as 'with an event', drivers kept whole
lo, hi = [], []
for c in usable:
    s = df[c]; low = c in low_set
    q = s.quantile(0.03) if low else s.quantile(0.97)
    ext = ((s <= q) if low else (s >= q)).values
    e_i = np.bincount(d_idx[ext], minlength=len(ids)).astype(float)
    null = (perms @ e_i / e_i.sum()) / (perms @ n_i / n_i.sum())
    lo.append(np.quantile(null, .025)); hi.append(np.quantile(null, .975))
facts = dict(n_trips=int(len(df)), n_noevent_drivers=int((~ev).sum()), n_noevent_trips=int(n_i[~ev].sum()),
             share_trips_event_drivers=float(n_i[ev].sum() / n_i.sum()), acc_count_zero_pct=float(100 * (acc == 0).mean()),
             n_usable=len(usable), null_lo_median=float(np.median(lo)), null_hi_median=float(np.median(hi)), B=B, seed=11)
json.dump(facts, open(out_p, "w"), indent=1)
print(facts)
