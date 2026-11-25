# -*- coding: utf-8 -*-
"""How much the likely event trips differ from the other trips (Appendix B.3, "Event-linked trips").
The likely event trips are chosen exactly as in shrp2_robustness.py: that script is run up to the line that marks them.
The difference of a variable is (mean on the likely event trips - mean on the other trips) / standard deviation over all
trips, for the eight variables of the index and for the sixteen indicators. Writes prelim_out/reviewer/event_trip_diffs.json.
usage: python3 -I check_event_trips.py   (aggregates only)"""
import io, contextlib, os, json
import numpy as np, pandas as pd
os.chdir("/home/claude/shrp2")
src = open("shrp2_robustness.py").read()
cut = "# ---------- fast Poisson IRLS"
assert src.count(cut) == 1
ns = {"__name__": "check_event_trips"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src.split(cut)[0], "shrp2_robustness.py (head)", "exec"), ns)
df, SIG, CLEAN = ns["df"], ns["SIG"], ns["CLEAN"]
ev = df["likely_event"].values
def d(c):
    v = df[c].astype(float)
    return float((v[ev].mean() - v[~ev].mean()) / v.std())
out = {"n_likely_event_trips": int(ev.sum()), "n_trips": int(len(df)),
       "index_variables": {c: d(c) for c in SIG}, "indicators": {c: d(c) for c in CLEAN}}
os.makedirs("prelim_out/reviewer", exist_ok=True)
json.dump(out, open("prelim_out/reviewer/event_trip_diffs.json", "w"), indent=1)
iv = np.array([abs(v) for v in out["index_variables"].values()]); ind = out["indicators"]
big = max(ind, key=lambda c: abs(ind[c]))
print(f"likely event trips {out['n_likely_event_trips']} of {out['n_trips']}; index variables |d| {iv.min():.2f} to {iv.max():.2f}; "
      f"largest indicator {big} {ind[big]:+.2f}; mean speed {ind['Mean Speed']:+.2f}")
