# -*- coding: utf-8 -*-
"""Does the screen depend on the 3% tail? Same preprocessing and rule as shrp2_prelim.py (Table 1 of the code).
usage: python3 -I check_screen_tail.py <SHRP2_merged_P4.csv> [<out.json>]   (aggregates only; the json is read by make_tables.py)"""
import sys, json, numpy as np, pandas as pd
src = sys.argv[1]
raw = pd.read_csv(src, low_memory=False)
def prep(df):
    df = df.copy()
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
    g = df.groupby("Participant_ID")
    ev = (g["N_crashes"].max() + g["N_near_crashes"].max()) > 0
    df["event_driver"] = df["Participant_ID"].map(ev)
    return df
TRIP_VARS = ["Trip Duration", "Mean Speed", "Maximum Acceleration", "Maximum Deceleration", "Maximum Lateral Acceleration",
             "Number of Longitudinal Accels > Threshold", "Number of Longitudinal Decels > Threshold", "Number of Lateral Accels > Threshold",
             "Brake Activations", "Trip Distance Origin to Destination", "ABS Activation", "Time at 70-80 mph", "Time at > 80 mph",
             "Time With Lead Vehicle", "Time Where Headway 0.0 - 0.5 s", "Time Where Headway 0.5 - 1.0 s", "Time Where Headway 2.0 - 2.5 s",
             "Time Where Headway 2.5 - 3.0 s", "Time Where Headway 3.0 - 3.5 s", "Time Where Headway > 3.5 s",
             "Minimum TTC Lead 0-10 mph", "Minimum TTC Lead 10-20 mph", "Minimum TTC Lead 20-30 mph", "Minimum TTC Lead 30-40 mph",
             "Minimum TTC Lead 40-50 mph", "Minimum TTC Lead >= 50 mph", "% Urb Frwy", "% Urb 2 Ln", "% Rur Frwy", "% Rur 2 Ln",
             "% Spd Lim 35 or Less", "% Spd Lim 40-50", "% Spd Lim 55-65", "% Spd Lim 70 or Greater"]
RISKY_LOW = {c for c in TRIP_VARS if c.startswith("Minimum TTC")} | {"Maximum Deceleration"}
def screen(df, tail):
    share = df["event_driver"].mean(); out = {}
    for c in TRIP_VARS:
        s = df[c]; low = c in RISKY_LOW or (s.min() < 0 and "Decel" in c)
        q = s.quantile(tail) if low else s.quantile(1 - tail)
        ext = (s <= q) if low else (s >= q)
        if ext.sum() < 50: continue
        r = df.loc[ext, "event_driver"].mean() / share
        out[c] = "mass" if ext.mean() > 0.10 else ("sig" if r > 1.08 else "clean")
    return share, out
base_share, base = screen(prep(raw), 0.03)
keep0 = sorted(c for c, v in base.items() if v == "clean")
print(f"paper (3 percent, all rows): share of trips of event drivers {base_share:.3f}, clean {len(keep0)}")
res = {"n_vars": len(TRIP_VARS), "versions": {}}
for label, df in [("all rows", prep(raw)), ("one row per trip", prep(raw.drop_duplicates(["Participant_ID", "Trip ID"])))]:
    _, ref = screen(df, 0.03); ref_keep = sorted(c for c, v in ref.items() if v == "clean"); res["versions"][label] = {}
    for tail in [0.01, 0.02, 0.03, 0.05, 0.10]:
        share, out = screen(df, tail)
        keep = sorted(c for c, v in out.items() if v == "clean")
        res["versions"][label][f"{tail:.2f}"] = {"n_pass": len(keep), "n_fail": len(TRIP_VARS) - len(keep), "same_as_3pct": keep == ref_keep,
                                                 "dropped": sorted(set(ref_keep) - set(keep)), "added": sorted(set(keep) - set(ref_keep))}
        print(f"{label:17s} tail {tail:4.0%}: share {share:.3f} | clean {len(keep):2d} | dropped vs paper: {sorted(set(keep0) - set(keep))} | added: {sorted(set(keep) - set(keep0))}")
if len(sys.argv) > 2:
    json.dump(res, open(sys.argv[2], "w"), indent=1); print("written", sys.argv[2])
