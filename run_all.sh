#!/bin/bash
# Full pipeline: one row per trip (duplicate trips removed in shrp2_prelim.py), then the checks the paper reads. Logs in prelim_out/rerun_*.log
set -e
cd /home/claude/shrp2
mkdir -p prelim_out/reviewer
echo "START $(date)"
python3 shrp2_prelim.py > prelim_out/rerun_prelim.log 2>&1;            echo "prelim done $(date)"
python3 -I check_screen_facts.py SHRP2_merged_P4.csv prelim_out/table1_event_signature_screen.csv prelim_out/reviewer/screen_facts.json > prelim_out/rerun_screen_facts.log 2>&1; echo "screen facts done $(date)"
python3 -I check_screen_tail.py SHRP2_merged_P4.csv prelim_out/reviewer/screen_tail.json > prelim_out/rerun_screen_tail.log 2>&1; echo "screen tail done $(date)"
python3 -I check_event_trips.py > prelim_out/rerun_event_trips.log 2>&1; echo "event trips done $(date)"
B_BOOT=2000 B_SENS=500 python3 shrp2_robustness.py > prelim_out/rerun_robustness.log 2>&1;   echo "robustness done $(date)"
B_SENS=500 B_KB=500 python3 shrp2_reviewer_checks.py > prelim_out/rerun_reviewer.log 2>&1;   echo "reviewer checks done $(date)"
B_KB=500 python3 shrp2_kappaB.py > prelim_out/rerun_kappaB.log 2>&1;    echo "kappaB done $(date)"
python3 shrp2_r5_baseline_weights.py > prelim_out/rerun_r5.log 2>&1;   echo "r5 done $(date)"
python3 shrp2_r6_tertiles_baseline.py > prelim_out/rerun_r6.log 2>&1;  echo "r6 done $(date)"
B_Q=2000 B_SENS=500 B_GRP=1000 python3 shrp2_r7_final_checks.py > prelim_out/rerun_r7.log 2>&1; echo "r7 done $(date)"
python3 shrp2_only_this_file.py > prelim_out/rerun_only_this_file.log 2>&1; echo "only_this_file done $(date)"
python3 shrp2_creative_figs.py > prelim_out/rerun_creative.log 2>&1; echo "creative figs done $(date)"
python3 check_hardcoded.py > prelim_out/rerun_check_hardcoded.log 2>&1; echo "checks done $(date)"
cd paper && ./build.sh > ../prelim_out/rerun_build.log 2>&1;            echo "build done $(date)"
echo "ALL DONE $(date)"
