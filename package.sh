#!/bin/bash
# Packages the deliverables after a build (paper/build.sh):
# deliver/{main.pdf, main.tex, supplement.pdf, supplement.tex, highlights.txt, paper_source.zip, code.zip, figures.zip, reviewer_checks.zip}
set -e
cd /home/claude/shrp2
rm -rf deliver figures_for_download; mkdir -p deliver figures_for_download
cp paper/main.pdf paper/main.tex paper/supplement.pdf paper/supplement.tex paper/highlights.txt deliver/
cp paper/figures/graphical_abstract.tiff paper/figures/graphical_abstract.pdf deliver/          # uploaded as a separate file
# figure files with their numbers: Figure 1, 2, ... in the order of main.tex, then Figure S1, S2, ... in the order of supplement.tex
python3 - << 'EOF'
import re, shutil, os
for doc, pre in [("main", ""), ("supplement", "S")]:
    src = open(f"paper/{doc}.tex").read()
    for i, f in enumerate(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/(fig_\w+)\.pdf\}", src), 1):
        shutil.copy(f"paper/figures/{f}.pdf", f"figures_for_download/Figure{pre}{i}_{f[4:]}.pdf")
print(sorted(os.listdir("figures_for_download")))
EOF
(cd figures_for_download && zip -q ../deliver/figures.zip *)
# LaTeX sources: the files that main.tex and supplement.tex read, and the scripts that write the tables, the numbers and the figures
(cd paper && zip -q -r ../deliver/paper_source.zip main.tex supplement.tex main.bbl supplement.bbl highlights.txt build.sh make_tables.py make_figs.py \
    make_graphical_abstract.py tables/numbers.tex tables/supp_labels.tex tables/floats $(grep -ho 'figures/fig_[A-Za-z_]*\.pdf' main.tex supplement.tex | sort -u) \
    $( [ -f references.bib ] && echo references.bib ))
zip -q deliver/code.zip shrp2_prelim.py shrp2_robustness.py shrp2_reviewer_checks.py shrp2_kappaB.py shrp2_r5_baseline_weights.py shrp2_r6_tertiles_baseline.py shrp2_r7_final_checks.py shrp2_only_this_file.py shrp2_creative_figs.py run_all.sh check_hardcoded.py check_screen_facts.py check_screen_tail.py check_event_trips.py compare_numbers.py package.sh paper/build.sh paper/make_tables.py paper/make_figs.py paper/make_graphical_abstract.py
# aggregate results and logs only: the two files with one row per driver stay out (participant-level data)
zip -q -r deliver/reviewer_checks.zip prelim_out/reviewer prelim_out/rerun_*.log prelim_out/zero_speed.json prelim_out/d_variance_ratios.json \
    -x prelim_out/reviewer/test_scores.csv prelim_out/reviewer/severity_decode.csv
ls -la deliver
