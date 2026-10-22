#!/usr/bin/env bash
# One-command reproduction of every number in the manuscript (CPU only, ~1.5 h on one core).
# Resumable: each benchmark appends to results/*.csv and skips finished episodes.
set -euo pipefail
cd "$(dirname "$0")"
python checks.py                      # 10 sanity checks (must all pass)
python bench.py main                  # 810 episodes  -> results/main_episodes.csv
python bench.py sweep                 # 240 episodes  -> results/sweep_episodes.csv
python bench.py sweep_person          # 120 episodes  -> results/sweep_person_episodes.csv
python joint_stress.py                # 80 episodes   -> results/sweep_joint_episodes.csv
python analyze.py                     # Tables III-V, Figs. 2-4 -> results/summary.json
python batch2.py                      # sensitivity   -> results/batch2_*.csv
python analyze_batch2.py              # Table VII, cluster bootstrap -> results/batch2_summary.json
python batch3.py                      # mechanism costs -> results/batch3_rows.csv
python summarize_batch3.py            # Table VIII -> results/batch3_summary.json
python make_figures.py                # Figs. 1-5 (diagrams + colour results)
python verify_paper.py                # compares all 56 paper values
(cd phase9 && python gen_video_data.py && python build_video.py)   # results video
echo "All done."
