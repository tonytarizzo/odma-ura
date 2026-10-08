# Supervisor discussion — 7 October 2026

Four-page meeting report for Deniz Gündüz. Read `report.pdf`; edit `report.tex`.
Evidence is frozen at 6 October: Job 033 complete (66/66), Job 032 161/174 with only D4 outstanding.
The report keeps original production results and a brief training-sensitivity caveat; detailed calibration diagnostics
remain in the local analysis. Its final page presents conceptual research questions rather than an experiment queue.

From the repository root:

```sh
uv run python docs/reports/2026-10-07_supervisor_meeting/build_figures.py
cd docs/reports/2026-10-07_supervisor_meeting
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
cp build/report.pdf report.pdf
```

`plot_data.json` is a frozen selection of aggregate results and bounds, with input paths and
SHA-256 hashes. No raw frames or checkpoints are bundled. The figure builder runs from that snapshot; use
`--refresh-data` only to deliberately refresh it from the locally audited inputs. Bounds are replotted from the saved
numerical evaluation, not recomputed by this script. The bound snapshot's older audit count is excluded: only its
numerical curves and interpretation metadata are reused. All plotted empirical groups contain all three seeds.

Figures are vector PDFs. The report and its curated figures are versionable; `build/` contains disposable compilation
and visual-QA output. The detailed source audit remains in
`jobs/033_matched_d2/results/analysis/review_2026-10-06/REPORT.md` (local results, not bundled here).
