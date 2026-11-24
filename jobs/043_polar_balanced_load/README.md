# Job 043 — required energy versus active users with the balanced encoder

The plotted quantity is the per-user physical $E_b/N_0$ in dB needed for 5% missed-copy PUPE, against the number of active users $K_a=50,100,150,200,250,300$. A point on this graph is a threshold estimated from a multi-energy receiver experiment. At each energy both catalogue defaults decode the same real Gaussian multiple-access observation independently, with unknown placements/messages and known load and noise. The job records membership misses, false copies, complete count actions and geometry-inclusive time as well.

The original balanced B100 encoder stays fixed at all six loads: 13 placement bits, 87 payload bits, CRC16, polar length 512, 8,192 placements, and 30,000 real channel uses. Every word has unit energy, so the real noise variance is $[2\,100\,10^{(E_b/N_0)_{\rm dB}/10}]^{-1}$. IID messages allow repeats. The positions and balanced amplitudes are generated from each seed. With a fixed seed and load, the messages, standardized noise and complete encoder are shared across energies and receiver arms. Different seeds define independent geometries. In particular the 50-user curve uses this same 13-bit placement map; the earlier 50-user paper control used 12. Özates et al. change their split, block length or user powers with load. Their published points and the theoretical references have separate labels. [Figure definitions and numerical sources](figure_notes.md)

## Search and confirmation

1. Use 16 scouting geometries per load. Start at −1.50, 0, 1.50, 4, 8 and 16 dB. Three rounds bisect the interval between the last empirical failing energy and the next passing energy for each receiver. Both run on the union of the proposed energies. Interval widths of at most 0.10 dB stop refinement. If the initial scan is open at an edge, one extension is available to −3 or +30 dB. A pass followed by a failure leaves an explicit witness pair for confirmation.
2. For each bracket, retain its endpoints, estimated crossing rounded to 0.05 dB, and offsets of ±0.20 and ±0.50 dB. For a still-open edge, test the edge plus two points inside the searched range. Take the union across receivers and add any pilot recrossing witnesses. Freeze this grid before reading confirmation outcomes.
3. Use **256 independent new geometries per load and energy** on that grid. The plotted point is the first persistent downward crossing of 5%, interpolated in dB between adjacent observed means. A curve that stays above or below target yields a one-sided limit. A pass followed by a fail is marked ambiguous. Save the whole evaluated curve, observed bracket and 2,000 paired frame-bootstrap draws' status counts and interval for each point. The resampling unit is a complete geometry, shared across both decoders and energies. Intervals are pointwise by load and conditional on the selected energy grid. A between-grid threshold assumes approximate linearity of PUPE over that bracket.

Job 042 studies flat-encoder thresholds at two loads and a three-energy balanced comparison at 100 users. Job 043 is independently runnable and uses a disjoint seed bank. Neither pilot nor local preview frames enter its 256-frame estimates. The graph also shows a matched 5% Polyanskiy random-coding achievability reference, a genie/list-Fano necessary-energy reference and the published Özates curve. The achievability result gives sufficient energy for an ensemble; a practical receiver can use less.

## HPC

In the HPC checkout, update the branch and environment and submit from the repository root.

```sh
git pull --ff-only origin main
uv sync --frozen
bash jobs/043_polar_balanced_load/submit.sh
```

The wrapper issues six dependent PBS submissions, prints their IDs, and leaves array concurrency to the scheduler. Four arrays of 48 tasks scout, one array of 768 tasks confirms, and the final job merges observations and writes the figure. Each simulation task handles two seeds for one load across its selected energies. A task requests one CPU, 12 GB and 12 hours. Jobs checkpoint individual physical observations with atomic writes and check the earlier release fingerprints. Re-running the wrapper resumes completed cells. The first six pilot energies require 576 paired observations. As a scale example, seven confirmation energies at every load would require 10,752 paired observations or 21,504 whole receiver calls in confirmation. The actual energy count and wall time depend on the pilot and the machine.

Outputs live in `results/dec43_balanced_load/`. The merge writes `required_ebn0_vs_load.pdf`, its PNG, the six-load `summary.json` and the numerical references. `plans/` records the pilot grids and integer miss totals. `confirm/` retains per-frame count actions and cost receipts. A partial grid cannot produce the completed six-load plot.

The merge also writes `grid_precision.json`. For each bootstrap draw the extra calculation retains both tested sides of the crossing. Its percentile interval therefore reflects the spacing of the energy grid as well as variation across frames. Open limits and recrossings remain visible. The paired difference uses the same resampled geometries for both receivers. If a previously submitted merge runs the earlier shell wrapper, the identical calculation can be run after the results arrive with `uv run python jobs/043_polar_balanced_load/grid_precision.py`. It reads the saved confirmation frames and makes zero additional decoder calls.

The three related HPC studies have independent output directories and submission chains.

| Study | Scientific output | Merged return |
|---|---|---|
| 039 | Flat DEC-35 paper curves, computation and feedback controls | `jobs/039_polar_paper/results/summary.json` |
| 042 | Flat DEC-38/39 comparisons and the three-energy K100 balancing control | `results/dec42_paper_curve/summary.json` |
| 043 | Balanced DEC-38/39 required energy versus six active-user loads | `results/dec43_balanced_load/summary.json` |

The DEC-35 terminal confirmation from Job 037 already has 128 checked local outcomes in `results/polar_dec35_exploration_2026-11-11/paper_position/terminal_local/`. Studies 040 and 041 ran locally. Before submitting a 039, 042 or 043 chain, check the live queue (`qstat -u "$USER"`) and the merged return on the cluster. An array whose raw frames are already complete can proceed directly to its reducer. The submission commands for 039 and 042 are in their [paper study](../039_polar_paper/README.md) and [curve protocol](../042_polar_combinations/curve_design.md).

The complete six-load calculation also runs on a local machine with `bash jobs/043_polar_balanced_load/run_local.sh 4` after `uv sync --frozen`. The argument sets the number of local worker processes. Local and PBS execution use the same search, release, independent confirmation seeds, and resumable observation files. For illustration, seven confirmation energies per load mean 10,752 paired observations. Scaling the separate one-frame-per-load local preflights gives roughly 57–61 serial worker-hours for that confirmation grid, before scouting. Actual cost depends on the final grids, load-specific decoding trajectories, and the machine.

## Local preview

Eight independent geometries at each load, tested at four fixed energies, give a separately labelled exploratory figure. Its seed ranges are disjoint from both HPC stages.

| Active users | Tested energies (dB) | DEC-38 misses at those energies | DEC-39 misses at those energies | Estimated 5% energy, DEC-38 / DEC-39 |
|---:|---|---|---|---|
| 50 | −1.0, −0.5, 0, 0.5 | 286, 158, 22, 1 of 400 | 286, 158, 20, 0 of 400 | 0.05 / 0.00 dB |
| 100 | −0.5, 0, 0.25, 0.5 | 705, 285, 12, 4 of 800 | 714, 343, 13, 4 of 800 | 0.22 / 0.23 dB |

At K100 both crossings lie between actual evaluations at 0 and 0.25 dB. The close interpolated energies depend on the slope between those two points. On this laptop the two receivers averaged about 6.1 seconds each per observation at K50 and 10.8 seconds at K100, charging each its cold geometry. These are local pilot costs across four energies. Separate one-frame execution checks at K300 found 274/289 misses at +10 dB and 0/266 at +20 dB for DEC-38/39 on the same geometry. They motivate extending the scout bracket at high load and reporting a censored threshold where necessary.

To regenerate the two-load preview locally:

```sh
for i in $(seq 1 8); do uv run python jobs/043_polar_balanced_load/load_curve.py --preview --shard "$i"; done
uv run python jobs/043_polar_balanced_load/load_curve.py --merge-preview
```

The outputs are `load_preview.pdf`, `load_preview.png` and `preview_summary.json`. Use `uv run python jobs/043_polar_balanced_load/load_curve.py --inventory` to verify the release. The targeted checks run with `uv run python -m unittest discover -s jobs/043_polar_balanced_load -p 'test_*.py'`.

## Twelve-frame refined preview

Retain the eight earlier geometries and add the next four consecutive preview seeds at each of K50 and K100. Run both complete receivers on each observation. On the original four energies per load, re-estimate the 5% crossing with all twelve. If its tested bracket exceeds 0.05 dB, run the midpoint on the same twelve geometries. Both arms see the union of the two requested midpoints. Allow four refinement rounds. For an exact grid hit, check unmeasured neighbors 0.05 dB to each side within the initial range. Record censored or ambiguous curves without an extrapolated point. These frames determine the exploratory grid as well as its estimates. Job 043's independent 256-frame confirmation has a separate seed bank.

| Load and receiver | Original 8, coarse grid | Same coarse grid, 12 | Refined grid, 12 | Final tested crossing bracket |
|---|---:|---:|---:|---|
| 50, DEC-38 | 0.048 | 0.078 | 0.038 | [0.03, 0.06] dB |
| 50, DEC-39 | 0.000 | 0.050 | 0.023 | [0.00, 0.03] dB |
| 100, DEC-38 | 0.224 | 0.217 | 0.126 | [0.12, 0.15] dB |
| 100, DEC-39 | 0.230 | 0.231 | 0.146 | [0.12, 0.15] dB |

All estimates are in dB. On the K100 panel, at 0.12 dB the two receivers missed 64 and 103 of 1,200 transmissions, and at 0.15 dB they missed 44 and 53. The 5% cutoff is 60 copies. The spacing of the earlier measurements therefore had a larger effect on the K100 estimate than adding four frames. Both K100 grid-aware bootstrap intervals are [0, 0.18] dB. A 0.03 dB search bracket on the observed panel still leaves appreciable population uncertainty. The 12-frame comparison does not resolve a ranking.

The original eight-geometry two-load simulation took 4 min 5 s of wall time with four local workers (15 min 44 s of summed receiver and unique geometry time). The extension cost another 8 min 11 s with four workers to evaluate 116 new paired observations and check the existing 64. The resulting figure uses 180 paired observations: eight energies at K50, seven at K100 and twelve geometries per energy. Plotting and the interval summary are additional short postprocessing steps.

From the repository root, the following command runs or resumes the preview and produces both plots and the grid-spacing audit. The four local workers share the laptop. The output files are `results/dec43_balanced_load/preview12/required_ebn0_vs_load.pdf`, its PNG, `summary.json` and `grid_precision.json`. Existing eight-frame files remain usable.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 uv run python jobs/043_polar_balanced_load/preview12.py --workers 4
```
