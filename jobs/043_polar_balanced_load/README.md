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
