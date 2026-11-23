# Job 039 — DEC-35 performance, computation and feedback

The paper reference is the registered strict count-growth DEC-35 on the original polar/CRC code. Job 039 uses constant occupied-chip amplitudes to compare receivers on single-power B100 packets. All results use unknown patterns, IID messages with replacement, known load/noise and the actual returned message lists and counts. The earlier balanced experiment is a separate encoder comparison. One additional study asks whether the signed change in a local channel input identifies worthwhile retries.

## What runs

| Study | Physical draws | Comparisons and readout |
|---|---|---|
| [Energy curves](curve_design.md) | 256 independent seeds for each of K100 and K50, paired over 12 and 14 energies respectively | Native optimized SCL128/SIC, progressive rollback, registered DEC-35. Required energy at 5% list and count PUPE, paired intervals, false copies, mean/median/95th/worst time and SCL work. |
| [One final query](terminal_design.md) | Every primary curve observation | DEC-35 with one residual-magnitude SCL4096 query when the fitted count is K−1. Charge the shared full prefix and all additional work. Report paired accuracy, runtime and energy differences. |
| [Detector convention](quadrature_design.md) | First 64 of those seeds per load, all energies | Native SIC with complex rather than real magnitude ranking on the same physical signal; compare the three thresholds on this same subset. These are reused frames. |
| [Matched computation and feedback](control_design.md) | 128 separate K100 frames at 0.5 dB | Conventional search at the registered receiver's actual continuation-call and measured-time budgets; fixed 210-call conventional, add-only and reversible searches; controlled replay of every registered continuation query with updated and preceding residual numerators. |
| [Retry mechanisms](retry_design.md) | Local development and separate validation panels | The likelihood score, signed codeword direction, and empty-list stability investigations have separate gates. Their local evidence is summarized in the research readback. |

The array has 416 curve shards of 16 cases and 32 control shards of four cases. The detector comparison adds 1,664 native runs inside the curve shards. The K50 and K100 packets carry 88 and 87 payload bits respectively after choosing a pattern. CRC16, length 512 and 30,000 real channel uses are common. The independent paper implementation and the published marker are distinct comparators.

## Submit from the HPC repository root

Synchronize once after pulling this release:

```bash
git pull --ff-only origin main
uv sync --frozen --python 3.12
```

Submit all measurements and queue their dependent analysis with this one line:

```bash
paper_id=$(qsub jobs/039_polar_paper/039_paper.sh) && qsub -W "depend=afterok:${paper_id}" jobs/039_polar_paper/039_merge.sh
```

The reducer uses the whole array's returned ID, including the square brackets. The site scheduler controls concurrency. Already saved cases from the same source release are checked and reused after a resubmission. Submission requests two CPUs, with dense math libraries limited to one thread for per-receiver timings. [Imperial PBS job dependencies](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-dependencies/)

## Results

Outputs go to `jobs/039_polar_paper/results/`. The reducer validates the declared seed/energy coverage, nested stages, raw count/list/false metrics, costs, causal trace and paired physical fingerprints before writing:

- `summary.json` with the target crossings, paired energy differences, censoring, control gates, detector sensitivity and SHA256 of every input record.
- `performance_cost.csv` with counts, pointwise PUPE intervals, timing distributions and SCL work at every main operating point.
- `required_energy.csv` and `energy_gain.csv` with bracketed and censored threshold estimates and paired dB differences. Empty interval limits denote unbounded tails.
- `terminal_curve.csv` and `terminal_energy.csv` with the optional final query on all paired curve observations. `terminal_frames.csv` lists paired per-frame results and additional costs.
- `list_pupe.pdf` with separate K50 and K100 curves and pointwise frame-bootstrap bands.
- `curve/`, `control/` and `logs/` with per-frame-endpoint records, environment and PBS logs.

The held-out causal readback conditions on DEC-35's actual search schedule. Its alternative readout is a mechanism diagnostic. The independently run budget controls are complete receivers. Bootstrap energy differences interpolate only between measured neighboring energies; ambiguous or unbracketed thresholds retain censoring. The submission does not repeat the previously queued balanced terminal Job 037.

## Local verification

```bash
uv run python -m unittest discover -s jobs/039_polar_paper -p 'test_*.py'
uv run python -m unittest tests.paper_controls_test tests.polar_full_pattern_test
uv run python jobs/039_polar_paper/run_study.py --inventory
```

The study scripts check their source manifest at execution. Each complete frame record is linked atomically before moving to the next case. Memory for aggregation is allocated separately from the array workers. For a targeted rerun, `--shard INDEX --out DIRECTORY` uses the same release and design; `--merge --out DIRECTORY` requires all declared records.

The balanced Job 037 confirmation is complete locally on the 128 prescribed seeds (7501–7628). Of 31 eligible frames, one wide query of the largest residual-magnitude pattern improved 12 final actions, one copy each. Whole-panel missed copies fell 198→186 and false copies 117→110, at 1124.004→1132.026 s including geometry. Folded selection improved one further frame and missed its two-frame gate. The independent audit rebuilt 31 eligible observations and two ineligible controls, checked both rankings and final count decisions, and matched eight cold whole-receiver replays. The queued cluster run uses the same seeds. To check a returned Job 037 directory:

```bash
uv run python jobs/039_polar_paper/audit_terminal.py --input jobs/037_polar_tractable/results_terminal/main --out jobs/037_polar_tractable/results_terminal/audit.json
```

The [retry readback](retry_results.md) records local mechanism and receiver evidence. The [empty-list certificate](retry_stability.md) accounts for cached calls separately. Source and output files for the local research are independent of the fresh performance curves. Historical mechanism replays require the archived Job037 traces named in the individual designs. The new complete receiver trial can regenerate its observations from the specified seeds.
