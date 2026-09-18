# Coordinate-wise amplitude frontier: B=14

Question: can each table reuse a small **scalar** bank without the loss caused by sharing whole amplitude vectors?
The forward rule is `g_t(w)=V[t,P_t w]`, then `alpha=g/||g||`, on the same selected hash support as jobs 029/030.
Only amplitudes and decoder weights may learn; discrete maps remain fixed. This is a full-message D0/D1
model-class test, not scalable decoding.

## Manifest

72 rows: `B=14,n=256,T=32`, D0/D1, seeds 2801/2802/2803. Per decoder/seed:

| Cases | Fixed bank | Joint bank + decoder | Purpose |
|---|---|---|---|
| Coordinate J=4,8,10; balanced initialization | Yes | Yes | Main compression frontier |
| Coordinate J=4; raw initialization | Yes | Yes | Isolate centering/RMS effect |
| Shared-label J=4; raw and balanced | Yes | No | Separate label-sharing from initialization |
| Unrestricted J=14; raw Gaussian | Yes | Yes | Matched parent |

Balancing centers each bank row and gives it unit RMS **once**. It does not enforce equal magnitudes or constrain
subsequent learning. Energy is normalized after gathering the message's entries, never by projecting bank columns.
At J=14, seeded physical columns exactly match the historical unrestricted Gaussian initialization. All job-031
joint rows use the same raw-parameter/gather-normalize optimization; historical job 030 additionally projected stored
columns, so its training trajectory is not an exact joint-training control.

All cases use 8 decoder layers, Adam at 0.001, at most 120 epochs of 100 batches of 8, 8 fixed validation batches,
patience 5, and best-checkpoint restoration including epoch zero. Power iteration starts deterministically.
Train K=7..22 and Eb/N0 uniformly over -4..12 dB; evaluate K=7,15,22,26 and -4,0,4,8,12 dB with 128 frames/cell.
Supports, maximum budgets, and data seeds are paired. Actual stopping epochs can differ. The receiver knows K,
noise variance and constant fading; messages are iid with replacement, as in the historical D0/D1 experiments.

Record pre/post column geometry, bank means and energy balance, effective support, loss curves, and per-cell PUPE.
The strict merger rejects missing rows/checkpoints, budget mismatches, incomplete grids and failed energy constraints.
It reports paired parent gaps and seed variability; three seeds do not establish a universal non-inferiority claim.
The main plot omits diagnostic ablations for clarity, but their results remain in the merged JSON.

## Run

From the repository root (the HPC environment must already be synchronized):

```bash
qsub jobs/031_coordinate_amplitude_frontier_B14/031_coordinate_amplitude_frontier_B14.sh
```

Merge after all rows return:

```bash
uv run python -m tests.framework_coordinate_amplitude_merge \
  --manifest jobs/031_coordinate_amplitude_frontier_B14/manifest.tsv \
  --results-root jobs/031_coordinate_amplitude_frontier_B14/results \
  --out-dir jobs/031_coordinate_amplitude_frontier_B14/results/merged
```

Local checks (outputs go to fresh temporary directories):

```bash
bash jobs/031_coordinate_amplitude_frontier_B14/local_smoke.sh
bash jobs/031_coordinate_amplitude_frontier_B14/local_mini.sh
```

`build_manifest.py` reproduces the checked-in order. Do not reorder submitted rows. `run_row.py --index N` runs one
production row; its explicit budget overrides are for smoke tests and will intentionally fail the production merger.

## Local evidence (18 September 2026)

Exact tests pass for operator/adjoint equality, gradients, nested projections, J=B endpoint equivalence, fixed versus
learned initialization, rank checks, checkpoint round trips, zero-vector energy handling, and B=100 sampled generation.
Twelve B=8 smoke paths and reduced-budget B=14 manifest execution pass. Four B=8 learning runs reduce held-out loss.
These are implementation checks; **no job-031 HPC performance result exists yet**. Full results should determine whether
coordinate J=4/8/10 matches its J=14 parent, whether balancing matters, and whether learning closes remaining gaps.
