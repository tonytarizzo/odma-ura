# 034 — Polar symbol energy and shared mixing

Two controlled encoder extensions of the **same ODMA–polar codebooks used in 033**. Keep full-alphabet D2 unchanged;
do not introduce a new receiver or candidate search. The separate [algebra report](../../docs/reports/07_polar_d2_inference.pdf)
derives D2, polar inference and the remaining receiver questions.

**Retired arm (10 October):** active code now supports only identity and learned energy. Mixer rows 7–12 remain in
the frozen manifest and merger so completed evidence is not lost; re-executing them requires revision `9d51cd8`.
The PBS default is now **1–6**, and the runner rejects retired mixer rows explicitly. Original energy checkpoints
still load strictly (the obsolete index buffer is discarded). No new jobs are requested by this cleanup.
The new [energy derivation](../../docs/reports/08_polar_energy_geometry.pdf) explains balancing and its limitations.

## Returned evidence — 9 October 2026

**12/12 complete**, strict audit passed, all 36 matched controls present. All new checkpoints and six fixed-polar
controls reproduce validation PUPE exactly and loss within `2e-8`. Energy and orthogonality constraints hold.

| Full-grid mean PUPE, distinct messages | B12 | B14 |
|---|---:|---:|
| Fixed dense / D2 | 0.17290 | 0.28526 |
| Fixed polar / D2 | 0.18346 | 0.29620 |
| Learned symbol energy / D2 | **0.17116** | **0.28776** |
| Shared mixer / D2 | 0.18735 | 0.30270 |

Energy learning improves every seed at both B values; at B14,K26,10 dB, PUPE falls from 0.13712 to 0.10617 (dense:
0.09831). The mixer is not supported by these runs; its class contains identity, so this is not proof of an intrinsic
model-class loss. Eleven runs stop early; only one energy run reaches the 200-epoch cap.

Checkpoint geometry shows strong channel-row energy balancing. Frozen receiver swaps preserve the encoder gain;
a deterministic support-only row-balancing diagnostic recovers much of it without encoder training. Those smaller,
fresh-sample diagnostics are separate from production rankings. See the local [full review](results/analysis/review_2026-10-09/REPORT.md)
for figures, seed differences, replay checks and implications. That review's checkpoint-analysis script uses the
original source fingerprints and mixer implementation: reproduce it at `9d51cd8`. Its saved diagnostics remain valid.
The historical merger below remains usable on the current code without loading mixer checkpoints.

## What changes

Split the message into pattern `p` and payload `v`. Fixed CRC/polar coding gives `b(v)` with `Nc` entries in `{−1,+1}`;
`S_p` inserts these at the original pattern's channel positions. The original waveform is `S_p b(v)/sqrt(Nc)`.

| Rows | Waveform | Learned encoder parameters | Question |
|---|---|---|---|
| 1–6 | `S_p diag(sqrt(softmax(theta_p))) b(v)` | One symbol-energy profile per pattern | Does reallocating energy among polar symbols help? |
| 7–12 | `S_p U b(v)/sqrt(Nc)`, `U=(I-K)^(-1)(I+K)`, `K=-Kᵀ` | One shared orthogonal pre-placement mixer | Can cross-pattern geometry improve without changing within-pattern distances? |

Zero logits and zero skew matrix reproduce the **original physical codebook exactly**, not just statistically.
Every message has unit energy by construction, also after training and for unseen payloads. No energy penalty or
identity-reconstruction loss is needed. The mixer is not a global rotation after placement. Its Cayley chart excludes
orthogonal matrices with eigenvalue −1; it is not an unrestricted dense map. Individual mixed symbols may be zero.

Both rows train encoder parameters and D2 jointly; polar/CRC bits, placements and discrete choices stay fixed.
There is no combined energy-plus-mixer arm. The fixed control for both is the completed 033 ODMA–polar/D2 run, so
duplicating it would waste compute. This tests additional waveform freedom, not polar without ODMA or a new decoder.

## Matched contract and controls

- `B=12/14,n=256`, seeds 3201/3202/3203: **12 new runs**. Reuse the original 032 pilot `selected.json`.
- Selected `Nc=32`, prefix lengths 6/8: 64/256 patterns, energy parameters 2,048/8,192; shared mixer 496 in either case.
- Eight D2 layers (16 scalars), 200 epochs maximum, patience 10, best encoder/decoder pair restored.
- 100 batches/epoch of 8, 16 validation batches; decoder/encoder learning rates 0.002/0.001.
- Train distinct messages with `K=7,15,22,26`, physical `Eb/N0` uniformly in `[-4,10]` dB. Evaluate distinct and IID
  separately at `-4:2:10` dB, 256 frames/cell/seed. Known K, noise and unit real channel, exactly as 033.

The merger reuses 36 completed controls: six 033 ODMA–polar/D2, 24 fixed/joint dense/sparse D2, and six native ODMA
from 032. It checks unchanged historical source fingerprints, selected settings, initial matrix hashes, parameter
counts and energy. Plot paired per-load/SNR curves separately by B and sampling mode; do not pool IID and distinct.
At original execution the two new source files were the only source-inventory additions; the merger verifies those
recorded fingerprints, not a claim that the current working source is unchanged.

The polar adapter computes D2's exact frame geometry from the binary generator, including repeated generator rows.
Its outputs and encoder gradients are checked against explicit Gram calculations. This saves redundant matrix work;
it does **not** change D2 inference or remove its `2^B` state. B100 codeword generation is tested, not B100 decoding.

## Submit and analyse

From the repository root, with the original pilot selection and environment available:

```bash
git pull --ff-only
uv sync
qsub jobs/034_polar_mapping/034_polar_mapping.sh
```

The current array covers energy rows 1–6 only (the original completed array was 1–12). Do not also submit overlapping split arrays.
Avoid updating a checkout while fingerprinted
jobs are running; use a separate checkout if necessary. After pulling the results:

```bash
uv run python jobs/034_polar_mapping/merge.py
```

Use `--allow-incomplete` only for a labelled interim analysis. Existing output directories must be empty for a new run;
there is no silent overwrite or training-resume fallback. The merger's `results/analysis/` contains completion audit,
aggregated metrics, learning curves and matched comparison plots. Historical bounds remain unchanged in 032/033.

## Local verification

```bash
uv run python -m tests.polar_mapping_test -v
uv run python -m tests.polar_energy_geometry_test -v
uv run python -m tests.polar_mapping_learning --out-dir jobs/034_polar_mapping/results_local_smoke
uv run python -m tests.polar_mapping_learning --mini --out-dir jobs/034_polar_mapping/results_local_mini
uv run python -m tests.polar_mapping_learning --production-shape --out-dir jobs/034_polar_mapping/results_local_B14
```

Original verification: all three smoke paths (fixed/energy/mixer), 40-epoch B8 mini-runs and reduced-budget B14/n256 execution.
Mini validation loss starts near 0.10815 and reaches 0.08205 fixed, 0.07585 energy and 0.07414 mixer. Best-pair restoration
and unit energy pass. These show initial learning, not a production ranking: the mini mixer has lower loss but higher
validation PUPE than the other two. Initialization matches all six returned 033 codebook fingerprints exactly.

The current smoke/mini runner exercises fixed and learned energy only. The geometry tests certify implicit moments,
the analytic energy gradient, balancing constraints/KKT conditions, known-K covariance and the failure of tightness
alone to guarantee recoverability. `uv run python -m benchmarks.polar_energy_geometry` writes exact returned-checkpoint
diagnostics to `results/analysis/energy_derivation.json`; it does not train or report new PUPE.
