# 035: shared polar decoder benchmark

[Driver](../../tests/polar_decoder_benchmark.py) · [harness](../../benchmarks/polar_decoders.py) ·
[callable bank](../../framework/polar_decoders.py) · [derivation/verification log](../../docs/POLAR_DECODER_DESIGN_LOG.md)

All thirteen Report-10 families now have an executable finite-budget variant. REF and the existing D2/ODMA
controls use the same harness. This implements the family catalogue, not every variation or combination in
Report 10. No job is submitted automatically. Generated results are ignored by Git.

## Implemented choices

| ID | Finite implementation and default budget | Source |
|---|---|---|
| REF | Exact count-state posterior, factorial prior and minimum-multiset-PUPE count-tail decision; tiny cases only | Both |
| DEC-01 | Excluded-word Gaussian residual queries with exact prior moments; finite-population correction for distinct messages | Both |
| DEC-02 | Identical-slot variational product; 100 L-BFGS iterations and binomial-tail decision | IID |
| DEC-03 | Binary message-prefix gain refinement using the product-law cell provider; 128 refinement splits | IID |
| DEC-04 | Ordered-configuration beam, width 32; exact retained weights and discarded-prefix evidence bounds | Both |
| DEC-05 | Independent sequential residual-guided importance; 1,024 tuples, whole-tuple defensive fraction .1 | Both |
| DEC-06 | Four local/global posterior chains; 128 burn-in plus 128 retained states per chain; exact single-slot conditionals, global refresh probability .1 | Both |
| DEC-07 | Word-prefix SMC; four independent batches of 64 particles; defensive fraction .1 per transition | Both |
| DEC-08 | Positive marked continuation recurrence; 32 candidates, projected rounding spacing .25, at most four representatives per occupancy | Both |
| DEC-09 | Positive regional contraction; singleton word regions, exact total-K constraint, deleted cross-region interactions | Both |
| DEC-10 | Positive connected-component contraction up to order two; 32 candidates; order three also implemented | IID |
| DEC-11 | Continuous Gaussian Fourier count queries; 512 shared frequencies and signed event/complement intervals | IID |
| DEC-12 | Signed SVD contraction of a split-tuple table; 32 candidates, width 32 per half, rank eight, measured error envelope | Both |
| DEC-13 | Corrected auxiliary-field/tuple importance; 512 independent pairs, physical field rank two, defensive fraction .1 | IID |
| D2 | Existing eight-layer effective-channel receiver and legacy frozen-encoder training protocol | Both evaluation sources |
| ODMA | Existing polar CRC-SCL/pattern/SIC receiver; unchanged on uniform energies, labelled amplitude adaptation on balanced energies | Both |

All implementations currently use the explicit alphabet M=2^B. Prefix refinement avoids some decision
queries but still constructs the M-way product law. DEC-08/10/12 restrict the candidate alphabet using only
the observation; discarded support uncertainty is reported separately, and candidate recall is measured
only after inference. DEC-12 constructs the retained cut table before its SVD; this is a verified contraction
prototype, not a claim of cheap tensor construction. DEC-09 is the edge-deletion regional variant, not a
complete implementation of every forest/cutset/feature variant. DEC-06 optionally supports joint two/three-slot
Metropolis moves. DEC-10 supports component orders one, two and three, not arbitrary orders.

Every receiver implements `run(scenario, **params) -> (counts, meta)`. Counts are a length-M nonnegative
integer vector with total at most K; the new designs allocate exactly K. Scenario contains the encoder,
observation, K, physical noise and source, never truth IDs. Count-tail metadata is optional; signed paths
identify their lower/upper intervals rather than pretending to have a normalized positive posterior.
New combinations can receive new registry IDs with declared parameters and source support.

The error certificates are intentionally conservative. Candidate truncation, unknown chain mixing and
insufficient signed-mode precision can produce a bound of one. DEC-11 can have no informative certified
lower tail at this budget; it still returns a legal, observation-based incumbent and records the vacuous
intervals. ESS and apparently good PUPE are not substitutes for a proof. DEC-03's search gap is relative to
its product provider, not to the exact physical posterior.

## Matched contract

- Unit-energy polar/CRC encoder; real AWGN variance `1/(2 B 10^(Eb/N0/10))`.
- Deterministic KL-balanced energies with fixed placements, or a separately labelled uniform-energy panel.
  Failed balancing is an error; no codebook repair or decoder-dependent selection.
- Identical observations for all receivers; multiset PUPE is `1 - sum_w min(true_count_w, decoded_count_w)/K`.
  Missing outputs count as errors. IID and distinct sources are never pooled.
- Frame keys depend on seed/source/K/frame. Decoder selection/order, load sharding and SNR ordering cannot
  alter truth or noise. SNRs reuse standardized noise. Stochastic receivers have independent fixed keys.
- D2 uses the existing objective, Adam at .002, eight layers, distinct-message training over K=7/15/22/26,
  and uniform Eb/N0 in [-4,10]. Research defaults: 200 epochs, 100 batches/epoch, batch eight,
  16 validation batches, patience ten, validation-based early stopping and best restoration.
  Training, validation and evaluation streams are separate; evaluation never tunes a parameter.
- New algebraic receivers have no learned parameters. Their declared computation budgets are independent
  of the common physical test. An optimizer iteration, importance sample and D2 layer are different costs.
- Float64 throughout, including D2. Historical production used float32: same recurrence/protocol, not
  bitwise replay. Existing support-oriented D2/ODMA decisions retain their IID multiplicity limitations.

Uniform ODMA is regression-tested against the unchanged repository baseline. Balanced ODMA uses known
symbol amplitudes in LLRs and cancellation, retaining pattern ranking, Gaussian interference, CRC-SCL and
SIC. Neither this small-B baseline adaptation nor a smoke-test win proves superiority to the paper at its
native operating point. Compare decoder effects within a fixed encoder. Historical IID membership PUPE
cannot be spliced into these curves; distinct controls remain comparable only when their other contracts match.

## Local commands and artifacts

From the repository root:

```bash
uv run python -m tests.polar_decoder_benchmark --list-decoders
uv run python -m tests.polar_decoder_benchmark --profile smoke --threads 2 \
  --out-dir jobs/035_polar_decoders/results_local_all_smoke_balanced
uv run python -m tests.polar_decoder_benchmark --profile smoke --threads 2 --mapping uniform \
  --out-dir jobs/035_polar_decoders/results_local_all_smoke_uniform
uv run python -m tests.polar_decoder_benchmark --profile smoke --threads 2 --mapping uniform \
  --sampling distinct --decoders REF DEC-01 DEC-04 DEC-05 DEC-06 DEC-07 DEC-08 DEC-09 DEC-12 D2 ODMA \
  --out-dir jobs/035_polar_decoders/results_local_all_smoke_distinct
```

The default smoke profile runs all 16 choices on B4/n12/Nc8/CRC4, placement seed3217, K=1/2/3/4,
Eb/N0=-4/4 dB, 128 frames/cell and a shortened 30-epoch D2 cap. Always use fresh output directories;
completed or partial runs are never silently overwritten. Select any subset with `--decoders`.

```bash
uv run python -m tests.polar_decoder_benchmark --profile smoke --decoders REF DEC-06 DEC-10 \
  --decoder-params '{"DEC-06":{"block_size":2},"DEC-10":{"order":3}}' --out-dir /tmp/polar-variant
uv run python -m tests.polar_decoder_benchmark --plot-only jobs/035_polar_decoders/results_local_all_smoke_balanced
```

Resolved config JSON can be passed with `--config`; CLI overrides it. `--d2-checkpoint` accepts this
harness's `D2/decoder.pt` after checking encoder and training settings. Reuse the checkpoint for identical
training-contract inference sweeps. Other checkpoint formats are not silently adapted.

Each run saves config/source hashes/Git revision, encoder.npz, and inputs/*.npz with truth, standardized
noise, observations and variance. frames.jsonl contains every decoded count multiset, input fingerprint,
PUPE, inference time and diagnostics. summary.json and cells.csv include uncertainty, paired differences
against REF/D2/ODMA, candidate coverage, certificate indicators and separate setup/geometry/training costs.
D2 retains a checkpoint and learning trace. A partial run has no complete summary and cannot be merged.

Plots show PUPE versus active users at each SNR, inference time versus users, D2 validation learning,
and exact-posterior conditional regret/tail error when REF is evaluated in the same run. The large-bank
posterior diagnostics are heatmaps; grey cells mean unavailable, not zero. For signed receivers the tail
comparison concerns the returned lower endpoints. These posterior diagnostics are evaluation-only.
Separate REF rows do not retroactively supply tail tables to other rows. One-seed uncertainty uses frame
t intervals, multi-seed panels use seed means; distribution-free Hoeffding intervals are also saved.
Zero observed errors do not establish zero PUPE. Inference timing includes readout, excludes separately
reported setup/training and shared I/O.

Two historical coding references remain visible: the Polyanskiy p_t+q_1 achievable-error upper bound and
the genie/list-Fano lower bound. Achievability is not a performance floor or a guarantee for this fixed
polar code; list-Fano is often weak/zero and its gap need not be removable by decoder improvements.
Tiny same-encoder REF is the specific Bayes target. Conditional regret measures approximation loss;
finite-frame empirical PUPE differences against REF can fluctuate around zero.

## B12/B14 compute and the complete CPU array

All 15 practical receivers completed B12/B14 timing probes locally at K7 and K26, n256, Eb/N0=4 dB,
using two CPU threads on the Apple M2/8 GB laptop. One observation per endpoint was used for the full
catalogue: these are cost probes, not performance estimates or worst-case walltime bounds.

| ID | B14 seconds/frame, K7 → K26 | K26 estimate per 2,048-frame load slice |
|---|---:|---:|
| DEC-01 | .212 → .705 | 24 min |
| DEC-02 | .094 → .111 | 4 min |
| DEC-03 | .214 → .383 | 13 min |
| DEC-04 | .107 → .405 | 14 min |
| DEC-05 | 2.77 → 10.44 | 5.9 h |
| DEC-06 | 1.26 → 1.72 | 59 min |
| DEC-07 | .984 → 4.05 | 2.3 h |
| DEC-08 | .101 → 1.38 | 47 min |
| DEC-09 | .396 → 1.10 | 38 min |
| DEC-10 | .012 → .016 | 34 s |
| DEC-11 | .936 → 3.46 | 2.0 h |
| DEC-12 | .005 → .011 | 22 s |
| DEC-13 | 2.07 → 7.11 | 4.0 h |
| D2 | .039 → .047 | Full four-load row; training extra |
| ODMA | .018 → .042 | Full four-load row |

Small candidate/list caps explain some very fast rows; compare their PUPE and coverage too. The local
B12 K26 DEC-05 probe was 3.03 s/frame, about 1.7 h per 2,048-frame load slice. Full D2 training is additional;
prior batch timings imply roughly 23–33 minutes at B12 and 58 minutes at B14 at the 20,000-batch cap,
plus validation. Early stopping can shorten this. Hardware, SNR, geometry and threading change timings.

CPU is sufficient. Current NumPy/SciPy/PyTorch inference has no GPU implementation; requesting a GPU
does not accelerate it. Laptop development and small comparisons work, but the full grid belongs on HPC.
REF is excluded: at K7 alone B12/B14 require approximately 3.9e21/6.3e25 IID count states.

The checked-in manifest now has **90 array jobs**: B12/14 × three seeds3201–3203 × fifteen receivers
(DEC-01–13, D2 and ODMA). Each job runs **balanced then uniform sequentially**, and each mapping runs
all four K values 7/15/22/26, eight SNRs -4,-2,0,2,4,6,8,10 and 256 frames/cell. This gives 180 separately
saved runs and the same **1,474,560 decoder-frame evaluations**. D2 still trains once per encoder/seed.
All training settings, physical inputs, decoder budgets and comparison coverage are preserved.
The research manifest sets DEC-07 `guide_log_floor=-1000` (smoke default -80), because n256 Gaussian
continuation potentials otherwise get clipped almost uniformly.

The former 648 rows came from 2 B × 2 mappings × 3 seeds × (13 decoders × 4 load shards + 2 baselines).
That was an unnecessarily large scheduling layout. Bundling reduces job count, **not total experiment
work**. Resource use is controlled independently: at most **four jobs run concurrently**, each requesting
**two CPUs, 16 GB and 72 h**, with no GPU. This array therefore requests at most **eight CPUs and 64 GB
at once**. Completed jobs release their resources; 72 h is a per-job time limit, not the expected runtime.
The two-thread local measurements are the relevant rough guide. Each default bundle contains 16,384
frames; the B14 DEC-05 endpoint rates imply about 13–48 hours per bundle before setup, with significant
hardware/SNR uncertainty. Shorter jobs will finish much sooner. The campaign takes longer elapsed time
with the concurrency cap. Halving frames or omitting a mapping would reduce total work, but is not done
silently here.

After transferring the code/environment, from the repository root:

```bash
uv sync
qsub -J '1-90%4' jobs/035_polar_decoders/035_polar_decoders.sh
# After every job finishes, merge on a CPU node:
uv run python jobs/035_polar_decoders/merge.py
```

The `%4` is PBS's simultaneous-subjob limit; it is also embedded in the script, so plain `qsub` retains
the cap. See the [PBS job-array documentation](https://help.altair.com/2024.1.0/PBS%20Professional/PBSUserGuide2024.1.pdf),
section 8.4.1.1. No job has been submitted by Codex. This replaces the old 648-row submission command.

Each manifest line is a named bundle with a `runs` list of ordinary resolved benchmark configurations.
The runner executes those runs sequentially. If a bundle is retried, it skips a completed child run only
when its config and source hashes match. Partial output is never overwritten: preserve/move that child
run's directory before retrying. Different source/config results are rejected rather than mixed.

Outputs remain in `results_all_decoders/`, logs in `logs_all_decoders/`. The merger reads every child
run and creates four B/mapping panels under `results_all_decoders/analysis/`, retaining seed-level
uncertainty and rejecting missing runs, incompatible configs/source hashes, duplicate records,
different budgets under one label, unequal seed sets or unpaired observations. Merging the complete
bank is also CPU/memory work; use an allocated CPU node if the collected frame records exceed local memory.

A balanced-only manifest still has 90 jobs, each with one mapping and **half the evaluation work**.
Both-mapping distinct controls have 60 jobs (unsupported IID-only families are omitted). Choose one
campaign or separate output roots; repeated run names cannot overwrite existing results.

```bash
uv run python jobs/035_polar_decoders/build_manifest.py --mappings balanced \
  --out jobs/035_polar_decoders/manifest_balanced.jsonl
qsub -J '1-90%4' -v POLAR_DECODER_MANIFEST=jobs/035_polar_decoders/manifest_balanced.jsonl \
  jobs/035_polar_decoders/035_polar_decoders.sh
```

Pass the matching `--manifest` to merge.py. Use `run_row.py --out-root` for different-budget campaigns
and `merge.py --results` for that root. `merge.py --out` chooses a fresh analysis directory.

## Verification

```bash
uv run python -m unittest tests.polar_decoder_families_test tests.polar_decoder_bank_test \
  tests.pupe_metrics_test tests.polar_mapping_test tests.published_bounds_test
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  uv run python -m tests.polar_decoder_cost --bits 12 14 --loads 7 26 --repeats 1 --threads 2 \
  --out /tmp/polar-all-cost.json
```

Checks cover independent exact limits, covariance formulas, proposal normalization/detailed balance,
SMC evidence/marks, quantization error, regional/connected exact endpoints, Fourier signed intervals,
SVD envelopes, auxiliary-field correction, source constraints, fixed seeds and job merging. See the design
log for completed smoke measurements and interpretation. All 62 tests and 44,032 saved smoke scores
passed their checks; all 12 research encoders build. Both load-sharded and bundled CLI/merge execution
were exercised, including safely retrying a completed bundle.
All thirteen new families also passed one-frame B14/K26 numerical checks at -4/10 dB on both mappings;
the research particle-floor setting was checked separately. These probes do not estimate PUPE curves.
A good trend alone is not implementation proof,
and a poor approximation at its declared budget is not automatically a coding bug.
