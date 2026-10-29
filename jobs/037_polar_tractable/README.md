# Job 037 — procedural polar decoder comparison

DEC-25–34 on the same balanced encoder and paired observations as Jobs 035/036. All new receivers are nonlearned, IID-source, known-K/noise variants. The common benchmark still builds the explicit small-B matrix for comparison; the receivers' separate procedural interface does not require it. B12/B14 retain **six payload bits**, so this campaign does not establish large-payload scaling or B100 performance.

## Queue

From the repository root, after transferring this code to the HPC and synchronizing its environment:

```bash
module load miniforge/3
uv sync --frozen
qsub -J '1-60%4' jobs/037_polar_tractable/037_polar_tractable.sh
```

The array is **10 receivers × 2 message lengths × 3 seeds = 60 jobs**. Each job covers all four loads and eight SNRs, with 256 frames/cell: **8,192 evaluations/job; 491,520 total**. Each requests two CPUs, 16 GB and 72 hours; the cap permits at most **8 CPUs and 64 GB** simultaneously. No GPU is requested. No training is performed. D2, adapted polar ODMA and earlier explicit receivers are reused, not queued again.

The fixed experiment contract is n256, polar length32, CRC8, balanced mapping, prefix bits B-6, K7/15/22/26, Eb/N0 -4:-2:10 dB, seeds3201/3202/3203. Both original coding-bound overlays remain enabled. The Polyanskiy achievable-error curve is not a floor for this fixed code; the genie/list-Fano lower bound can be weak. REF remains a tiny-case fixed-encoder reference.

| ID | Finite variant | Main budget |
|---|---|---|
| DEC-25 | SCL candidate multiset beam | width32, 8 screened patterns, SCL16 |
| DEC-26 | Reliable-basis candidate multiset beam | width32, 8 patterns, radius1, cap32 |
| DEC-27 | Independent-parity single-slot MH | basis initializer32; burn64 + samples64 |
| DEC-28 | Repaired-parity single-slot MH | same initializer/chain; radius1, cap32 |
| DEC-29 | Frozen-cache repaired-parity MH | same; four centers |
| DEC-30 | Four-symbol same-pattern pair / SCL single moves | same; list16, prior mixture0.1 |
| DEC-31 | XOR-first same-pattern pair / SCL single moves | same; list16, prior mixture0.1 |
| DEC-32 | SCL-list single-slot MH | same; list16, prior mixture0.1 |
| DEC-33 | Exchangeable parity-product / beam importance sampling | beam32, 1,024 independent tuples, mixture0.5 |
| DEC-34 | Union SCL/basis candidate multiset beam | width32, SCL16, radius1, cap32 |

The chain initializer is the top configuration in the basis beam; DEC-26 outputs its count-tail Bayes action, which need not coincide. Initial PUPE is recorded after inference, never supplied to a decoder. At K26, 128 single-slot updates are only about five nominal sweeps; pair variants alternate update sizes. Exact-invariant kernels do not establish mixing. Accepted self-proposals and actual changes are counted separately. A short chain retaining a good initializer is not evidence of good posterior exploration.

## Results and reuse

Raw frames, sparse tails, inputs, timings, work counters and source/config hashes are under `results_tractable/`; logs are under `logs_tractable/`. Completed runs are skipped only when config and runtime hashes match. Partial/incompatible outputs are not overwritten; preserve them and select a fresh `--out-root` for a deliberate retry. Keep runtime code stable while jobs run.

After all chosen new runs and controls are complete:

```bash
uv run python jobs/037_polar_tractable/merge.py --out jobs/037_polar_tractable/results_tractable/comparison
```

Default controls are D2, ODMA, DEC-05, DEC-07, DEC-14, DEC-16, DEC-17 and DEC-24. These cover the learned reference, the small-B paper adaptation, independent/particle inference, explicit multiset search and stronger combinations. Incomplete controls cause an explicit error. Select a complete cohort for an early comparison, for example:

```bash
uv run python jobs/037_polar_tractable/merge.py --B 12 --seeds 3201 \
  --controls D2 ODMA DEC-14 DEC-24 \
  --out jobs/037_polar_tractable/results_tractable/early_B12_s3201
```

Use `--decoders DEC-26 DEC-33 DEC-34` to compare a chosen subset, or an empty `--decoders` to audit/plot only historical controls. `--retry-results PATH` explicitly selects matching Job035 retries; `--baseline-results` and `--combination-results` override original result roots. Job036 load shards must jointly cover all four loads. Every selected decoder must cover the same requested seeds and cells. Use a new output directory for each merge.

`reference_contract.json` pins the reviewed current runtime, three original Git source snapshots and all selected research parameters (including DEC-07's -1000 guide floor). Existing encoder, metrics, fixture, training and decoder dependencies are unchanged; new setup/adapter/diagnostic branches were reviewed separately. Merge checks complete configurations, sources, parameter values, codebook hashes and per-frame observation hashes. Original records are not rewritten. Source changes require another explicit audit, not bypassing the guard. Use this merger for the current campaign; the older Job036 merge CLI was written for its own contemporary run snapshot.

Outputs include PUPE/runtime versus K, paired differences against D2/ODMA, uncertainty intervals and focused discovery, independent-sampling, single-correction and pair-correction panels. `summary.json` and raw metadata retain initial PUPE, ESS, expected-regret bounds and work counters. Candidate recall refers to all physically scored words, not posterior coverage. Comparisons reuse exploratory seeds and are not held-out validation.

## Verification and cost probe

```bash
uv run python -m unittest tests.polar_tractable_test tests.polar_tractable_execution_test tests.polar_tractable_harness_test tests.polar_pair_proposals_test tests.polar_tractable_importance_test tests.polar_tractable_jobs_test
uv run python -m tests.polar_tractable_budget_probe --B 14 --out results/job037_probe_B14.json
uv run python jobs/037_polar_tractable/build_manifest.py
```

One local K26/4dB frame at the declared budgets took 0.58–3.59 s/receiver for B12 and 0.54–5.40 s for B14. Multiplying these individual times by 8,192 gives approximately **1.2–12.3 job-hours per row** and 271 job-hours overall (542 CPU-hours at two CPUs/job). With four active jobs this is roughly 68 hours, before queueing. These are crude extrapolations from one frame/SNR/seed on a laptop, not HPC forecasts; candidate outcomes and hardware affect cost. The full sweep belongs on CPU HPC; small checks remain practical locally.

The 29 October paired B4 smoke, tiny exact checks and B12 matched-initializer diagnostic are indexed in `docs/EXPERIMENT_BANK.md` and `docs/POLAR_TRACTABILITY_LOG.md`. Their outputs are ignored, and they do not establish a production ranking.
