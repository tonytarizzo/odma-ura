# Job 036: non-learned decoder combinations

This round combines the useful Job-035 ingredients without selecting only two final architectures.
It adds **DEC-14 through DEC-24** to the existing shared decoder bank. All new receivers are IID-only,
use the explicit alphabet, know K and physical noise variance, and return full-message counts through
`run(scenario, **params) -> (counts, meta)`. There is no encoder or receiver training in this array.

The objective is better multiset PUPE than D2, especially through the transition region. This is a
controlled exploration of declared finite variants, not an exhaustive search over combinations,
a claim of equal runtime, or proof of a Bayes-optimal decoder. No new implementation has yet established
a research-scale advantage over D2.

## Decoder coverage

| ID | Combination | What it isolates |
|---|---|---|
| 14 | Multiset beam, width 32 | Removes permutation duplicates; retains the full multinomial prior on each retained count state. |
| 15 | Residual-guided + product importance sampling | Adds the optimized product proposal to 05-style discovery. |
| 16 | Residual-guided + multiset-beam importance sampling | Adds deterministic configuration discovery without the product proposal. |
| 17 | Residual-guided + product + multiset-beam importance sampling | Combines both proposal sources. |
| 18 | 17 + terminal single-message Gibbs | Tests posterior-preserving refinement of the weighted samples. |
| 19 | 17 + terminal guided two-message Metropolis | Tests joint replacement of interfering messages. |
| 20 | Product-guided SMC | Changes the proposal, retaining the Gaussian continuation potential and physical terminal correction. |
| 21 | Gaussian-guided SMC + single-message Gibbs | Refinement-only control against 07. |
| 22 | Product-guided SMC + guided two-message Metropolis | Combines both changes. |
| 23 | Product-guided SMC + single-message Gibbs | Completes the proposal × refinement comparison. |
| 24 | Gaussian-guided SMC + guided two-message Metropolis | Joint-refinement control against 07. |

The SMC grid is two proposal types × three refinement choices: none, single message, pair.
Historical DEC-07 supplies the Gaussian/no-refinement cell. The importance grid compares guide alone
(historical 05), guide+product, guide+beam and all three, followed by two refinement variants of the
full mixture. Historical 02, 04 and 06 remain useful standalone controls.

Default budgets:

- IS: 1,024 independent tuple draws, batches of 128 for memory, defensive whole-tuple prior probability .1.
  DEC-15 uses mixture weights (prior, guide, product, beam)=(.1,.6,.3,0);
  DEC-16=(.1,.6,0,.3); DEC-17/18/19=(.1,.5,.2,.2).
- Product fit: up to 100 L-BFGS iterations per observation; convergence is recorded, not assumed.
- Beam: 32 unique multisets per depth, scanning the full alphabet for each retained parent.
- SMC: four independent batches of 64 particles, resampling each prefix step. Product variants use
  step-proposal weights (prior, Gaussian guide, product)=(.1,.65,.25); the others use (.1,.9,0).
  Intermediate continuation floor is -1000; the terminal physical likelihood is never clipped.
- Refined variants: eight random-block updates per final weighted sample. Block size is one or two;
  two-message proposals use the residual outside the selected block and an explicit Hastings correction.
  This costs more than eight one-message updates and is not an equal-runtime comparison.

The mixtures are fixed before sampling, conditional on y. Their weights are starting choices, not
optimized mixture weights. There is no hidden oracle selection, likelihood-only hard-output selector,
truth-guided candidate list, or restriction to 32 words: the beam limits configurations, while the
importance/SMC proposals retain full alphabet support.

## Posterior and multiplicity accounting

For ordered tuples x, the IID prior is M^(-K). For counts c, it is
K! / (prod_w c_w!) * M^(-K). The new beam stores a sorted tuple for each unique multiset and recomputes
this full count-state prior, including permutations whose separate paths were never retained. Its
posterior is conditional on the retained count states; omitted mass remains uncertified. Retaining the
top `width` children of each parent is sufficient before merging because a parent's children are all
distinct count states. No approximate count averaging or loss of message identity is introduced.

To use the beam as an ordered-tuple proposal, sample a retained multiset according to its normalized
physical posterior weight, then uniformly permute its entries. The tuple density is the multiset
probability divided by its orbit size K!/prod c_w!. The complete IS density is the sum of all mixture
component densities, including overlaps. Importance weights are physical likelihood × ordered prior /
that complete density. We never simply average the separate receivers' final decisions.

Gibbs and guided block Metropolis preserve the full ordered physical posterior. Terminal moves
transport each weighted sample **without changing its existing importance/SMC weight**. Reweighting a
moved state using the old proposal density would be incorrect. For IS, unnormalized transported count
marks remain unbiased; normalized finite-sample decisions can still be wrong. For SMC, independent
batches' unnormalized evidence and marks are pooled before normalizing; individual particles are not
independent confidence units. These terminal moves do not change the recorded evidence estimate.

The final action allocates K copies from posterior count tails, not from MAP configurations or means.
All new finite error certificates are conservatively 1; ESS and uniqueness are diagnostics, not accuracy
proofs. Evidence includes the source prior but omits the common AWGN density normalizer. Restore it when
comparing different noise variances or other models that change that normalizer.

## Research array and reuse

The default research manifest keeps **exactly the Job-035 balanced-encoder setup and seeds**:
B12/B14, n256, Nc32, six non-prefix message bits, CRC8, seeds 3201/3202/3203,
K=7/15/22/26, Eb/N0=-4,-2,0,2,4,6,8,10 dB, 256 frames per cell, IID messages.

Only the eleven new variants are queued. D2, ODMA, DEC-02/04/05/06/07 are **reused from completed Job-035
runs**, not retrained or rerun. This preserves paired inputs and saves computation. It also means these
are previously explored seeds, **not held-out validation**; fresh-seed validation belongs after selection.
The uniform mapping is not repeated in this round. The prior uniform results remain available.

- 2 B × 3 seeds × 11 new variants = 66 full decoder sweeps.
- B14 DEC-15–19 are each split into K7/15 and K22/26 shards: 15 extra array rows.
- **81 array rows, 540,672 decoder-frame evaluations**; sharding does not add statistical work.
- Each row requests two CPUs, 16 GB, 72 hours. No GPU is required or used.
- At most four subjobs: eight CPUs / 64 GB reserved by this array. If Job035 also has four active
  subjobs, the two arrays together can reserve sixteen CPUs / 128 GB.

A one-frame laptop probe at K26/4 dB measured approximately 0.075–4.9 seconds/receiver at B12 and
0.31–20.5 seconds at B14. The slowest B14 variant is pair-refined mixture IS (19). A 4,096-frame shard
at that single measured latency would take about 23.3 hours; this is an extrapolation, not a worst-case
bound or HPC timing prediction. Splitting the heavy B14 runs provides margin for different nodes/SNRs.
The main remaining runtime risk is hardware/optimizer variability under the 72-hour cap.

### Queue safely while Job035 is still running

Do **not** pull into or modify the working checkout used by active Job035 processes: their final source
fingerprint checks would reject changed files. From the existing HPC repository, create a separate
checkout of the pushed main commit, then prepare its environment:

```sh
cd ~/odma-ura
git fetch origin
git worktree add --detach ../odma-ura-036 origin/main
cd ../odma-ura-036
module load miniforge/3
uv sync --frozen
qsub -J '1-81%4' jobs/036_polar_combinations/036_polar_combinations.sh
```

If `../odma-ura-036` already exists, inspect/reuse it rather than overwriting it. Submit from its repository
root. Do not change that checkout while its runs are active. The worker uses `uv run --no-sync`.
If using an already updated, separate checkout, the last `qsub` line is sufficient after environment setup.

The runner skips only completed children with matching config/source fingerprints. It does not overwrite
partial or incompatible outputs. Preserve/rename an interrupted child before deliberately rerunning it.
No jobs are submitted automatically by these scripts.

### Retrieve and merge

With the separate checkout above, retrieve its results into this local repository:

```sh
rsync -avz at5424@login.cx3.hpc.ic.ac.uk:/rds/general/user/at5424/home/odma-ura-036/jobs/036_polar_combinations/ jobs/036_polar_combinations/
```

After all selected runs are complete, locally:

```sh
uv run python jobs/036_polar_combinations/merge.py
```

When merging on HPC from the separate checkout, point at the original controls:

```sh
uv run python jobs/036_polar_combinations/merge.py \
  --baseline-results ../odma-ura/jobs/035_polar_decoders/results_all_decoders
```

The default control set is D2, ODMA, DEC-02/04/05/06/07. If some controls remain incomplete, explicitly
select a complete subset with `--controls D2 ODMA DEC-02 DEC-04`, or use an empty `--controls` for
new receivers only. The merger never quietly pools different seed sets or incomplete cells. It requires
all new manifest rows. For an early subset, supply a deliberately reduced manifest and label that scope.
Use `--out` for a new analysis directory when remerging; existing output is not overwritten.

Historical source compatibility is deliberately narrow: `reference_contract.json` pins the Job035 source
fingerprints and control parameters. Shared model/evaluation definitions and all unchanged dependencies
must match; only registration/new-module additions, input-validation, merge and rendering changes are
allowed by this audited transition. Full experiment/training configs, same-seed codebook hashes, paired
observation hashes, seed coverage and control budgets are checked. Original summaries are never rewritten.
Mixed-source aggregate summaries retain per-run source hashes/revisions, rather than pretending all
receivers ran from the same checkout. Ordinary harness merges still reject different sources by default.

Outputs retain frame counts/PUPE/diagnostics, summary JSON, cell tables, runtime and PUPE plots, the two
reference bounds, and original per-run provenance. Focused `groups/` panels separate discovery, IS
refinement and SMC refinement so the eighteen-curve aggregate is not the only view.

## Verification and local smoke commands

```sh
uv run python -m unittest tests.polar_decoder_bank_test tests.polar_decoder_families_test \
  tests.polar_decoder_combinations_test tests.polar_combination_jobs_test \
  tests.pupe_metrics_test tests.polar_mapping_test tests.published_bounds_test
uv run python -m tests.polar_decoder_benchmark --profile smoke \
  --decoders REF DEC-05 DEC-07 DEC-14 DEC-15 DEC-16 DEC-17 DEC-18 DEC-19 DEC-20 DEC-21 DEC-22 DEC-23 DEC-24 \
  --loads 1 3 4 --ebn0 -4 4 12 --frames 64 \
  --out-dir jobs/036_polar_combinations/results_local_smoke_final
uv run python jobs/036_polar_combinations/plot_groups.py jobs/036_polar_combinations/results_local_smoke_final
```

Use a fresh output path if reproducing locally. Completed verification: 77 regression tests; exact multiset
posterior exhaustion; mixture normalization and sampled-density agreement including repeated messages;
conditional block detailed balance; actual-kernel posterior stationarity; large-sample tail/evidence
agreement; historical-SMC reduction; legal reproducible output; preserved Job035 manifest; cross-source
merge opt-in/provenance; split-run pairing and safe retry. The tiny smoke has 8,064 frame-decoder evaluations
(6,336 new-variant evaluations), checks conditional regret against REF, and shows decreasing average PUPE
from -4 to 4 to 12 dB for every new variant, with zero observed high-SNR errors. This validates mechanics,
not research-scale superiority. Generated results and plots remain ignored by git.

Production-size endpoint smoke also passed: 44 evaluations (all eleven new variants, B14/K26,
two frames at -4 and 10 dB). Every variant improves across those endpoints; high-SNR PUPE is 0 or
1/52 on these two frames. This is not a performance ranking. Messages and standardized noise match
Job035; local/HPC matrix reconstruction differs by at most 5.6e-17. These local endpoint results are
not used as paired production controls: the production merger still requires exact matrix/observation
hashes. An actual two-row CLI pilot completed, skipped a repeated row and generated merged/group plots.
