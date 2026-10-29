# Research documentation

The working documents index the current polar decoder study:

| Document | Owns |
|---|---|
| [Current state](CURRENT_STATE.md) | Fixed Bayes problem, completed foundations, decoder catalogue and current implementation boundary. |
| [Tractability log](POLAR_TRACTABILITY_LOG.md) | Exploratory reasoning, structural combinations, controls and implementation decisions for Report 11. |
| [Experiment bank](EXPERIMENT_BANK.md) | Each script's question, corresponding algebra, setup, controls, results and reproduction command. |

The decoder study has four separate reports:

| Report | Owns |
|---|---|
| [7. Exact Bayesian decoding](reports/07_polar_decoding_from_posterior.pdf) ([source](reports/07_polar_decoding_from_posterior.tex)) | Exact posterior/decision, representations, identities, reductions, obstructions and certified decisions. The broad exact phase is complete for the fixed known-K contract. |
| [9. Bayesian approximation algebra](reports/09_polar_bayesian_approximations.pdf) ([source](reports/09_polar_bayesian_approximations.tex)) | Approximation derivations and completion audit for A–T: certified polar inner queries and a convergent residual hierarchy, field geometry/integration, variational references, signed contractions, robust guided branches and a low-order moment obstruction. Preserves the earlier contraction, sampling and decision-query paths with explicit combination errors and costs. Ready for implementation specification; no receiver implementation or empirical ranking. |
| [10. Complete decoder options](reports/10_polar_decoder_designs.pdf) ([source](reports/10_polar_decoder_designs.tex)) | One exact reference family and thirteen decoder families, with high-level explanations, complete discovery/inference/count decisions, variations and coverage of Reports 7/9. Adds product-law top-word discovery, exact auxiliary-field joint correction and finite prefix/tempered particle-batch guarantees. The report records the design-stage mathematics; subsequent implementations and measurements are recorded separately below. |
| [11. Tractable polar decoding](reports/11_tractable_polar_decoding.pdf) ([source](reports/11_tractable_polar_decoding.tex)) | Nonlearned computational reductions: parity proposals, selective envelope repair, reliable-basis search, residual reuse, bounded same-pattern pair discovery, mixed-pattern parity conditioning and sparse count decisions. Includes costs, conditional guarantees, counterexamples and established-method attribution; no validated scalable receiver is claimed. |

[Report 8](reports/08_polar_energy_geometry.pdf) defines and analyzes the frozen balanced encoder.
The Bayes count-tail objective stays fixed across all four decoder reports.
The [035 shared harness](../jobs/035_polar_decoders/README.md) implements REF, all thirteen DEC families
and D2/ODMA controls, with paired frames, PUPE/runtime plots and a complete 90-job CPU array capped
at four simultaneous two-CPU jobs. Each family has one declared finite variant; this does not claim every Report-10 variation is implemented.
The [036 combination study](../jobs/036_polar_combinations/README.md) adds DEC-14–24. These implementations retain full-alphabet dependence and serve as small-B accuracy references for the tractability investigation.
DEC-25–34 add procedural tractable variants; their algebra, exact checks and local smoke evidence are indexed in the experiment bank. The common small-B harness still constructs an explicit encoder; the procedural interface does not. [Job 037](../jobs/037_polar_tractable/README.md) compares these variants on the paired balanced Job-035/036 grid, reusing audited historical controls.
The reports retain their design-stage claims; new measurements belong to the experiment bank and job records.
Old approximation catalogues and empirical rankings are no longer active documents.

The [PUPE metric record](PUPE_METRIC.md) separately owns the 11 October correction, output-budget conventions
and historical-result migration. It is not another roadmap. Tests remain in tests/; the experiment bank is
the primary index for polar research, while [tests/README.md](../tests/README.md) covers the wider repository.

## Earlier technical history

| Report | Scope |
|---|---|
| [1. Initial ODMA decoding](reports/01_initial_odma_decoding.pdf) | Signal model, BP/EP and the transition to global recovery. |
| [2. Support-recovery bottleneck](reports/02_support_recovery_bottleneck.pdf) | Global posterior, classical decoders and dense/ODMA oracle-support evidence. |
| [3. Factorised encoder framework](reports/03_factorised_encoder_framework.pdf) | Explicit M=2^B algebra, implicit operators, gradients and jobs 018–022. |
| [4. Sectioned framework](reports/04_scalable_sectioned_framework.pdf) | Outer code, energy, local inference, association and jobs 023–026 failures. |
| [5. Generated sparse codebooks](reports/05_hash_skeleton_generator.pdf) | Hash supports, coordinate amplitudes and the original 031 design. |
| [6. Decoder ladder](reports/06_decoder_ladder.pdf) | D0–D4, uncertainty approximations and the candidate-search boundary. |

Reports 1–6 retain their historical assumptions and proposed experiments. Report 7 replaced the old D2-first
report on 11 October; Report 8 followed Job 034. The September restoration recovered derivations from retained
code/ledgers/passages, not byte-for-byte original documents. Each numbered report has adjacent LaTeX source
in `reports/` and uses the shared `report_style.tex`.

Detailed evidence remains in [Results 03](../results/03_results.md) (jobs 001–017),
[Results 04](../results/04_results.md) (framework/sectioned/hash studies), individual job READMEs/manifests and
the local [HPC notes](../jobs/README.md). The [conditioned-inference brief](archive/conditioned_inference_brief.md)
is a historical proposal. Older research is retained for reproduction, without competing current plans.

The 15–16 October cleanup consolidated overlapping companions, separated exact and approximate reports,
and reset the approximation study to algebra. Historical scripts, codebooks and existing outputs are unchanged;
pre-reset approximation drivers have a short parked index in the experiment bank.
