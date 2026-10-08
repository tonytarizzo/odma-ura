# Research Documentation

Start with [CURRENT_STATE.md](CURRENT_STATE.md) for the research question, evidence and open decisions.
Use [EXPERIMENT_BANK.md](EXPERIMENT_BANK.md) for dated job status and links to exact contracts. The handoff is the live
research summary; the ledger is the experiment index, not a second roadmap.

## Meeting summary

[7 October 2026: evidence and next steps](reports/2026-10-07_supervisor_meeting/report.pdf) is the concise supervisor
report covering matched published-codebook comparisons, both bounds, generated encoders and the decoder ladder.
Its [folder](reports/2026-10-07_supervisor_meeting/README.md) contains LaTeX, curated figures and a reproducible snapshot.
It freezes the 6 October evidence; subsequent research decisions belong in the current handoff.

## Technical history

Read these for the derivation or the reason for a past decision. They are chronological records, not competing
current plans. Dates, local checks and proposed experiments describe their stage; Reports 5/6 predate the completed
031–033 analysis. Report 7 develops the current polar/D2 question. Consult the ledger for subsequent outcomes.

| Report | Scope |
|---|---|
| [1. Initial ODMA decoding](reports/01_initial_odma_decoding.pdf) | Signal model, BP/EP equations, early variants and transition to global recovery. |
| [2. Support-recovery bottleneck](reports/02_support_recovery_bottleneck.pdf) | Global count posterior, classical decoder algebra, dense/ODMA comparisons and oracle-support controls. |
| [3. Factorised encoder framework](reports/03_factorised_encoder_framework.pdf) | Explicit M=2^B model, implicit operators, gradients and jobs 018–022. |
| [4. Sectioned framework](reports/04_scalable_sectioned_framework.pdf) | Outer code, exact energy, local inference, association and the failures in jobs 023–026. |
| [5. Generated sparse codebooks](reports/05_hash_skeleton_generator.pdf) | Hash-support algebra, coordinate amplitudes, initialization and the original 031 test design. |
| [6. Decoder ladder](reports/06_decoder_ladder.pdf) | D0–D4 algebra, approximations, complexity and the candidate-search boundary. |
| [7. Polar structure and D2 inference](reports/07_polar_d2_inference.pdf) | D2 derivation, exact polar matrix identities, soft decoding, related multiuser work and the two Job 034 encoder tests. |

Each report has adjacent .tex source; report_style.tex is the shared preamble. The September revision restored
derivations from retained implementations, ledgers and recovered passages, not byte-for-byte original documents.
Keep historical failures and oracle assumptions visible rather than rewriting history around a new direction.

## Detailed evidence and operation

- [Results 03](../results/03_results.md): jobs 001–017, dense/ODMA sweeps and oracle-support evidence.
- [Results 04](../results/04_results.md): jobs 018–031, derivations, local checks and historical artifact audits.
  Later 031–033 outcomes are indexed in the [experiment ledger](EXPERIMENT_BANK.md).
- Individual job READMEs: exact manifests, training/evaluation contracts, submission and merge commands.
- [HPC notes](../jobs/README.md): private cluster workflow; intentionally not versioned.
- [Tests](../tests/README.md): implementation checks and executable analysis tools.
- [Archived conditioned-inference brief](archive/conditioned_inference_brief.md): theory proposal, not an active paper plan.

Distinguish exact algebra/regression equivalence, measured results, interpretations and untested hypotheses.
Compact generation, isolated-message injectivity and scalable multiuser decoding are separate claims. Raw outputs
stay in jobs/ and results/; curated report figures are the explicitly retained presentation exception.
