# Research Narrative and Evidence Map

The documentation is arranged as a causal research story rather than a catalogue of every attempted decoder. Read the
four reports in order; use the Markdown ledgers only when the exact experimental audit trail is needed.

## Supervisor-facing reports

1. [`reports/01_initial_odma_decoding.pdf`](reports/01_initial_odma_decoding.pdf) — the original ODMA factor graph,
   BP/EP algebra, decoder variants, early behaviour, and why the project moved to global support recovery.
2. [`reports/02_support_recovery_bottleneck.pdf`](reports/02_support_recovery_bottleneck.pdf) — the common global count
   model, the decoder search, explicit dense-versus-ODMA experiments, and the oracle-support decomposition.
3. [`reports/03_factorised_encoder_framework.pdf`](reports/03_factorised_encoder_framework.pdf) — the explicit
   factorisation, implicit global execution, D0/D1 gradients, exact representation checks, and jobs `021--022`.
4. [`reports/04_scalable_sectioned_framework.pdf`](reports/04_scalable_sectioned_framework.pdf) — procedural outer
   encoding, local banks, exact energy, association/BP algebra, jobs `023--027`, and the sparse-global density frontier.

The `.tex` source for each report sits beside its PDF. `report_style.tex` is the shared formatting preamble.

## Current handoff

[`CURRENT_STATE.md`](CURRENT_STATE.md) is a concise, neutral context document suitable for starting a new conversation.
It distinguishes verified observations, interpretations, limitations, and open research choices. It is the authoritative
summary after the returned job-`027` audit, including the four rows that still require repair.

## Detailed evidence

- [`../results/03_results.md`](../results/03_results.md) records jobs `001--017`, including explicit dense/ODMA sweeps
  and oracle-support controls.
- [`../results/04_results.md`](../results/04_results.md) records jobs `018--027`, implementation checks, returned-job
  audits, full numerical tables, and interpretation limits.
- [`EXPERIMENT_BANK.md`](EXPERIMENT_BANK.md) records current experiment contracts and latest job status.
- [`../jobs/README.md`](../jobs/README.md) records private HPC operation, submission, and merge commands.

## Evidence language

The reports use four levels of claim:

- **Exact/regression equivalence:** equality checked algebraically or numerically under a stated tolerance.
- **Measured result:** returned artifacts were complete enough to audit and the reported metric was observed.
- **Interpretation:** a causal explanation supported by controls but not proved for all decoders or codebooks.
- **Open hypothesis:** a direction requiring a new experiment or theoretical result.

Generated plots and raw checkpoints are intentionally not part of the narrative layer. They remain in `jobs/` and
`results/` so conclusions can be re-audited without forcing every intermediate artifact into the readable document set.
