# Research Narrative and Evidence Map

The documentation is arranged as a causal research story rather than a catalogue of files. Read the five reports in
order: each retains the algebra, assumptions, failed routes, evidence, and decision that motivated the next stage. Use
the Markdown ledgers when the exact run-level audit trail is needed.

## Supervisor-facing reports

1. [`reports/01_initial_odma_decoding.pdf`](reports/01_initial_odma_decoding.pdf) — the original ODMA signal model,
   resource/block BP/EP equations, V1--V4 progression, instability mechanism, and transition to global recovery.
2. [`reports/02_support_recovery_bottleneck.pdf`](reports/02_support_recovery_bottleneck.pdf) — the common global count
   posterior; explicit BP/EP, MAP-ADMM, NNOMP, VAMP, BlockMAP/BlockCD/SIC algebra; dense-versus-ODMA experiments; and
   the oracle-support decomposition.
3. [`reports/03_factorised_encoder_framework.pdf`](reports/03_factorised_encoder_framework.pdf) — the explicit
   `M=2^B` starting point, two-layer factorisation, representation library, implicit execution, encoder/D0/D1 gradients,
   exact checks, and jobs `018--022`.
4. [`reports/04_scalable_sectioned_framework.pdf`](reports/04_scalable_sectioned_framework.pdf) — procedural outer
   encoding, local banks, exact deployment-time energy, modular parity algebra, local D0, association and BP/list
   algebra, and the failure analysis from jobs `023--026`.
5. [`reports/05_hash_skeleton_generator.pdf`](reports/05_hash_skeleton_generator.pdf) — why the explicit sparse result
   from completed job `027` motivates a generated, searchable `L=1` replacement; the table/hash algebra; and the fixed
   and joint-learning tests `028--029`, followed by coordinate-wise amplitude generation and job `031`.

The `.tex` source for each report sits beside its PDF. `report_style.tex` is the shared formatting preamble.

## Documentation provenance

The September 2026 revision restores derivational depth that was removed during an earlier cleanup. The first two
reports and the explicit factorised framework were reconstructed from retained source implementations, experiment
ledgers, prior report section maps, and recovered original passages. They preserve the verified equations and historical
decision sequence while using the newer common visual style. They are not presented as byte-for-byte copies of the
superseded documents.

Historical alternatives are retained when they clarify an assumption or a dead end. They are labelled as historical,
oracle-aided, approximate, or current so that a future paper can reuse the derivation without mistaking an abandoned
variant for the present method.

## Current handoff

[`CURRENT_STATE.md`](CURRENT_STATE.md) is a concise, neutral context document suitable for starting a new conversation.
It distinguishes verified observations, interpretations, limitations, and open research choices. It is the authoritative
summary through the completed jobs `028--030` and the locally checked coordinate-amplitude batch `031`.

## Detailed evidence

- [`../results/03_results.md`](../results/03_results.md) records jobs `001--017`, including explicit dense/ODMA sweeps
  and oracle-support controls.
- [`../results/04_results.md`](../results/04_results.md) records jobs `018--031`, implementation checks, returned-job
  audits, numerical tables, and interpretation limits. Jobs `028--030` have returned; job `031` HPC results are pending.
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
