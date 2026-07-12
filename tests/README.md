# Test And Experiment Scripts

This folder contains runnable experiment drivers rather than conventional unit tests. Most scripts are intended to be run from the repository root with `uv run python -m tests.<script_name> ...` and write plots or JSON summaries under `results/`.

## `__init__.py`

Marks `tests/` as a Python package so the scripts can be run with `python -m tests.<name>` and can import shared helpers from one another.

## `single_test.py`

Runs one ODMA+URA scenario with one or more decoders from `src.decoders.registry`. It prints a decoder comparison table, writes per-decoder count/metric plots under `results/single/`, and appends the metrics to the JSONL cache.

## `sweep_test.py`

Runs cached parameter sweeps over settings defined in `src.config.SWEEP_CONFIGS`, such as active-user load, SNR, antenna count, or codeword count. It can run new decoder trials or regenerate plots from the cache with `--plot-only`.

## `threshold_test.py`

Estimates required `Eb/N0` for URA recovery at target PUPE values. It supports either a fixed `Eb/N0` grid or adaptive bisection, runs selected decoders directly without using the cache, and writes threshold summaries/plots under `results/threshold/`.

## `arrangement_threshold_test.py`

Compares one ODMA resource arrangement against a dense global-codebook arrangement using the same payload, frame length, active-user loads, seeds, decoder, and threshold targets. This is an older paired dense-vs-ODMA threshold wrapper around `threshold_test.py`.

## `arrangement_sweep_threshold_test.py`

Runs the main multi-arrangement dense-vs-ODMA threshold sweeps used for HPC jobs. It accepts labelled arrangement specs such as `dense:2048:1` or `d512_b8:512:8`, checkpoints after every `(arrangement, K)` point, supports resume, optional bootstrap confidence intervals, and optional Polyanskiy-bound overlays.

## `plot_arrangement_sweep.py`

Provides plotting and bootstrap-confidence-interval helpers for `arrangement_sweep_threshold_test.py`. It is a helper module, not a standalone CLI script.

## `framework_odma_test.py`

Runs a framework-side inference sanity check for dense or ODMA presets. It verifies the known algebraic construction, runs oracle-K OMP over a grid of `K` and `Eb/N0` values, writes summary plots/JSON, and produces encoder morphology analysis for the representative framework encoder.

## `framework_equivalence_curve.py`

Checks exact equivalence between the legacy `src.scenario` dictionary and the new framework `(R, C, U, T)`
construction. For dense and/or ODMA presets, it asserts that both dictionaries match exactly and that a shared oracle-K
NNOMP decoder gives identical counts on the same observation. Default outputs go under
`results/framework_equivalence_odma/` and include Polyanskiy canonical/count/strict bound references plus empirical
required-Eb/N0 curves overlaid with those bounds.

## `framework_ccs_test.py`

Runs a direct CCS/tree-coded compressed sensing construction against the same global codebook represented through the
framework. It builds a shared section-level Gaussian sensing matrix, preceding-fragment random linear parity checks,
section-wise NNLS candidate lists, and root-wise tree stitching, then asserts exact direct/framework `Phi` equality and identical decoded counts before
writing Eb/N0/K comparison plots.
Default outputs go under `results/framework_equivalence_ccs/`; the dense baseline is included by default when using the
same `(B, n, K, Eb/N0, seeds)` grid. The required-Eb/N0 output is grid-threshold based, so use a fine enough
`--ebn0-grid` for meaningful curves.

## `ccs_bound_curve.py`

Runs an implicit paper-scale NNLS/tree CCS experiment without forming the `2^B` global dictionary. Its defaults match the
original paper's `B=75`, `N=22517`, `L=11`, `J=14` dimensions and physical real-AWGN Eb/N0 convention. Full NNLS and
the original `K_a+10` list rule are available. The current sensing matrix is Gaussian rather than BCH-derived, and the
default parity profile is uniform rather than the paper's load-dependent optimised profile; generated validation reports
surface both mismatches. A positive `--nnls-pool` enables a faster correlation-preselected approximation, but that mode is
not a paper-faithful NNLS reproduction.

## `ccs_amp_author_curve.py`

Runs the original factor-graph and AMP classes from a pinned checkout of the authors' `CCS-AMP-Code` repository. The
checkout is not vendored because upstream declares no software licence. The `paper_b128` preset matches the published
one-pass core dimensions (`B=128`, `n=38400`, 16-bit sections); the paper curve additionally used a two-pass SIC extension
whose empirical delta schedule is not present in the public code. The separately labelled `adapted_b100` preset uses a
Triadic10 graph with ten-bit sections and must not be compared directly with the paper's `B=128` points.

## `framework_ccs_amp_test.py`

Builds the authors' reduced `Triadic4(2)` CCS-AMP construction at `B=8`, extracts its exact seeded subsampled-Hadamard
inner operator, and represents the resulting global message dictionary with explicit framework components. It requires
machine-precision dictionary agreement and identical author-AMP estimates and decoded lists on shared observations.

## `ccs_amp_merge.py`

Merges per-load CCS-AMP author-code summaries produced in parallel by the HPC job into one required-Eb/N0 and PUPE plot.

## `framework_codebook_morphology.py`

Builds small explicit global codebooks for several framework-style families, including dense, slotted, ODMA, spreading, CCS, SPARC, and coded-pattern constructions. It does not run a decoder; it analyses codebook geometry through coherence, support overlap, row load, active-set conditioning, and related morphology plots.

## `framework_geometry_optimisation.py`

Runs decoder-free optimisation of framework codebook factors against global recovery-geometry objectives. It keeps the message plumbing fixed, learns `C` by default, optionally learns values on a fixed `R` mask, and writes before/after encoder analysis plus optimisation curves for AMP-style active-set Gram loss, VAMP-style spectral loss, or support-margin loss.

## `framework_geometry_decoder_eval.py`

Loads the `encoder_before.pt` and `encoder_after.pt` checkpoints from a geometry-optimisation run and evaluates them with the same sampled active messages and noise realisations. It reports paired before/after decoder metrics over an `Eb/N0` grid so geometry improvements can be checked against actual sparse-recovery performance.

## `framework_unrolled_decoder_test.py`

Trains the unrolled nonnegative ISTA decoder on a frozen dense or ODMA framework encoder. It compares the learned decoder against matched filtering and oracle-K NNOMP, writes a training summary and progress plot, and is mainly used to test whether the learned decoder is a credible training surrogate.
