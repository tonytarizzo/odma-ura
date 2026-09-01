# Explicit ODMA Gap Experiments

Jobs `001`--`008` use `NNOMP-OracleK`, target `PUPE <= 0.05`, bisection bracket `[-4, 4]` dB, tolerance `0.1` dB, and `50` seeds. Sparse arrangements are compared against the dense global-codebook baseline under the same `(B, n, K_a)`.

# 001_B10_n1024 - PBS 2836925.pbs-7
Baseline explicit-codebook test. At `K=200`, dense requires `0.44 dB`; ODMA arrangements require `1.56`--`2.38 dB`, a penalty of `1.13`--`1.94 dB`.

# 002_B12_n1024 - PBS 2836926.pbs-7
Increasing the alphabet while holding `n=1024` makes the separation clearer. At `K=100`, dense requires `1.00 dB`; ODMA requires `2.38`, `3.25`, or `>4 dB`. From `K=125`, all ODMA arrangements miss the bracket while dense still succeeds.

# 003_B14_n1024 - PBS 2836927.pbs-7
The larger alphabet strengthens the same trend. At `K=60`, dense requires `0.88 dB` while ODMA requires `1.94`--`2.56 dB`. At `K=100`, dense requires `2.81 dB` while all ODMA arrangements miss the bracket.

# 004_B10_n2048 - PBS 2836928.pbs-7
Doubling `n` without scaling `K` makes the regime much easier. At `K=200`, dense requires `-0.56 dB` and ODMA requires `-0.13` to `-0.06 dB`: a smaller `0.44`--`0.50 dB` penalty. Non-monotone/decreasing regions should be treated as finite-size and collision-sensitive behaviour, not as a general scaling law.

# 005_B12_n2048 - PBS 2836929.pbs-7
At the larger frame length, increasing `B` makes the gap visible again. At `K=200`, dense requires `0.31 dB`; ODMA arrangements require `1.00`--`1.44 dB`, a penalty of `0.69`--`1.13 dB`.

# 006_B8_n512 - PBS 2840994.pbs-7
Smaller matched-information-load mirror of job `001`. At `K=125`, dense requires `-0.06 dB`; ODMA requires `0.56`--`1.00 dB`, a penalty of `0.63`--`1.06 dB`. The qualitative ODMA penalty persists, but collisions are more significant than in job `001`.

# 007_B6_n256 - PBS 2840996.pbs-7
Collision-heavy stress test rather than a clean standard-URA comparison. At `K=83`, dense requires `-3.50 dB`; ODMA requires `-3.38` to `-2.62 dB`. The strongly decreasing curves show that `K_a B/n` alone is insufficient once the alphabet is very small and repeated-message effects dominate.

# 008_B12_n2048_scaledK - PBS 2840997.pbs-7; rerun PBS 3199833.pbs-7
Complete. The checkpointed full rerun in `jobs/008_B12_n2048_scaledK/results_full/` is now the canonical job `008` artifact. The older partial and dense-only debris was removed to avoid mixing inconsistent runs.

The full matched comparison confirms the practical ODMA penalty at high load. Dense remains bracketed through `K=333`, requiring `1.00 dB`. At the same point, `d256_b16` and `d512_b8` require `3.69 dB` and `3.25 dB`; `d1024_b2` misses the tested bracket at `K=333` but requires `2.56 dB` at `K=250` while dense requires `0.50 dB`. Averaged over bracketed points, the ODMA-minus-dense NNOMP gap is about `0.51 dB` for `d256_b16`, `0.41 dB` for `d512_b8`, and `0.49 dB` for `d1024_b2`, but the gap is load dependent and reaches `2+ dB` at the top of the grid.

# 012_B12_n2048_genie - PBS 3199834.pbs-7
Complete. Same scaled `B=12`, `n=2048` geometry as job `008`, but using `Genie-OracleSupport`: the true active support is handed to the decoder and counts are estimated by NNLS.

The main result is that the dense-vs-ODMA gap is almost absent when support recovery is removed. For bracketed points, ODMA-minus-dense required-`Eb/N0` gaps average about `0.05`--`0.11 dB` across the three ODMA arrangements, with high-load means about `0.08`--`0.19 dB` for `K>=100`. At `K=333`, dense requires `-2.88 dB`, while `d256_b16`, `d512_b8`, and `d1024_b2` require `-2.56`, `-2.62`, and `-2.56 dB`. This strongly suggests that the large practical ODMA penalty seen under NNOMP-OracleK is mostly support-recovery/dictionary-search difficulty, not a large oracle-support geometry loss.

# 013_B12_n512_scaledK - PBS 3202778.pbs-7
Complete. Smaller-`n` stress probe at fixed `B=12`, using dense plus ODMA arrangements `d64_b16`, `d128_b8`, and `d256_b2`. The results show that compressing the ambient resource length from `n=1024/2048` down to `n=512` makes the dense-vs-ODMA support-recovery gap much sharper.

Dense remains bracketed through `K=62` (`K_aB/n=1.45`), requiring `3.88 dB`, and misses the `12 dB` bracket only at `K=83` (`PUPE=0.074` at the upper bracket). ODMA arrangements break earlier: `d64_b16` and `d128_b8` are bracketed through `K=52`, requiring `11.56 dB` and `6.44 dB` respectively; `d256_b2` is bracketed only through `K=42`, requiring `5.38 dB`, and misses at `K=52`. At the shared high-load point `K=52` (`K_aB/n=1.22`), dense requires `2.19 dB`, while `d64_b16` and `d128_b8` require `11.56 dB` and `6.44 dB`; `d256_b2` is not reached.

# 014_B12_n256_scaledK - PBS 3202779.pbs-7
Complete. Harder smaller-`n` stress probe at fixed `B=12`, using dense plus ODMA arrangements `d32_b16`, `d64_b8`, and `d128_b2`. This run confirms that the effect in job `013` is not caused by low-`B` collision artefacts: here `M=4096`, so repeated-message collisions remain small even though the recovery problem is extremely compressed.

Dense remains bracketed through `K=31` (`K_aB/n=1.45`), requiring `8.12 dB`, and misses only at `K=42`. ODMA arrangements miss much earlier: `d32_b16` and `d128_b2` are bracketed through `K=17` (`K_aB/n=0.80`), while `d64_b8` is bracketed through `K=21` (`K_aB/n=0.98`). At `K=17`, dense requires `2.19 dB`; `d32_b16`, `d64_b8`, and `d128_b2` require `9.12 dB`, `5.75 dB`, and `5.56 dB`.

# 016_B12_n512_genie - PBS 3203628.pbs-7
Complete. Genie-OracleSupport reference for job `013`, matching `B=12`, `n=512`, arrangements, `K_a` grid, seeds, and Eb/N0 bracket.

The result is decisive: the large practical ODMA penalty in job `013` mostly disappears when support recovery is oracle-aided. Mean Genie ODMA-minus-dense gaps are about `0.10 dB` for `d64_b16`, `0.12 dB` for `d128_b8`, and `0.28 dB` for `d256_b2`. Mean NNOMP-minus-Genie support-recovery loss is much larger: about `4.79 dB` for dense, `6.22 dB` for `d64_b16`, `5.39 dB` for `d128_b8`, and `5.23 dB` for `d256_b2`. At `K=52`, dense NNOMP requires `2.19 dB` while dense Genie requires `-2.69 dB`; `d64_b16` NNOMP requires `11.56 dB` while its Genie reference also requires `-2.69 dB`.

# 017_B12_n256_genie - PBS 3203629.pbs-7
Complete. Genie-OracleSupport reference for job `014`, matching `B=12`, `n=256`, arrangements, `K_a` grid, seeds, and Eb/N0 bracket.

The same decomposition is even clearer at `n=256`. Mean Genie ODMA-minus-dense gaps are approximately `0.08 dB` for `d32_b16`, `-0.01 dB` for `d64_b8`, and `0.03 dB` for `d128_b2`, effectively zero at this resolution. Mean NNOMP-minus-Genie support-recovery loss is about `6.07 dB` for dense, `6.48 dB` for `d32_b16`, `6.74 dB` for `d64_b8`, and `5.75 dB` for `d128_b2`. At `K=17`, dense NNOMP requires `2.19 dB` versus dense Genie `-3.12 dB`; `d32_b16` NNOMP requires `9.12 dB` versus Genie `-2.62 dB`.

## Conclusions

- Dense global codebooks consistently outperform ODMA-restricted codebooks under NNOMP-OracleK once the explicit recovery problem is sufficiently stressed.
- The practical gap is small in easy regimes, then grows to roughly `1`--`2+ dB`, or causes ODMA curves to leave the tested bracket earlier than dense.
- Increasing `B` at fixed `n` gives cleaner evidence: repeated-message collision artefacts decrease while the dense-vs-ODMA support-search gap remains.
- Reducing `n` at fixed `B=12` strongly amplifies the practical dense-vs-ODMA gap. This is not mainly a finite-alphabet collision artefact; at the largest tested loads, per-user repeated-message collision probabilities remain only around `1%`--`2%`.
- Jobs `012`, `016`, and `017` show that the oracle-support geometry penalty is small across `n=2048`, `512`, and `256`. ODMA does not lose much once the true support is known.
- The main empirical gap is therefore practical support recovery for structured ODMA dictionaries, not an unavoidable loss in the oracle-supported codebook geometry.
- This conclusion is strongest for the large-alphabet, low-collision URA regime. In a collision-rich low-`B` regime, support knowledge would not make the count/magnitude task trivial, so the same Genie decomposition should not be overgeneralised.
- These are empirical algorithm results under specific decoders and finite sweeps, not an information-theoretic ODMA converse or proof over all ODMA-aware decoders.

## Follow-up

Jobs `001`--`008`, `012`--`014`, `016`, and `017` do not need rerunning. The gap decomposition is now stable enough to use in the supervisor discussion:

```text
Observed ODMA penalty under NNOMP is real.
Oracle-support ODMA penalty is small.
Therefore the main research gap is practical support recovery for structured URA dictionaries.
```

The next experiment should not be another broad dense-vs-ODMA sweep. For the standard large-alphabet URA direction, it should either improve the practical support decoder for fixed dictionaries, or test whether constrained dictionary learning can make support recovery easier without relying on oracle support. Collision-rich count recovery should be treated as a separate, explicitly labelled branch if pursued.
