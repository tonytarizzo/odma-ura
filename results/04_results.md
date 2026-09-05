# URA Framework Results and Current Experimental Status

**Last updated:** 5 September 2026
**Scope:** verified framework results through completed job `027`, plus the implemented and locally certified jobs
`028--029`. Job `023` remains explicitly partial.

## Executive Summary

The repository has moved from representation validation through controlled product and section-domain encoder/decoder
banks. Jobs `018--026` have now been copied back and inspected, with the explicit exception that job `023` is only
partially complete.

The strongest completed conclusions are:

1. The explicit factor framework represents the tested ODMA, reduced sectioned CCS, and reduced CCS-AMP constructions exactly when both paths share the same underlying objects.
2. Fixed-factor dense and ODMA experiments run through a common framework decoder interface and reproduce the previously observed practical support-recovery gap. This is not evidence that ODMA is information-theoretically inferior.
3. Decoder-free geometry objectives improved their own diagnostics at `B=12,n=512`, but did not produce a robust NNOMP performance gain. Decoder-aware learning is therefore the current main path.
4. The new implicit factor implementation performs normal forward and adjoint operations without constructing the global `n x 2^B` codebook. Real sign masks, complex phase masks, ODMA mappings, CCS mappings, numerical adjoints, collision-aware losses, and tiny exact ML have all passed local certification.
5. Job `018` shows that, once the true message support is given to the receiver, dense and several ODMA-style arrangements have similar count-estimation thresholds. The large practical ODMA gap therefore remains primarily a support-search problem. This is an oracle result and its sub-bound thresholds are not achievable URA points.
6. Job `019` successfully runs the pinned CCS-AMP authors' one-pass `B=128,n=38400` core. Enhanced AMP-BP reaches 5% PUPE between `2.25` and `4.0 dB` for `K=10--150`, but is `0.55--1.01 dB` behind the published enhanced points; `K=175` does not reach 5% by `4 dB`. The omitted two-pass/SIC stage is the largest explicit algorithmic difference and is a plausible major source of the gap, but the current evidence does not prove that it is the only source.
7. Job `020` establishes that the repository's `B=100` Triadic10 adaptation is not a usable scalable CCS control: it collapses to zero decoded messages for `K>=50` and has PUPE `1.0` throughout the tested SNR range. It must not be compared with the paper curve.
8. Jobs `021` and `022` completed all 80 encoder/decoder runs with finite training histories and all expected summaries/checkpoints. The implicit operator path scales to `M=16384` without materialising the global codebook.
9. Across the returned product bank, D0 is effectively a calibrated matched-filter/gradient baseline: its mean PUPE gain over matched filtering is only `0.0039`. D1 is meaningfully better at high SNR, improving average PUPE by about `0.042` at 8--12 dB, but the gain is smaller at `B=14` and D1 costs a median `5.2x` the D0 runtime.
10. The all-pairs product encoder is competitive with dense and sparse-global controls but does not show a consistent advantage; no stable best `Q` emerges, and learning `C` is not reliably beneficial. The strongest structural lead remains the sparse-global control: it has the ODMA control's 25% column density but stays near dense performance, separating sparsity from repeated, unbalanced mask geometry.
11. The scalable backend now includes procedural sparse-linear outer coding, exact unit-codeword energy, Binomial local D0, differentiable modular BP, and complete valid-path extraction without a `2^B` state axis.
12. Job `025` closes the implementation bridge: the `L=1` compatibility route is exactly equal to global D0 in logits, soft outputs, and hard outputs for every returned row. Independently trained Binomial D0 differs by less than `0.004` PUPE at every evaluated point. The scalable backend and collision-aware prior do not explain a performance loss by themselves.
13. Job `026` identifies the new difficulty. With `L>1`, the identity/no-outer control is non-identifiable under section-count observations; triadic constraints substantially help at low load, but the current structured encoder plus association remains far behind dense/sparse-global controls at moderate load. This is a practical encoder-plus-decoder gap, not proof that the structured model class excludes an optimum.
14. Modular BP does not improve the current local-D0 association route in jobs `023` or `026`. Its single-user marginal messages are not yet a validated way to resolve multiuser path association.
15. Job `024` is a genuine `B=128` execution and exact-energy certificate, but not a learning success: PUPE is `1.0` throughout, D0 support loss is pinned near `15`, and gradients are initially essentially zero. The large-alphabet denoiser/evidence calibration saturates before useful learning begins.
16. Job `027` completes the sparse-global density frontier. D0 remains near dense through roughly `s=48`; D1 degrades
less sharply but has a less certain transition under its shorter training budget. Equal-density random sparse support
still substantially beats four-mask ODMA, so the next encoder question is how to generate diverse support compactly.
17. Job `028` is implemented to test that question at `B=14`: balanced table supports isolate the one-per-table
restriction, random affine hashes isolate linearity, and collision-selected hashes test offline geometry design. Its
local checks pass; no HPC performance conclusion exists yet.
18. Job `029` adds a focused joint-learning control: dense, iid sparse, and selected-hash amplitudes are trained with
D0/D1 for 120 epochs while sparse supports and exact unit energy are preserved. Local checks pass; HPC results are pending.

The present research conclusion and next question are therefore:

> The explicit sparse-global family has a broad useful density range, but its support and decoder state scale with
> `2^B`. The current sectioned route removes that axis but has not retained recovery. The active next experiment returns
> to global `L=1` geometry and asks whether its supports can be generated by a compact affine-hash skeleton without
> measurable model-class damage at `B=14`, including after allowed amplitudes and decoder co-adapt. Candidate proposal
> and scalable amplitude decoration remain later gates.

---

## 1. Validation Standard

The report separates four levels of evidence.

### 1.1 Exact algebraic validation

At small `B`, construct the direct implementation and framework from the same objects and require machine-precision equality of:

- the induced global dictionary;
- noiseless channel output;
- intermediate decoder estimates where available;
- final decoded lists/counts.

This proves representation equivalence. It does not prove good communication performance.

### 1.2 Local implementation validation

Run forward/backward, adjoint, collision, and exact-recovery tests at manageable size. These tests establish that the code executes the claimed model and gradients, but are not scaling or PUPE results.

### 1.3 Controlled local performance experiments

Train and compare decoders on fixed encoders using identical fresh evaluation batches. These results can support an architectural hypothesis at the tested scale, but require multiple encoder/training seeds before being treated as robust.

### 1.4 HPC performance experiments

Large array jobs produce the main encoder/decoder comparison. A submitted or running job is not a result. Logs, checkpoints, completion status, monotonicity, collision rates, and merged metrics must be inspected first.

---

## 2. Completed Framework Evidence

### 2.1 ODMA representation and inference

Jobs `009`, `010`, and `011` established the original framework surface:

| Job | Construction | Main verified outcome |
|---|---|---|
| `009_framework_odma_job1` | `B=10,n=1024,d=128,Q=16` ODMA | Framework dictionary matches the direct legacy ODMA construction in the exact check; inference has the expected operating range. |
| `010_framework_dense_job1` | Dense `Phi=C` control | Unit-energy Gaussian-like dense codebook behaves as the dense special case. |
| `011_framework_odma_d512_b2` | `B=10,n=1024,d=512,Q=2` ODMA | Wider-placement ODMA runs through the same framework path and preserves its intended sparse structure. |

The cleanest direct comparison is the saved legacy/framework equivalence curve at `B=10,n=256,d=64,Q=4`, `K=2--40`, two SNR points, and 20 seeds:

- `max_phi_abs_err = 0.0`;
- zero difference in mean L1 accuracy;
- zero difference in mean PUPE at every plotted point.

This supports the narrow claim that the framework can exactly represent the tested explicit ODMA codebook and inference path.

The observed dense-versus-ODMA gap remains a practical decoder/support-search result. It must not be promoted to an information-theoretic statement without tiny exact ML/MAP and broader matched-decoder comparisons.

### 2.2 Sectioned CCS and CCS-AMP equivalence

The corrected small sectioned NNLS/tree CCS construction matches its explicit framework representation. The earlier large-`B` scaffold is not labelled a paper reproduction because it differs in parity design, sensing matrices, list handling, and refinement.

The enhanced CCS-AMP adapter invokes the authors' pinned repository at commit `92080d85408d5d19a123d1d61ba76ec6f15451a5`. In the reduced `B=8,L=8,J=2,n=64` comparison over 20 seeds:

- maximum direct/framework dictionary error was `4.44e-16`;
- noiseless signals matched;
- AMP estimates matched exactly;
- final decoded lists matched exactly.

This is an exact representation result in a deliberately tiny alphabet. It is not a competitive PUPE result.

The scalable modes remain separately labelled:

- `paper_b128`: public one-pass core at the paper payload;
- `adapted_b100`: a distinct Triadic10 adaptation, not a paper point.

The final published two-pass AMP/SIC curve uses an undocumented load-dependent `delta` schedule, so the public one-pass sweep must not be described as a complete reproduction of the final plotted curve.

### 2.3 Decoder-free geometry learning

Job `015_geometry_B12_n512` tested ODMA AMP, support-margin, and VAMP geometry objectives at `B=12,n=512,K_a=8`, plus dense AMP.

The objectives improved their own diagnostics:

- AMP objective: `0.002969 -> 0.002415`;
- support-margin objective: `0.613423 -> 0.588275`;
- VAMP objective: `0.001561 -> 0.000840`.

The paired NNOMP threshold barely moved:

- baseline ODMA: approximately `1.42 dB`;
- AMP-trained: approximately `1.42 dB`;
- support-margin-trained: approximately `1.37 dB`;
- VAMP-trained: approximately `1.36 dB`;
- dense AMP changed from approximately `1.17 dB` to `1.23 dB`.

This weakens the claim that coherence-like or AMP/VAMP geometry objectives alone are sufficient. They remain useful diagnostics or pretraining losses, but the main learning experiment now uses a differentiable decoder and direct support/count supervision.

---

## 3. New Reduced `L=1` Product Experiment

### 3.1 Encoder model

The reduced experiment uses:

```text
L = 1
M = 2^B = QV
U = I
T = I
d = n
Phi = [R_1 C | R_2 C | ... | R_Q C]
```

Every `(q,v)` pair is legal. This tests whether transformed copies of a shared local alphabet are useful before introducing resource sparsity or restricted validity constraints.

For the submitted bank:

- `R_q` is a fixed real diagonal sign mask;
- masks are stored compactly as `(Q,n)`, not `(Q,n,n)`;
- `C` is either fixed random or learned;
- normal channel and decoder operations use implicit `matvec/rmatvec`;
- the global codebook is only materialised for explicit diagnostics or legacy oracle baselines.

The ten encoder variants at each `(B,n)` point are:

| Family | Variants |
|---|---:|
| Product, fixed `C` | `Q=4,16,64` |
| Product, learned `C` | `Q=4,16,64` |
| Dense global | fixed and learned |
| Legacy ODMA control | fixed |
| Sparse-global non-product control | fixed |

Each encoder is paired with D0 and D1, producing 20 array elements per `(B,n)` point.

### 3.2 Decoder D0

D0 is an unrolled exact-gradient Bernoulli projected-gradient decoder:

1. collapse the known single-antenna channel;
2. compute the residual through implicit `Phi a`;
3. compute the exact adjoint gradient through implicit `Phi^H r`;
4. apply a calibrated Bernoulli posterior-inspired sigmoid;
5. differentiably normalise the iterate to the known `K_a`;
6. perform hard nonnegative integer projection only at final output.

D0 contains no dense learned message-index matrix and no convolutional locality assumption.

### 3.3 Decoder D1

D1 starts from D0 and adds nonlocal factor-aware latent-slot attention:

- within each pattern `q`;
- across patterns for the same shared local index `v`;
- globally across all candidate messages.

The correction head is initialised to zero, so D1 initially has exactly the D0 forward map. The factor attention is `O(MS)` in the number of candidates and slots, rather than dense `O(M^2)` message attention. It does not assume that adjacent flattened message indices are semantically local.

D1 is an experimental expressive decoder, not a guaranteed improvement. The local comparison below shows faster convergence but not uniform final superiority.

### 3.4 Loss and collisions

The decoder objective is intentionally small:

- balanced support BCE at every layer;
- Smooth-L1 count loss for multiplicities;
- D1 analysis/inverse symmetry loss.

There is no data-consistency loss, count-sum loss, ranking term, or L1 sparsity penalty.

If two users select the same message, the BCE target contains one support-positive coordinate while the count target contains multiplicity two. Thus occasional collisions are not silently discarded or represented by duplicate binary labels.

### 3.5 Training loads and SNR

One `K_a` is sampled uniformly per training batch:

```text
K_min = round(0.4 n / B)
K_max = round(1.2 n / B)
```

| `B` | `n` | train `K_a` | fixed evaluation `K_a` | extrapolation |
|---:|---:|---:|---:|---:|
| 12 | 128 | 4--13 | 4, 9, 13 | 15 |
| 12 | 256 | 9--26 | 9, 17, 26 | 30 |
| 14 | 128 | 4--11 | 4, 7, 11 | 13 |
| 14 | 256 | 7--22 | 7, 15, 22 | 26 |

Training samples one `Eb/N0` value per batch uniformly in dB over `[-4,12]`. Evaluation uses `[-4,0,4,8,12] dB`. Each result records empirical collision frequency, theoretical collision probability, and expected users per pattern `K_a/Q`.

Learned encoders receive 40 D0-surrogate epochs before being frozen. D0 receives 80 decoder epochs because local tests showed that it was still improving after 20. D1 receives 20 epochs because it normally plateaued around epochs 10--15. This prevents an equal-epoch comparison from artificially undertraining the simpler decoder.

---

## 4. Completed Local Learned-Decoder Experiment

### 4.1 Setup

The main local check used:

```text
B = 8          M = 256
n = d = 64     Q = 4     V = 64
K_a = 5        antennas = 1
fixed product encoder
training Eb/N0 sampled uniformly from [-2,8] dB
```

`B=8` was chosen instead of `B=6` because the probability that any collision occurs at `K_a=5` is only about 4%, rather than roughly 15% at `M=64`.

The final fair comparison uses the seed-`8002` D0 checkpoint after 80 epochs and D1 checkpoint after 20 epochs. Both were reevaluated on exactly the same independent batches:

- 50 batches per SNR;
- batch size 64;
- 3,200 realisations and 16,000 user transmissions per SNR;
- evaluation seed `91001`.

### 4.2 Paired PUPE result

| `Eb/N0` | matched filter | D0, converged | D1 | best learned decoder |
|---:|---:|---:|---:|---|
| -2 dB | 0.6198 | 0.6163 | **0.6109** | D1, small difference |
| 0 dB | 0.5188 | 0.5073 | **0.5010** | D1, small difference |
| 2 dB | 0.4105 | **0.3803** | 0.3846 | D0, small difference |
| 4 dB | 0.3294 | **0.2006** | 0.2467 | D0 |
| 6 dB | 0.2680 | **0.0604** | 0.0665 | D0 |
| 8 dB | 0.2185 | 0.0246 | **0.0176** | D1 |

At 8 dB:

- D0 L1 count accuracy: `0.9411`;
- D1 L1 count accuracy: `0.9522`;
- D0 exact count-vector recovery: `0.8931`;
- D1 exact count-vector recovery: `0.9078`.

At 4 dB, D0 is materially better:

- D0 L1 count accuracy: `0.5962`;
- D1 L1 count accuracy: `0.5064`;
- D0 exact count-vector recovery: `0.2916`;
- D1 exact count-vector recovery: `0.1606`.

### 4.3 Interpretation

The local experiment behaves sensibly:

- both learned decoders strongly beat matched filtering once the problem leaves the noise-dominated regime;
- D1 learns much faster and reaches a low training loss in roughly 10--15 epochs;
- D0 needs approximately 60--80 epochs but becomes highly competitive;
- D1 is not uniformly better at convergence;
- D0 is better in the tested 4--6 dB middle regime, while D1 is best at 8 dB.

The first equal-20-epoch comparison made D1 appear dominant because D0 had not converged. The extended D0 control and paired reevaluation corrected that interpretation before the HPC sweep.

This is one encoder/training seed at a reduced alphabet. It validates the pipeline and shows that both learned decoders can work. It does not establish a general D1 advantage.

Artifacts:

- `results/mini_B8_n64_Q4_K5/seed8002/d0_e80/`;
- `results/mini_B8_n64_Q4_K5/seed8002/d1/`;
- `results/mini_B8_n64_Q4_K5/seed8002/paired_d0e80_vs_d1e20.json`.

---

## 5. Returned-Artifact Audit

Jobs `018--022` are now locally present and were inspected directly rather than inferred from scheduler state.

| Job | Returned evidence | Completion assessment |
|---|---|---|
| `018_B8_collision_genie` | 5,458-line stdout, empty stderr, full threshold JSON, bound diagnostics, two plots | Complete |
| `019_CCS_AMP_paper` | 8 load summaries, 8 checkpoints, 8 plots, merged JSON/plot, 3,520 trials | Complete |
| `020_CCS_AMP_B100` | 8 load summaries, 8 checkpoints, 8 plots, merged JSON/plot, 3,200 trials | Complete, scientifically negative |
| `021_product_decoder_pilot` | 20/20 summaries and checkpoints; 400 learned and 400 matched-filter evaluation cells | Complete |
| `022_product_decoder_scale` | 60/60 summaries and checkpoints; 1,200 learned and 1,200 matched-filter evaluation cells | Complete |

All 80 product-bank training histories contain the expected number of epochs and finite values:

- fixed D0: 80 epochs;
- fixed D1: 20 epochs;
- learned-encoder D0: 40 encoder epochs plus 80 decoder epochs;
- learned-encoder D1: 40 encoder epochs plus 20 decoder epochs.

The paired D0/D1 checkpoints contain bit-identical encoder state dictionaries for all 40 encoder pairs. Thus D0 and D1 within a named encoder pair do use the same final codebook.

The array stdout paths were shared, as anticipated. Only 90 pilot and 50 scale stdout lines remain, apparently from the last writers, while the individual result directories are intact. Future arrays should use per-index stdout/stderr paths; the lost console streams do not invalidate the complete JSON histories, but they remove an independent per-subjob log.

---

## 6. Completed Product-Encoder and Learned-Decoder Bank (`021--022`)

### 6.1 What this bank was designed to answer

The experiment deliberately reduced the full framework to `L=1`, `U=T=I`, `d=n` in order to separate four questions:

1. **Representation/computation:** can normal training and decoding scale to `M=2^B` through implicit factor operations?
2. **Product sharing:** does `Phi=[R_1C|...|R_QC]` help relative to a dense global codebook or a sparse-global control?
3. **Encoder learning:** does learning the shared local alphabet `C` help while `R,U,T` remain fixed?
4. **Decoder expressiveness:** does D1's nonlocal factor-aware proximal correction improve on the simpler D0 unrolling?

The bank was not designed to test sparse resource placement with `d<n`, restricted legal `(q,v)` pairs, learned `R`, multiple latent contributions, unknown load/noise, or fading/channel estimation.

### 6.2 Statistical contract and limitations

Each run evaluates four loads at five SNRs, but each cell contains only:

```text
4 evaluation batches x batch size 8 = 32 independent realisations
```

This is adequate for detecting large effects and pipeline failures. It is not adequate for ranking configurations separated by one or two PUPE points. For example, at `K=4`, one user miss in one realisation moves the aggregated PUPE by about `0.0078`; correlations within a realisation make a naive binomial confidence interval optimistic.

There is one training seed per encoder variant. Fixed-versus-learned variants use different seeds, so their random initial codebooks are not paired. D0 and D1 share identical encoder checkpoints, but their evaluation samples are also not paired: the common random generator is advanced by 80 D0 epochs versus 20 D1 epochs before evaluation. The correct strong follow-up is therefore a checkpoint-only reevaluation using common evaluation seeds and many more batches.

Fourteen of 160 D0 SNR curves and 15 of 160 D1 curves have at least one non-monotone step. Most are small, but the worst D1 increase is `0.080`. This is consistent with 32-sample evaluation noise and warns against reading isolated cells literally.

### 6.3 D0 mostly preserves the matched-filter ranking

Across all 800 D0 evaluation cells:

- mean `PUPE_D0 - PUPE_MF = -0.0039`;
- median difference is numerically zero;
- 392 cells improve, 222 are exactly equal at aggregate precision, and 186 worsen;
- only 38 cells improve by more than `0.02`;
- one cell is worse by more than `0.02`.

The gain rises gently with SNR, from `0.0008` PUPE at `-4 dB` to `0.0065` at `12 dB`, but remains small. D0's support loss also ends near `0.5--0.7` in many runs, and its count loss remains close to `0.5`.

This makes D0 useful as a cheap, model-consistent control, but not yet a convincing learned recovery algorithm. A likely interpretation is that scalar monotone calibration plus mass normalisation changes score magnitudes more than final top-`K_a` ordering. The exact implicit residual/adjoint iterations do not by themselves guarantee movement away from the matched-filter support ranking in eight layers.

This matters for encoder learning: every learned `C` was trained through a D0 surrogate. If D0 has weak support-search gradients, the absence of an encoder-learning gain may reflect the surrogate as much as the product-codebook hypothesis.

### 6.4 D1 gives a real high-SNR decoder gain, but not specifically a product gain

Across all 800 D1 evaluation cells:

- mean `PUPE_D1 - PUPE_MF = -0.0195`;
- median difference is `-0.0073`;
- D1 beats its matched-filter control in 540 cells, ties in 74, and loses in 186;
- 243 cells improve by more than `0.02`, while 20 worsen by more than `0.02`.

The gain is strongly SNR-dependent:

| `Eb/N0` | D0 mean change from MF | D1 mean change from MF | D1 cells better than MF |
|---:|---:|---:|---:|
| -4 dB | -0.0008 | -0.0003 | 66/160 |
| 0 dB | -0.0022 | -0.0046 | 99/160 |
| 4 dB | -0.0043 | -0.0081 | 102/160 |
| 8 dB | -0.0058 | **-0.0417** | 138/160 |
| 12 dB | -0.0065 | **-0.0429** | 135/160 |

D1 therefore helps most after thermal noise stops dominating and structured multiuser interference/support ambiguity becomes the main error source. That is exactly where a context-dependent proximal map should help.

However, D1 also improves dense and sparse-global controls, sometimes more than product encoders. For dense `Q=1`, the cross-pattern same-`v` relationship is absent, yet D1 still has global and within-pattern context. Therefore the current result supports **nonlocal learned context** but does not isolate the benefit of the `q/v` factorisation. A capacity-matched ablation removing pattern/value pooling while retaining the global learned map is required.

### 6.5 Main geometry-level result

The table below averages PUPE over all four loads at `8` and `12 dB`. It is a high-SNR summary, not a substitute for the full curves.

| Geometry | D0 mean gain over MF, all cells | D1 mean gain over MF, all cells | Best high-SNR D1 run | Best D1 PUPE | Dense D1 average | Product D1 average | Sparse-global D1 | ODMA D1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `B=12,n=128` | 0.0016 | 0.0149 | dense learned | 0.2858 | 0.2912 | 0.3118 | 0.3399 | 0.5417 |
| `B=12,n=256` | 0.0050 | **0.0360** | product learned `Q=64` | **0.1934** | 0.2185 | 0.2268 | 0.2182 | 0.4561 |
| `B=14,n=128` | 0.0026 | 0.0103 | dense fixed | 0.3453 | 0.3556 | 0.3717 | 0.3714 | 0.5886 |
| `B=14,n=256` | 0.0065 | 0.0168 | dense fixed | 0.3020 | 0.3099 | 0.3264 | 0.3040 | 0.5295 |

Three conclusions are stable enough to retain:

1. D1 is strongest at `B=12,n=256`; its advantage becomes much weaker at `B=14`.
2. Product encoders are viable but do not outperform dense controls on average. Their high-SNR average is worse than the dense average at all four geometries by roughly `0.008--0.021` PUPE.
3. Sparse-global is within `0.016` PUPE of dense at every geometry and is essentially tied with dense at both `n=256` points, despite each codeword using only 25% of the resource rows.
4. The ODMA control remains much worse under the same generic decoder. D1 narrows its gap but does not remove it.

The ODMA result remains an approximate-decoder result, not a statement about the induced codebook's Bayes-optimal performance. Here ODMA also retains sparse placement while product/dense runs use `d=n`; the comparison deliberately asks whether the generic decoder can recover each construction, not whether all geometries have equal constraints.

### 6.6 No stable best `Q`

The best product `Q` changes with geometry and whether `C` is fixed or learned:

| Geometry | Best fixed-product D1 at high SNR | Best learned-product D1 at high SNR |
|---|---:|---:|
| `B=12,n=128` | `Q=4`, PUPE 0.3061 | `Q=4`, PUPE 0.2946 |
| `B=12,n=256` | `Q=16`, PUPE 0.2213 | `Q=64`, PUPE 0.1934 |
| `B=14,n=128` | `Q=64`, PUPE 0.3648 | `Q=4`, PUPE 0.3651 |
| `B=14,n=256` | `Q=64`, PUPE 0.3091 | `Q=4`, PUPE 0.3271 |

There is no monotone improvement with more patterns, no consistent occupancy sweet spot, and no repeatable `Q` winner. The standout `B=12,n=256,Q=64` learned result is interesting but is one seed with unpaired 32-sample cells; it is a candidate for replication, not a selected design.

### 6.7 Learning `C` is inconclusive and often neutral

For D1, the mean high-SNR learned-minus-fixed change is:

| Geometry | Dense learned minus fixed | Product learned minus fixed, averaged over `Q` |
|---|---:|---:|
| `B=12,n=128` | -0.0110 | -0.0068 |
| `B=12,n=256` | -0.0096 | -0.0083 |
| `B=14,n=128` | +0.0206 | -0.0038 |
| `B=14,n=256` | +0.0158 | +0.0061 |

Negative is better. Thus learning appears mildly helpful at `B=12`, mixed at `B=14,n=128`, and harmful or neutral at `B=14,n=256`. Individual `Q` results vary even more.

This is not evidence that codebook learning is impossible. It says that **40 epochs through the current D0 surrogate, one seed, and unpaired low-sample evaluation do not establish a transferable encoder gain**. The fixed and learned variants also start from different random seeds, so some apparent changes can be ordinary codebook-seed variation.

### 6.8 Load, payload, and collision effects

As expected, PUPE worsens with load and improves with SNR. Extrapolation loads do not trigger a qualitative decoder collapse, which is encouraging, but high-load PUPE remains large even at 12 dB.

Increasing the payload from `B=12` to `B=14` quadruples `M` while leaving the observation length fixed. D1's average improvement over matched filtering falls from `0.0149/0.0360` at `B=12` to `0.0103/0.0168` at `B=14`. This is a genuine scaling warning: the current D1 capacity/training schedule does not preserve its B12 advantage as the candidate set grows.

Message collisions remain a secondary effect for per-user PUPE, although the probability of at least one collision is noticeable at the largest B12 extrapolation load:

| Point | Probability of any collision | Per-user collision probability |
|---|---:|---:|
| `B=12,K=15` | 2.53% | 0.34% |
| `B=12,K=30` | 10.10% | 0.71% |
| `B=14,K=13` | 0.48% | 0.07% |
| `B=14,K=26` | 1.97% | 0.15% |

Empirical collision estimates are noisy because each cell has only 32 samples. The maximum observed cell rate is 25%, while the maximum theoretical rate is 10.1%.

Every returned hard estimate has `total_count_err=0`, but this is guaranteed by the final oracle-`K_a` integer projection. It is not evidence that activity was inferred. Exact count-vector recovery is common only at low load/high SNR and falls rapidly with load.

### 6.9 Runtime and scalability

The 80 summaries record 77.3 aggregate wall-hours. The important computational result is that all `B=14,n=256,M=16384` runs complete through implicit forward/adjoint operations without constructing an `n x M` codebook during normal training.

D1 is expensive:

- D1/D0 runtime ratio: `2.8x--20.1x`, median `5.2x`;
- longest D1 run: 229 minutes;
- longest D0 run: 71 minutes;
- B14 D1 runs commonly take 2--3.8 hours.

Thus D1 is computationally feasible at this scale but not yet a low-complexity decoder. The latent-slot implementation avoids dense `O(M^2)` attention, yet its `O(MS)` feature analysis across eight layers still dominates runtime.

### 6.10 Post-hoc matrix and support-geometry audit

The returned D0 and D1 checkpoints contain identical encoders for each run pair, so the 40 distinct D0 checkpoints are sufficient for an encoder audit. The global matrices were materialised only post hoc and measured by exact support density, number of distinct support masks, effective support, row-energy balance, row-Gram spectrum, and 30,000 sampled column pairs per checkpoint.

The central sparse comparison is:

| Property | Dense fixed | Product fixed/learned | Sparse-global fixed | ODMA fixed |
|---|---:|---:|---:|---:|
| Exact nonzero density per codeword | 100% | 100% | 25% | 25% |
| Distinct exact support masks | 1 | 1 | `M` | 4 |
| Row-energy coefficient of variation | 0.011--0.023 | 0.021--0.179 | 0.023--0.053 | **0.839--0.875** |
| Row-Gram eigenvalue CV | 0.087--0.250 | 0.090--0.305 | 0.089--0.249 | **0.848--0.891** |
| Mean absolute sampled column correlation, `n=128` | 0.071 | about 0.071 | 0.069 | **0.087--0.089** |
| Mean absolute sampled column correlation, `n=256` | 0.050 | about 0.050 | 0.049 | **0.061** |

The ODMA imbalance is not a subtle learned effect. Its four independently sampled size-`n/4` placement masks cover resources as follows:

| Geometry | Rows unused by every mask | Rows used by multiple masks |
|---|---:|---:|
| `B=12,n=128` | 32.0% | 26.6% |
| `B=12,n=256` | 30.1% | 25.0% |
| `B=14,n=128` | 29.7% | 23.4% |
| `B=14,n=256` | 30.5% | 26.6% |

This is close to the `31.6%=(3/4)^4` uncovered fraction expected from four independent random quarter-support masks. Within one ODMA pattern, all columns occupy the same `d=n/4` subspace. Their mean absolute correlation is about `0.143` at `n=128` and `0.100` at `n=256`, roughly twice the dense/sparse-global values. Sparse-global columns also use only `n/4` resources, but every message receives an independently sampled support; aggregate row loading is balanced and pairwise overlap is distributed rather than concentrated into identical-mask and cross-mask cases.

This explains why the current result does **not** say that sparsity itself causes the ODMA loss. It says that coarse reuse of a very small, unbalanced support bank is harmful under this decoder and codebook. Sparse-global already demonstrates that 25% resource use can retain near-dense performance. Its limitation is scalability: it stores an independent support and codeword for every one of the `M` messages, so it is a statistical gold-standard control rather than a compact ODMA replacement.

Learning did alter the matrices, but in the opposite direction from sparse structure. Fixed Gaussian dense/product codewords have mean effective support around `0.34n`; learned product codewords increase this to roughly `0.39n--0.65n`, with fewer small-magnitude entries. Thus the current loss encourages amplitude flattening/diffusion. It cannot learn resource sparsity because `d=n`, every product operator is a fixed dense diagonal sign mask, `R` is frozen, and there is no support or sparsity constraint.

### 6.11 Product-bank answer

The bank answers its four questions as follows:

| Question | Answer from current evidence |
|---|---|
| Can the implicit framework scale to `M=16384`? | **Yes**, operationally validated. |
| Does product sharing beat dense/sparse-global controls? | **No consistent evidence.** Product is competitive but slightly worse on average. |
| Does learning `C` help? | **Inconclusive/mixed.** Mild B12 gains do not persist at B14. |
| Does D1 help over D0/MF? | **Yes at moderate/high SNR**, especially B12; weaker at B14. |
| Is D1's gain specifically factor-aware? | **Not yet shown.** Generic global-context capacity is confounded. |

---

## 7. Collision-Rich Oracle-Support Experiment (`018`)

### 7.1 Purpose

Job `018` asks a different question from the low-collision product bank. At `B=8,M=256,n=512`, `K=25,62,125`, two antennas, and 50 seeds, it gives the decoder the exact transmitted-message support and asks nonnegative least squares plus integer projection to estimate counts.

This isolates active-column conditioning/count estimation from the combinatorial support-search problem.

### 7.2 Required `Eb/N0` for PUPE <= 0.05

| Arrangement | `K=25` | `K=62` | `K=125` |
|---|---:|---:|---:|
| `d64_b16` | -1.391 dB | -1.859 dB | -1.781 dB |
| `d128_b8` | -1.625 dB | -1.625 dB | -1.703 dB |
| `d256_b2` | -1.547 dB | -1.781 dB | -1.625 dB |
| dense | -1.625 dB | -1.625 dB | -2.172 dB |

All 12 searches bracket the target with no mean-curve monotonicity violations. The curves use 50 seeds, but the threshold bootstrap intervals are still broad because the evaluated SNR grid is sparse. For example:

- `d64_b16,K=25`: `[-1.625,4.0] dB`;
- most `K=62/125` intervals: about `[-1.94,-1.0] dB`;
- dense `K=125`: `[-2.25,-1.625] dB`.

The apparent dense advantage at `K=125` is therefore suggestive, not decisive. The main robust observation is that all arrangements occupy roughly the same oracle-support regime.

![Job 018 oracle-support arrangement thresholds](../jobs/018_B8_collision_genie/results/arrangement_sweep_required_ebn0_ci.png)

### 7.3 Correct interpretation of the bound comparison

The oracle curves lie about 5 dB below the plotted canonical/count Polyanskiy RCU benchmark. This is not a violation of the finite-blocklength URA benchmark because the receiver is given the true support. It has already received most of the information that an actual URA decoder must discover.

At B8, the strict collision-as-error floors are:

- `K=25`: 4.55%, barely below the 5% target;
- `K=62`: 11.03%, making 5% strictly impossible;
- `K=125`: 20.76%, also impossible.

The plotted PUPE metric is support/list based. Once a collided message remains nonzero in the estimate, underestimating its multiplicity contributes to L1 count error but not necessarily to missed-user PUPE. Therefore the canonical, strict, and count/multiset interpretations must remain separate.

The scientifically useful conclusion is narrow but important:

> Once active message identities are known, count estimation is comparatively easy and is not strongly controlled by the tested placement arrangement. The difficult part of the earlier dense-versus-ODMA gap is support discovery, not merely solving amplitudes on the correct support.

---

## 8. CCS-AMP Author-Code Experiments (`019--020`)

### 8.1 Paper-scale one-pass core (`019`)

Job `019` uses the pinned authors' code at commit `92080d85408d5d19a123d1d61ba76ec6f15451a5`, with the paper-core dimensions `B=128,n=38400`, 16 sections of 16 bits, 10 AMP iterations, one BP iteration, 20 seeds, and eight loads. It evaluates both enhanced AMP-BP and the original AMP mode.

The enhanced one-pass required-SNR results are:

| `K` | Author-code one-pass | Published enhanced point | One-pass minus paper | Polyanskiy canonical | One-pass minus Polyanskiy |
|---:|---:|---:|---:|---:|---:|
| 10 | 2.25 dB | 1.70 dB | +0.55 dB | -0.503 dB | +2.75 dB |
| 25 | 2.50 dB | 1.85 dB | +0.65 dB | -0.643 dB | +3.14 dB |
| 50 | 3.00 dB | 2.08 dB | +0.92 dB | -0.706 dB | +3.71 dB |
| 75 | 3.00 dB | 2.31 dB | +0.69 dB | -0.617 dB | +3.62 dB |
| 100 | 3.25 dB | 2.38 dB | +0.87 dB | -0.414 dB | +3.66 dB |
| 125 | 3.50 dB | 2.65 dB | +0.85 dB | -0.186 dB | +3.69 dB |
| 150 | 4.00 dB | 2.99 dB | +1.01 dB | 0.043 dB | +3.96 dB |
| 175 | not reached by 4 dB | 3.12 dB | >+0.88 dB | 0.288 dB | >+3.71 dB |

At 4 dB the enhanced decoder's PUPE ranges from zero at `K=10/25` through 0.041 at `K=150` and 0.072 at `K=175`. The original mode does not reach 5% anywhere in the grid; its 4 dB PUPE ranges from 0.057 to 0.120.

![Job 019 paper-scale CCS-AMP one-pass results](../jobs/019_CCS_AMP_paper/results/merged/ccs_amp_validation.png)

This is strong evidence that:

1. the pinned public AMP-BP core is integrated and behaves sensibly at paper scale;
2. enhanced BP information materially improves over the original mode;
3. the public one-pass core does **not** reproduce the final published enhanced curve;
4. the missing two-pass AMP/SIC stage and unpublished load-dependent remainder schedule are plausibly quantitatively important, but the one-pass comparison cannot identify how much of the `0.6--1.0` dB discrepancy they explain.

The paper's two-pass procedure removes the `K_a-delta` first-round messages with the highest likelihood, subtracts their waveforms, and reruns AMP plus tree decoding on the residual to recover the remaining `delta`. This can create a waterfall improvement: correctly removing most users reduces both multiuser interference and the effective sparsity of the second problem. It can also amplify errors if false first-round messages are cancelled. The unpublished, load-dependent `delta` therefore controls an important precision/recall tradeoff rather than merely setting another AMP iteration count.

The current adapter matches the public one-pass dimensions, graph, transform family, and main AMP/BP core, but it uses a deterministic transform seed and a finite 20-seed, 0.25-dB threshold grid. Together with unspecified simulation details in the plotted curve, these are secondary possible sources of discrepancy. The run is therefore a faithful public-core validation, not a full paper reproduction, and the missing two-pass schedule should not be fitted post hoc to force agreement with the paper points.

### 8.2 `B=100` Triadic10 adaptation (`020`)

Job `020` is explicitly non-paper-comparable. Its purpose was to test whether the authors' implicit AMP/BP machinery could be adapted into a scalable `B=100` control, closer to the payload used by the project's standard URA comparisons, without ever enumerating `2^100` messages. It changes the graph to 20 ten-bit sections through `Triadic10(10)` while retaining `n=38400`; it was an adaptation/scalability stress test, not a paper reproduction.

The adaptation fails as a scalable decoder:

- at `K=10`, enhanced reaches exactly 5% at 5 dB;
- at `K=25`, enhanced/original PUPE at 5 dB is 0.718/0.682;
- for every `K>=50`, both modes have PUPE 1.0 across the entire 0.5--5 dB grid;
- at `K>=50`, the graph decoder returns essentially zero codewords even at 5 dB.

![Job 020 B100 adaptation results](../jobs/020_CCS_AMP_B100/results/merged/ccs_amp_validation.png)

This is a structural failure of the adaptation, not a marginal SNR shortfall. The main mechanism is the much smaller section alphabet. The outer decoder retains `ell=K+10` candidates per section. For a triadic additive check and approximately independent candidate lists, the expected number of locally surviving triples scales as `ell^3/2^J`. At `K=50`, this heuristic is about `60^3/1024=211` candidates per check for `J=10`, versus `60^3/65536=3.3` for the paper's `J=16`. Even at `K=25`, it is about 42 versus 0.65. The ten-bit graph therefore becomes extremely ambiguous as load grows; its hard root-conditioned propagation does not collapse to valid full paths and returns zero messages. Correlated lists and the cyclic graph mean this calculation is diagnostic rather than an exact theorem, but its load trend matches the observed `K=10` partial success, `K=25` degradation, and `K>=50` collapse.

Lower per-section signal energy and changed AMP operating conditions may also contribute, and the current checkpoints do not record pre-tree top-list recall. Increasing the SNR range is nevertheless not the appropriate next response. The graph alphabet/rate/list design must be changed or the mode retired. The result validates the earlier insistence that a nominal payload match (`B=100`) is not enough to make an adapted CCS construction paper-faithful or useful.

---

## 9. Receiver-Knowledge Audit

The product bank assumes the receiver knows:

| Quantity | Status |
|---|---|
| Full factorised encoder `C,R,U,T` | Known system design; used through exact forward/adjoint operations. |
| Realised `K_a` | Oracle-known; used in the Bernoulli prior, mass normalisation, and hard count projection. |
| AWGN variance / selected `Eb/N0` | Oracle-known scalar used for calibration. |
| Channel `H` | Known and trivial: one constant antenna with `H=1`. |
| Message prior | Uniform i.i.d., giving `rho=K_a/M`. |

The product decoders do not know active identities, multiplicities, collision locations, the noise realisation, or true per-pattern occupancy. D1 estimates soft pattern occupancy from its iterate.

Job `018` is much more strongly oracle-aided: it receives the exact message support and `K_a`. Job `019/020` uses each CCS construction's own matched graph/AMP decoder and known load/noise conventions.

These distinctions prevent three invalid comparisons:

1. job `018` cannot be treated as an achievable communication scheme or placed below Polyanskiy as a practical point;
2. product-bank D1 gains are conditional on known load and noise;
3. CCS-AMP results use a matched specialised decoder and should be an external benchmark, not described as the same generic decoder comparison.

---

## 10. Consequences for the General Project

### 10.1 What is now genuinely validated

The project now has evidence for all of the following:

- one algebraic framework can exactly represent the tested ODMA, reduced sectioned CCS, and reduced CCS-AMP constructions;
- equivalent induced dictionaries give identical outputs when the same decoder and initial objects are used;
- normal factorised training/inference can operate implicitly at `M=16384`;
- collision-aware count targets and hard output projection work without crashing or losing total mass;
- a nonlocal learned proximal decoder can improve support recovery beyond scalar calibrated gradient steps;
- public paper-scale CCS-AMP can be exercised through a pinned external implementation;
- the support-search stage, not oracle-known-support count estimation, is the dominant practical bottleneck in the tested ODMA/dense comparison.

### 10.2 What the new results do not validate

The returned bank does not yet establish:

- that product sharing improves the induced codebook;
- that `Q` has an optimum transferable across `B,n,K`;
- that the learned `C` is better than a matched random control;
- that D1's gain comes from factor awareness rather than generic global capacity;
- robustness to unknown `K_a` or unknown noise;
- an advantage under an independent decoder such as matched AMP, NNOMP, or exact ML;
- state-evolution validity for D0/D1;
- a successful, well-calibrated scalable `L>=2` factor decoder;
- a reproduction of the published two-pass CCS-AMP curve;
- a viable `B=100` CCS adaptation.

### 10.3 Research interpretation

The broad project hypothesis was that structured/factorised encoders could retain or improve statistical performance while making the exponential message space computationally tractable. The current evidence splits that hypothesis in two:

1. **Computational half:** supported. The factor interface and implicit operators scale, and D1 can use factor labels without dense `M x M` transforms.
2. **Statistical half:** not yet supported for the all-pairs sign-mask product family. Dense and sparse-global controls remain at least as good on average, and learned `C` does not transfer reliably.

That is a useful negative result. It says not to proceed automatically to learning `R`, adding restricted pairs, or building a larger `L>=2` graph. Those additions create more capacity and more confounding before the simplest product hypothesis has earned them.

The decoder results point to a more precise direction: improve **support-ranking dynamics** and isolate which nonlocal contexts matter. D0's near-equivalence to matched filtering and job `018`'s easy oracle-support performance both indicate that active-set discovery is the core problem. D1 partially addresses it, especially at high SNR, but its generic gains and runtime show that the current architecture is an exploratory proof of capability rather than the final algorithm.

---

## 11. Recommended Next Experiments

This section records the recommendation after jobs `021/022`. The returned section-domain evidence in Sections 13--15
supersedes its ordering: decoder calibration and complete-path association now come before another sparse-structure bank.

### Priority 1: bridge sparse-global performance and ODMA compactness

Treat the current `021/022` bank as representative and move to a fixed sparse-structure pilot. Match every sparse encoder at support budget `s=n/4` and unit column energy, then vary support diversity and balance rather than merely varying density:

1. dense Gaussian control;
2. sparse-global independent-support gold standard;
3. current four-mask random ODMA control;
4. four exactly balanced/disjoint masks;
5. balanced random mask banks with `Q in {16,64}`;
6. compact composed/hashed masks that generate many message supports from a small factor library.

For each support bank, separate two effects: independent local codewords per message versus an all-pairs shared `C` alphabet. This identifies whether performance is lost through repeated placement masks, local-codeword reuse, or both. Start at `B=12,n=256`; use one `B=14,n=256` point only as a scaling confirmation.

Primary structural diagnostics should be support count, uncovered-row fraction, row-load CV, overlap distribution, within-pattern correlation, sampled active-set conditioning, and PUPE. Resource efficiency must be reported separately as (i) per-codeword support `s/n`, (ii) encoder/operator storage, and (iii) decoder complexity. Sparse-global wins the first criterion but not the latter two.

### Priority 2: add cheap identifiability and decoder controls to that pilot

Before the scalable run, compare dense, current ODMA, balanced-mask, and sparse-global constructions with tiny-`B` exact ML/MAP. This tests whether the mask geometry changes the induced statistical problem independently of D1. In the scalable pilot, retain D0, D1, and matched filtering and add one independent model-based decoder if affordable.

Use D1 provisionally as the stronger workhorse, but include a capacity-matched global-only learned-proximal control at the representative point. This is enough to test whether D1's benefit truly comes from `q/v` relations without delaying the sparse-geometry branch for a full decoder-ablation bank.

### Priority 3: learn within a fixed sparse budget only after the fixed pilot

The `021/022` bank did not learn sparsity: `R` was frozen, product `d=n`, and learning touched only dense `C`. If balanced high-diversity fixed masks retain sparse-global performance, then learn one constrained object at a time:

1. fixed support, learned nonzero signs/amplitudes;
2. fixed balanced support bank, learned `C`;
3. discrete/top-`s` support selection with an exact column budget and row-balance constraint;
4. only then consider joint `R,C` training.

Save before/after matrices and evaluate with multiple decoders so that decoder co-adaptation is visible. Avoid an unconstrained soft `R`: it would become dense and would reintroduce the `R,C` gauge ambiguity.

### Priority 4: strengthen the model-based decoder ladder

D0 needs a mechanism that can change candidate ordering more strongly. Low-complexity options, in order, are:

1. momentum or residual memory across unrolled layers;
2. a small candidate-wise MLP using residual/evidence history;
3. learned damping/step functions conditioned on load and residual variance;
4. a matched AMP/Bernoulli denoiser control with explicit iteration diagnostics;
5. candidate pruning between layers to reduce D1 cost.

Retain exact `Phi`/`Phi^H` operations and avoid dense learned message-index matrices.

### Priority 5: separate robustness from oracle calibration

After the sparse-structure pilot:

- train over a `K_a` range as now, but evaluate with perturbed/estimated `K_a`;
- add a noise-blind or noise-estimating calibration ablation;
- report graceful degradation rather than only the oracle-known setting.

### Priority 6: CCS and collision branches

- Keep job `019` as the pinned one-pass public-core anchor.
- Do not spend more compute on `adapted_b100` without redesigning the graph/list decoder.
- For collision-rich studies, report both support PUPE and multiplicity-aware L1/count error; never use the oracle-support curve as an achievable bound.
- Do not implement the paper's SIC stage merely to tune `delta(K)` against the plotted targets. If exact reproduction becomes important, first obtain the schedule/procedure from the authors or pre-register a load-only selection rule; otherwise retain job `019` as the honest one-pass anchor.

---

## 12. Scalable Section-Domain Refactor

The framework now has two deliberately separate execution backends for the same algebra

\[
\Phi=\sum_\ell F_\ell T_\ell,\qquad F_\ell=B_\ell U_\ell.
\]

The original explicit backend remains available for small-`B` exact certification and legacy experiments. Its
`ProductComponent` stores `msg_to_atom[M]`, and its decoders retain global `(batch,M)` states. The new section-domain
backend removes that axis:

- `LocalAtomBank` owns only `R`, `C`, and the valid local atoms represented by `U`;
- `SectionedEncoder` synthesises `y=sum_l F_l s_l` from local count tensors `(batch,N_l)`;
- active messages are stored as paths `(batch,K_max,L)` and scattered directly into local counts;
- `local_adjoint` returns one evidence tensor `(batch,N_l)` per section;
- section-domain D0 maintains `sum_l N_l` states and never returns a global message-count vector;
- complete path association and sparse-linear outer parity/BP are implemented separately from the physical operator.

The refactor has passed the following local certification:

1. For an explicit four-section construction, section-domain and global-message synthesis agree to numerical precision.
2. The local adjoints, mapped through the small explicit `T_l` objects only in the test, reconstruct the original global
   adjoint exactly.
3. For `L=1`, global D0 and section-domain D0 have matching layer logits, soft counts, hard counts, balanced support/count
   loss, and all decoder-parameter gradients.
4. A separate multi-section learning check produces finite loss and finite gradients for every learned `C_l` and the
   section-domain decoder.
5. A `B=100,L=10,N_l=1024` construction generates a channel batch and executes a decoder layer with exactly 10,240
   local states. The scalable objects have no `num_codewords`, `msg_to_atom`, `(batch,2^100)` count tensor, or global
   `Phi`.
6. The pre-existing implicit-factor check, a small legacy ODMA algebra/inference run, a small CCS/tree equivalence run,
   and a mini product-D0 training/evaluation run still pass after the refactor. Existing checkpoint parameter keys remain
   unchanged.

This establishes implementation equivalence and removes the global-message storage barrier. Job `025` now independently
confirms exact `L=1` compatibility on the HPC path, while job `024` confirms that the complete `B=128` representation can
execute with exact codeword energy. The latter does not yet decode successfully: its local evidence saturates before the
model can learn.

---

## 13. Returned Sectioned Experiments (`023--024`)

### 13.1 Completion audit

| Job | Expected rows | Summaries/checkpoints | Audit status |
|---|---:|---:|---|
| `023_sectioned_outer_pilot` | 26 | 16 | Partial: ten result directories are empty. Shared PBS log names were overwritten, so the failure mode cannot be reconstructed reliably. |
| `024_sectioned_B128_scale` | 8 | 8 | Complete; every result is finite and has a checkpoint. |

The completed part of `023` covers all principal triadic controls and some random sparse-graph rows, but several graph
degree/rank cells and one seed of the `J=8` row are missing. Those missing cells should not be rerun wholesale: the broad
random-graph portion changed too many factors at once and no longer addresses the most discriminating next question.

### 13.2 `B=16` pilot result (`023`)

Mean PUPE below averages the four evaluation SNR points and available seeds. Rows with one seed are explicitly marked and
are not reliable rankings.

| Completed configuration | Seeds | Trained D0 | Trained D0+BP | BP minus D0 |
|---|---:|---:|---:|---:|
| `J=4`, triadic, exact energy, learned, `K=2` | 2 | **0.5500** | 0.5640 | +0.0139 |
| `J=4`, random rank 4/degree 3, `K=4` | 2 | 0.5087 | 0.5765 | +0.0679 |
| `J=4`, triadic, exact energy, learned, `K=4` | 2 | 0.5117 | 0.5881 | +0.0764 |
| `J=4`, random rank 4/degree 2, `K=2` | 1 | 0.6113 | 0.6284 | +0.0171 |
| `J=4`, triadic, exact energy, fixed, `K=2` | 2 | 0.6833 | 0.6858 | +0.0024 |
| `J=8`, random rank 2/degree 2, learned, `K=2` | 1 | 0.6797 | 0.7012 | +0.0215 |
| `J=2`, triadic, exact energy, learned, `K=2` | 2 | 0.6921 | 0.7085 | +0.0164 |
| `J=4`, random rank 2/degree 2, `K=2` | 2 | 0.7595 | 0.7468 | -0.0127 |
| `J=4`, triadic, overlapping bank, learned, `K=2` | 2 | 0.7769 | 0.7998 | +0.0229 |

The cleanest within-family result is that learned exact-energy `J=4` triadic banks improve over fixed banks: mean BP PUPE
falls from `0.6858` to `0.5640`, and at 14 dB it falls from `0.452` to `0.190`. `J=4` also outperforms the returned
`J=2` triadic control. The incomplete `J=8` and random-graph cells do not support a general alphabet or graph ranking.

The overlapping sampled-energy row is both worse and non-deployable as returned. Across its two checkpoints the sampled
codeword energies range approximately from `0.82` to `1.29`; the exact orthogonal mode stays within `5.4e-7` of one.
Exact projected energy should remain the default.

BP is worse than D0 in eight of the nine completed configuration aggregates. More importantly, the outer-loss curriculum
is numerically ill-conditioned: pre-clipping gradient norms jump from ordinary values to as high as `10^15` when the outer
loss is ramped. Gradient clipping keeps checkpoints finite, but it does not make this a trustworthy optimisation of BP.
This weakens fine configuration rankings, although the exact-energy and broad D0 trends remain useful.

### 13.3 `B=128` scale result (`024`)

All eight rows completed for `B=128,J=16,n=38400`, two seeds each at `K=10,25,50,75`. The run certifies the intended
scalable representation:

- local state size is `16 x 65,536 = 1,048,576`, with no `2^128` object;
- codeword-energy deviation is at most `4.8e-7`;
- every checkpoint and metric is finite.

It does **not** certify useful learning. Initial PUPE, trained D0 PUPE, and trained BP PUPE are all exactly `1.0` for every
load and SNR. Mean total loss remains approximately `16.24`; its support component is approximately `15.0`. Initial
gradient norms are only `10^-14--10^-13`.

The support loss value diagnoses clamp saturation: the balanced BCE sees positive logits at the fixed lower clamp near
`-30`, giving about `0.5 x 30 = 15`, while the negative term is nearly zero. The large local alphabet and current
step/evidence calibration drive pseudo-data below the denoiser threshold, so evidence is killed before parameters can
learn. Candidate caps and beam search occur later and cannot cause this initial saturation. More epochs or an unchanged
rerun would waste compute; evidence scale, denoiser centring/temperature, and clamp occupancy must be instrumented and
fixed first.

## 14. Completed Controlled Bridge (`025--026`)

### 14.1 `L=1` backend and prior bridge (`025`)

All six rows completed. For dense, 25%-sparse global, and four-placement ODMA codebooks across both seeds, the global D0
and `L=1` compatibility backend have exactly zero maximum difference in layer logits, soft estimates, and hard outputs.
This is an exact behavioural bridge, not just similar aggregate performance.

Separately trained Binomial D0 is also practically neutral at this scale. Across every load and SNR, its maximum absolute
PUPE difference from global Bernoulli D0 is below `0.0037`; mean absolute differences are `0.0005--0.0007` by encoder
family. The high-SNR dense/sparse/ODMA landscape agrees with job `022`. Therefore neither the section-domain backend nor
the Binomial local prior caused the structured gap seen in `026`.

### 14.2 `L>1` encoder/decoder bridge (`026`)

All six rows completed. The induced global signal agrees with section synthesis within `6.0e-7`; exact-energy deviation is
below `6.6e-7`. The table reports average PUPE over 8 and 12 dB, where association differences are easiest to see.

| Encoder and load | Induced global D0 | Local D0 + association | Local D0 + BP + association |
|---|---:|---:|---:|
| identity fixed, `K=9` | 0.9045 | 0.9453 | n/a |
| identity fixed, `K=17` | 0.9301 | 0.9692 | n/a |
| identity fixed, `K=26` | 0.9303 | 0.9850 | n/a |
| identity fixed, `K=30` | 0.9266 | 0.9862 | n/a |
| triadic fixed, `K=9` | 0.7465 | 0.5191 | 0.5399 |
| triadic fixed, `K=17` | 0.8153 | 0.7909 | 0.8033 |
| triadic fixed, `K=26` | 0.8573 | 0.8666 | 0.8738 |
| triadic fixed, `K=30` | 0.8685 | 0.8945 | 0.9169 |
| triadic learned, `K=9` | 0.6901 | **0.4141** | 0.4340 |
| triadic learned, `K=17` | 0.7918 | 0.8272 | 0.8465 |
| triadic learned, `K=26` | 0.8206 | 0.9129 | 0.9198 |
| triadic learned, `K=30` | 0.8328 | 0.9469 | 0.9573 |

The identity control confirms the association problem from first principles. Permuting which user's section-1 atom is
paired with which user's section-2 atom leaves every section count and hence the received sum unchanged. Without an outer
constraint, the observations do not identify complete messages even when local counts are known perfectly.

Triadic constraints genuinely help: at `K=9`, learned triadic local association improves high-SNR PUPE from the identity
control's `0.9453` to `0.4141`. They are not sufficient at higher occupancy. Every `J=4` section has only 16 atoms, so an
active batch has a local collision with probability essentially one at every tested load. Complete-message collisions,
by contrast, affect any batch only about `1.6%`, `2.8%`, `4.4%`, and `10%` of the time for `K=9,17,26,30`. The observed
failure is therefore local occupancy and cross-section association, not primarily repeated global messages.

Learning helps substantially at `K=9`, but generalises poorly at `K>=17`, where learned local association becomes worse
than the fixed bank. The encoder was optimised through local D0 rather than a complete-path objective, so this is
consistent with decoder co-adaptation and an objective that does not reward globally resolvable associations.

BP worsens high-SNR local-association PUPE in every fixed and learned cell, by `0.007--0.022`. Together with job `023`,
the current marginal BP route should be treated as unvalidated. A multiuser aggregate observation does not decompose into
one independent valid path; improving section marginals is not automatically the same as resolving the set of paths.

Job `026` is the decisive causal bridge. Job `025` says the refactor is innocent; the identity row proves that outer
structure is necessary; triadic rows show that the present structure helps but does not retain the dense/sparse-global
performance landscape. Because the induced-global comparator is D0 rather than exact MAP/ML, this is a practical
structured-encoder/decoder gap, not an impossibility result for the model class.

---

## 15. Current Defensible Claim

```text
The framework exactly represents the tested ODMA, reduced sectioned CCS, and
reduced CCS-AMP constructions. It supports implicit global-message decoding at
M=16384 and has a separately certified section-domain backend with no M axis.
Job 025 proves that backend is exactly behaviour-compatible at L=1; the Binomial
prior is practically neutral there. Job 024 proves M-free B=128 execution and
exact unit energy, but its decoder saturates and achieves PUPE 1.0, so it is not
a successful learning result. Job 026 proves that unstructured section counts do
not identify complete messages, and that triadic constraints help association at
low load but do not retain dense/sparse-global performance at higher local occupancy.
Current modular marginal BP does not improve that route. Job 027 independently shows
that explicit sparse-global codebooks retain the dense landscape over a broad density
range: D0 is near-flat through about s=48 and D1 is near dense at s=16 under its current
training budget, while equal-density four-mask ODMA is much worse. This identifies a
promising model class but not a scalable implementation, since the experiment still
stores and scores M=4096 columns. These are practical encoder-plus-decoder results,
not information-theoretic exclusions.
```

---

## 16. Sparse-Global Density Frontier (`027`)

### 16.1 Completion and validity audit

The manifest contains 72 rows: 16 sparse support sizes, dense and ODMA controls, two seeds, and D0/D1. All 72 rows now
contain both `summary.json` and `checkpoint.pt`, and the strict merger passes without completeness notes.

| Failed array indices | Runs | Failure point |
|---|---|---|
| `3,4` | sparse global `s=256`, seed 2702, D0/D1 | after all training and evaluation |
| `67,68` | dense, seed 2702, D0/D1 | after all training and evaluation |

The posthoc diagnostic defines numerical support by `abs(Phi)>0` and requires identical support size in every column.
PyTorch seed 2702 produced one exactly zero float32 Gaussian entry in the `256 x 4096` draw, so one column had support
255. The intended model is an exact-support continuous-Gaussian construction, where such a zero has probability zero.
Initialisation now resamples exact zeros, and the four rows were rerun successfully. The repair changed random-number
consumption for seed 2702, so the full-density rerun is not a clean nested paired endpoint; intermediate supports retain
the original nested comparison.

The complete artifacts pass the substantive audit:

- expected histories: 80 D0 epochs or 20 D1 epochs;
- exactly 20 learned and 20 matched-filter cells per summary;
- no nonfinite history, metric, or diagnostic values;
- a checkpoint for every summary;
- bit-identical encoder states and identical matched-filter streams in paired D0/D1 rows;
- exact nested support/sign inclusion across every returned support within both seeds;
- maximum unit-energy deviation `9.54e-7`;
- no unresolved traceback among the 72 final rows.

The tables below use the strict 72-summary aggregate. The full-density values are reported transparently but should not
be treated as nested paired endpoints because of the repair above.

### 16.2 Primary frontier

The primary outcome averages PUPE over 8 and 12 dB and `K=9,17,26,30`.

| Family/support | Density `p=s/256` | D0 PUPE | D1 PUPE | Replication |
|---|---:|---:|---:|---|
| dense | 1 | 0.2935 | 0.2030 | two summaries; repaired seed 2702 not nested-paired |
| sparse global `s=256` | 1 | 0.2976 | 0.2300 | two summaries; repaired seed 2702 not nested-paired |
| sparse global `s=192` | 0.75 | 0.2949 | 0.2150 | two summaries |
| sparse global `s=64` | 0.25 | 0.2932 | 0.2256 | two summaries |
| sparse global `s=48` | 0.1875 | 0.2975 | 0.2188 | two summaries |
| sparse global `s=32` | 0.125 | 0.3051 | 0.2276 | two summaries |
| sparse global `s=24` | 0.09375 | 0.3120 | 0.2399 | two summaries |
| sparse global `s=16` | 0.0625 | 0.3312 | 0.2323 | two summaries |
| sparse global `s=12` | 0.046875 | 0.3524 | 0.2413 | two summaries |
| sparse global `s=8` | 0.03125 | 0.3984 | 0.2813 | two summaries |
| sparse global `s=4` | 0.015625 | 0.5171 | 0.3994 | two summaries |
| sparse global `s=1` | 0.00390625 | 0.8542 | 0.8380 | two summaries |
| four-mask ODMA | 0.25 | 0.5363 | 0.4884 | two summaries |

Three conclusions survive the completed audit.

First, the no-loss region is broad. Relative to dense, D0 at `s=48` differs by only `+0.0040`, and `s=32` by `+0.0116`.
D1 at `s=16` differs by `+0.0293` in the strict aggregate. The D1 curve is noisier and its training budget is shorter;
its exact transition is therefore training-budget and replication dependent, not evidence that one support is optimal.

Second, equal density does not imply equal recovery. At `p=0.25`, random sparse global beats ODMA by `0.2431` PUPE for
D0 and `0.2628` for D1. Even `s=4`, which uses only 1.56% of rows per column, remains nominally better than ODMA by
`0.0192` and `0.0889`. This strongly extends the earlier `p=0.25` finding: the four reused ODMA placements lose useful
support diversity/geometry, not merely nonzero entries.

Third, D1 exploits sparse-global geometry more strongly. D0 improves on its matched-filter control by only about
`0.005--0.015` across the sweep. Around `s=4--16`, D1 improves on the same matched-filter observations by roughly
`0.11--0.13`. D1 is also approximately `4.96x` slower than D0 by median paired wall time and is trained for only 20
epochs versus 80. These runs compare the chosen trained systems, not decoder capacity at equal compute or convergence.

### 16.3 Discrete/geometric transition

Raw support capacity does not explain the performance transition. At `s=2`,
`log2 binom(256,2) > 12`, so enough unsigned masks exist in principle to label all 4,096 messages. Empirically, both
seeds have all 4,096 distinct support masks and signed patterns through `s=4`; repetition is still negligible at `s=3`.
Performance nevertheless begins degrading around `s=16--32`.

| Support `s` | sampled correlation q99.9 | `K=30` occupied-row fraction | row-load CV | signed-pattern duplicate fraction |
|---:|---:|---:|---:|---:|
| 256 | 0.206 | 1.000 | 0.000 | 0.000 |
| 64 | 0.219 | 1.000 | 0.027 | 0.000 |
| 32 | 0.256 | 0.982 | 0.042 | 0.000 |
| 16 | 0.348 | 0.856 | 0.061 | 0.000 |
| 8 | 0.479 | 0.613 | 0.085 | 0.000 |
| 4 | 0.677 | 0.377 | 0.118 | 0.000 |
| 2 | 0.904 | 0.210 | 0.171 | 0.015 |
| 1 | 1.000 | 0.111 | 0.247 | 0.875 |

As support shrinks, high-correlation column pairs become more severe and simultaneous users occupy a smaller part of the
resource space. Row imbalance grows but remains modest near the useful transition. At `s=1`, only 256 unsigned masks and
512 real signed directions exist, forcing extensive repeated directions. Thus a procedural support generator should be
judged by induced coherence/intersections, active-set geometry, row balance, and decoder searchability, not only by
`binom(n,s)`.

### 16.4 What the experiment does and does not buy

Every sparse-global checkpoint still stores `C` as a dense `256 x 4096` tensor; checkpoint size is about 4.35 MiB
regardless of `s`. Normal D0/D1 also maintain `M`-length scores. Job 027 therefore establishes existence of a forgiving
small-`B` sparse model class, not scalable storage or `B=100` decoding.

The next controlled sequence is job `028`: compare a procedural support family against iid sparse controls at
`B=14,n=256` using both recovery and geometry, while keeping the sectioned large-`B` calibration/association branch
separate.

---

## 17. Generated Hash-Skeleton Experiment (`028`)

### 17.1 Algebraic object

For support `T=s`, partition `n=TR` rows into `T` tables. Message `w in GF(2)^B` selects

```text
h_t(w) = A_t w + b_t mod 2,       A_t shape (r,B), b_t shape (r), R=2^r,
row_t(w) = tR + integer(h_t(w)).
```

Every `A_t` is full row rank for exact bin balance, and the vertically stacked `A_t` matrices have rank `B` so the full
hash tuple is injective. For two messages separated by XOR difference `d`, their support overlap is exactly the number
of tables satisfying `A_t d=0`. At `B=14`, that complete collision spectrum is enumerated and used to select the best
of 128 random full-rank candidates without gradient-relaxing the discrete hash.

If `P_t in {0,1}^{R x 2^B}` contains the one-hot table selections and `D_t` contains the fixed, normalised amplitudes,
the certification matrix is

```text
Phi = [P_0 D_0; P_1 D_1; ...; P_(T-1) D_(T-1)] in R^(n x 2^B).
```

This exact matrix lets the established global D0/D1 comparison remain unchanged. Only the support rule is compact;
amplitude generation and receiver-side candidate proposal remain exponential objects in this experiment.

### 17.2 Controlled manifest

The 36 rows fix `B=14,n=256`, two seeds, D0/D1, the job-`022` load/SNR/training contract, and two supports:
`T=16` with `(R,r)=(16,4)`, and `T=32` with `(R,r)=(8,3)`. At each support and seed, the sparse families share the
same Gaussian amplitude array before exact unit-energy normalisation.

| Comparison | Question |
|---|---|
| balanced random tables versus iid arbitrary sparse | Is one resource per predetermined table damaging? |
| random affine hash versus balanced random tables | Is repeated-XOR linear structure damaging? |
| selected affine hash versus random affine hash | Does offline collision-spectrum design help? |
| dense versus all rows | Context only; it is not density matched. |

Diagnostics add active-set Gram spectra and sampled disjoint `K`-sum distance to the job-`027` support, row-load,
occupancy, and pair-correlation audit. Geometry can explain a recovery change but cannot replace PUPE.

### 17.3 Current evidence boundary

Implementation checks pass for exact support/energy, table membership, GF(2) ranks, injectivity,
procedural/materialised equality, shared amplitude magnitudes, deterministic selection, and forward-matrix equality.
Both D0 and D1 complete the local smoke and actual `B=14` manifest path. A ten-run `B=8,n=64` mini bank covers every
family/decoder combination with finite training and evaluation. The mini runs are deliberately under-trained and are
not a model ranking.

No job-`028` HPC result is currently recorded. The first decision is whether balanced tables match iid sparse at the
same `T`; only if that restriction survives should linear-hash choice, procedural amplitudes, and scalable inversion be
developed further.

---

## 18. Joint Encoder/Decoder Learning (`029`)

The sparse-frontier and hash-certification jobs fix every codeword and train only D0/D1. Job `029` tests whether that
understates the model class. At `B=14,n=256`, it jointly trains the selected decoder and either all dense amplitudes or
the amplitudes on fixed iid/selected-hash supports at `T=16,32`. Gradients outside sparse support are zeroed and every
column is projected to unit energy after each step.

The 20 rows use two seeds and both decoders. Each receives 120 epochs, compared with 80 for D0 and 20 for D1 in job
`028`. Initial/final geometry and complete loss curves are saved. The primary analysis is fixed versus joint performance,
then selected hash versus iid at matched support. This remains a small-`B` model-class test because learned amplitudes
and the global decoder still have a `2^B` axis. Local invariant, smoke, and six-row mini tests pass; no HPC result is yet
recorded.
