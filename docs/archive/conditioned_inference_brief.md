# Research Brief: Conditioned Bayesian Unfolding

Historical proposal, archived 7 October 2026. This is not an agreed paper plan or implemented receiver. The useful
noise/interference/calibration distinction is retained in [CURRENT_STATE.md](../CURRENT_STATE.md); its broader theory
and application ideas remain here for reference. The current URA direction does not commit to this architecture.

## Aim

Assess a combined theory-and-method paper on an unrolled sparse/count decoder that adapts to the prior, load, physical noise and known codebook while keeping three quantities distinct:

1. physical channel noise;
2. effective uncertainty within an iteration;
3. decision calibration or "sharpness".

The intended structured applications are URA and learned digital over-the-air computation. The codebook is fixed and known during decoding, although it may have been learned beforehand.

## Starting point

Let

\[
y=Cx+z,\qquad z\sim\mathcal N(0,\sigma_{\rm phys}^2I),
\]

where \(C\) is generally nonorthogonal and \(x\) is sparse binary or nonnegative integer-valued. A gradient-style layer gives

\[
u^t=\hat x^t+\eta_tC^H(y-C\hat x^t),
\]

and, with \(G=C^HC\),

\[
u^t-x=(I-\eta_tG)(\hat x^t-x)+\eta_tC^Hz.
\]

The denoiser therefore sees physical noise plus current reconstruction error transformed by the codebook. Under a diagonal approximation,

\[
\tau_{i,t}\approx\eta_t^2\sigma_{\rm phys}^2G_{ii}
+\sum_j|\delta_{ij}-\eta_tG_{ij}|^2v_{j,t}.
\]

The research hypothesis is that this effective uncertainty should be derived from known physics and current state, with at most a bounded learned correction, rather than represented by an unconstrained learned "SNR".

## Prior-matched denoising

For a pseudo-observation \(U=X+\sqrt\tau Z\), use the actual prior:

\[
q_{i,t}(c)\propto p_i(c\mid\text{load/context})
\exp\!\left[-\frac{(u_{i,t}-c)^2}{2\tau_{i,t}}\right].
\]

Relevant priors include Bernoulli activity, exact Binomial counts and Poisson count approximations. The project does not assume that one \(\ell_q\) penalty can represent arbitrary prior behaviour.

For the Bayes posterior-mean denoiser,

\[
\eta_\tau'(u)=\frac{\operatorname{Var}(X\mid U=u)}{\tau}.
\]

This established identity connects posterior uncertainty and local denoiser response. Novelty must come from new theory or its use in coupled, conditioned decoding, not from the identity alone.

## Candidate contribution

Develop a low-parameter unroller conditioned on

\[
(\sigma_{\rm phys}^2,\ \text{load/prior},\ \text{operator geometry},\ \text{residual state},\ t)
\]

that produces prior-matched posterior means, variances and calibrated evidence. Possible contributions are:

- an identifiable separation of physical noise, effective uncertainty and decision calibration;
- a response-based treatment covering binary and count priors without reducing them to \(\ell_q\);
- scalable inclusion of nonorthogonal procedural-codebook geometry;
- transfer to unseen joint combinations of load, SNR, prior and codebook;
- theory connecting variance/response mismatch to support error, calibration or PUPE.

## Novelty boundary

AMP/VAMP effective channels, prior-matched denoisers, TISTA variance estimation, learned AMP/VAMP, EM noise/prior estimation, dictionary-conditioned LISTA and learned Bayes denoisers are established. Learning a threshold from SNR/load or inserting an arbitrary-prior posterior mean is therefore insufficient by itself.

The main questions are:

1. When is scalar effective variance sufficient for nonorthogonal \(C\)?
2. Is scalar sharpness insufficient for priors with different threshold, transition and tail behaviour?
3. Can physical noise and algorithmic uncertainty be made identifiable?
4. Can mismatch be bounded in excess support error, MSE or PUPE?
5. Can the required geometry be computed scalably for procedural/hash codebooks?
6. Does conditioning improve held-out transfer rather than only interpolation?

## Minimum evidence

The paper should contain at least one theorem, bound or explicit counterexample; matched comparisons with adaptive unrolled baselines; held-out combinations of SNR, load, prior, matrix geometry and dimension; ablations separating each conditioning variable; and MSE, false-alarm/miss, calibration, PUPE, runtime and parameter-count results. Simple compressed-sensing cases should isolate mechanisms, with URA or FEEL as the structured application.

Essential starting references: [LAMP/LVAMP](https://arxiv.org/abs/1612.01183), [TISTA](https://arxiv.org/abs/1801.01978), [EM-GM-AMP](https://arxiv.org/abs/1207.3107), [Ada-LISTA](https://arxiv.org/abs/2001.08456), and [learned Bayes inference by unrolling](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f40b522ef32bbc050d37a2a1ab7ffc2b-Abstract-Conference.html).

The combined paper is plausible if it yields substantive theory or convincing out-of-distribution calibration and transfer beyond these methods. In-distribution gains from a conditioned denoiser alone would be useful engineering but not the intended contribution.
