"""Receiver bank. Every entry returns (hard full-message counts, metadata).

The thirteen original families and Job-036 combinations use an explicit alphabet.
Variants and budgets are recorded; statistical exactness is not inferred from a family name.
Report-11 entries also accept a procedural PolarScenario and return sparse count dictionaries.
"""

from dataclasses import dataclass
import itertools
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom
import torch

from baselines import message_bits, message_indices


from framework.polar_decoder_core import Scenario, _guide, allocate, count_tails


def hypothesis_count(m, k, sampling):
    return math.comb(m, k) if sampling == "distinct" else math.comb(m + k - 1, k)


def prepare_reference(phi, k, sampling, max_states=200_000, max_bytes=512_000_000):
    m, n = phi.shape[1], phi.shape[0]
    size = hypothesis_count(m, k, sampling)
    estimated_bytes = size * (16 * k + 8 * n + 32) + 8 * m * k
    if size > max_states or estimated_bytes > max_bytes:
        raise ValueError(f"REF needs {size:,} count states (~{estimated_bytes:,} bytes); "
                         "raise its explicit budget or omit REF for this sweep")
    choose = itertools.combinations if sampling == "distinct" else itertools.combinations_with_replacement
    tuples = np.fromiter((w for words in choose(range(m), k) for w in words), dtype=np.int64, count=size * k).reshape(size, k)
    signals = np.zeros((size, n))
    for j in range(k):
        signals += phi[:, tuples[:, j]].T
    if sampling == "distinct":
        log_prior = np.full(size, -math.log(size))
    else:
        log_prior = np.full(size, math.lgamma(k + 1) - k * math.log(m))
        run = np.ones(size, dtype=np.int64)
        for j in range(1, k):
            run = np.where(tuples[:, j] == tuples[:, j - 1], run + 1, 1)
            log_prior -= np.log(run)
    return {"tuples": tuples, "signals": signals, "log_prior": log_prior, "phi": phi,
            "k": k, "sampling": sampling, "estimated_bytes": estimated_bytes}


def reference(scenario, *, prepared=None, max_states=200_000, max_bytes=512_000_000):
    s = scenario
    bank = prepared if prepared is not None else prepare_reference(s.phi, s.k, s.sampling, max_states, max_bytes)
    if bank["phi"] is not s.phi or bank["k"] != s.k or bank["sampling"] != s.sampling:
        raise ValueError("Reference bank does not match this scenario")
    log_weight = bank["log_prior"] - np.square(bank["signals"] - s.y).sum(1) / (2 * s.noise_var)
    log_z = float(logsumexp(log_weight))
    tails = count_tails(bank["tuples"], np.exp(log_weight - log_z), s.phi.shape[1])
    counts = allocate(tails, s.k, s.sampling)
    gain = sum(tails[w, :c].sum() for w, c in enumerate(counts) if c)
    return counts, {"tails": tails, "log_evidence": log_z, "posterior_risk": float(1 - gain / s.k),
                    "states": len(log_weight), "excess_risk_bound": 0.0, "target": "exact_count_posterior",
                    "certificate": "exact algebra in floating point; not an interval-arithmetic certificate"}


def product(scenario, *, max_iterations=100, tolerance=1e-9):
    s = scenario
    if s.sampling != "iid":
        raise ValueError("DEC-02's initial identical-slot product implementation requires IID messages")
    m = s.phi.shape[1]

    def objective(theta):
        score = theta @ s.phi
        log_z = logsumexp(score)
        probability = np.exp(score - log_z)
        mean = s.phi @ probability
        value = s.noise_var * (theta @ theta) / 2 + (s.k - 1) * (log_z - math.log(m)) - s.y @ theta
        return value, s.noise_var * theta + (s.k - 1) * mean - s.y

    if s.k == 1:
        theta, iterations, converged = s.y / s.noise_var, 0, True
    else:
        result = minimize(objective, s.y / (s.noise_var + (s.k - 1) / len(s.y)), jac=True, method="L-BFGS-B",
                          options={"maxiter": max_iterations, "gtol": tolerance, "ftol": tolerance})
        theta, iterations, converged = result.x, int(result.nit), bool(result.success)
    score = theta @ s.phi
    log_probability = score - logsumexp(score)
    probability = np.exp(log_probability)
    tails = binom.sf(np.arange(s.k)[None, :], s.k, probability[:, None])
    mean = s.phi @ probability
    elbo = -(np.square(s.y - s.k * mean).sum() + s.k * (1 - mean @ mean)) / (2 * s.noise_var)
    elbo -= s.k * np.dot(probability, log_probability + math.log(m))
    if not np.isfinite(elbo) or not np.isfinite(tails).all():
        raise FloatingPointError("Nonfinite product-law inference")
    return allocate(tails, s.k), {"tails": tails, "elbo": float(elbo), "iterations": iterations,
                                 "optimizer_converged": converged, "gradient_norm": float(np.linalg.norm(objective(theta)[1])),
                                 "excess_risk_bound": min(1.0, 2 * math.sqrt(max(0.0, -elbo) / 2)),
                                 "target": "identical_slot_variational_product", "readout": "explicit_all_word_binomial_tails",
                                 "certificate": "Pinsker using log Z<=0; floating point"}



def guided_log_density(s, tuples):
    residual = np.repeat(s.y[None], len(tuples), axis=0)
    density = np.zeros(len(tuples))
    mean = s.phi.mean(1)
    for j in range(s.k):
        logp = _guide(s, residual, tuples[:, :j], s.k - j, mean)
        density += logp[np.arange(len(tuples)), tuples[:, j]]
        residual -= s.phi[:, tuples[:, j]].T
    return density


def importance(scenario, *, samples=1024, defensive=0.1, seed=0, batch_size=256, failure_probability=0.05):
    s, m = scenario, scenario.phi.shape[1]
    if samples < 1 or batch_size < 1 or not 0 < defensive <= 1 or not 0 < failure_probability < 1:
        raise ValueError("Invalid importance budget, mixture or failure probability")
    rng = np.random.default_rng(seed)
    mean = s.phi.mean(1)
    all_words, all_log_weights = [], []
    log_prior = -s.k * math.log(m) if s.sampling == "iid" else -sum(math.log(m - j) for j in range(s.k))
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        prior_component = rng.random(size) < defensive
        words = np.empty((size, s.k), dtype=np.int64)
        residual = np.repeat(s.y[None], size, axis=0)
        log_guide = np.zeros(size)
        for j in range(s.k):
            logp = _guide(s, residual, words[:, :j], s.k - j, mean)
            draw_p = np.exp(logp)
            prior_p = np.full((size, m), 1.0)
            if s.sampling == "distinct" and j:
                prior_p[np.arange(size)[:, None], words[:, :j]] = 0
            prior_p /= prior_p.sum(1, keepdims=True)
            draw_p[prior_component] = prior_p[prior_component]
            cdf = np.cumsum(draw_p, axis=1)
            cdf[:, -1] = 1.0
            words[:, j] = np.sum(cdf < rng.random((size, 1)), axis=1)
            log_guide += logp[np.arange(size), words[:, j]]
            residual -= s.phi[:, words[:, j]].T
        log_q = np.full(size, log_prior) if defensive == 1 else np.logaddexp(
            math.log(defensive) + log_prior, math.log1p(-defensive) + log_guide)
        all_words.append(words)
        all_log_weights.append(log_prior - np.square(residual).sum(1) / (2 * s.noise_var) - log_q)
    tuples, log_weights = np.concatenate(all_words), np.concatenate(all_log_weights)
    normalized = np.exp(log_weights - logsumexp(log_weights))
    tails = count_tails(tuples, normalized, m)
    log_mean_weight = float(logsumexp(log_weights) - math.log(samples))
    log_b = -math.log(defensive) + .5 * math.log(math.log(2 * (m * s.k + 1) / failure_probability) / (2 * samples))
    relative = math.exp(log_b - log_mean_weight) if log_b < log_mean_weight else 1.0
    error = min(1.0, 2 * relative / (1 - relative)) if relative < 1 else 1.0
    return allocate(tails, s.k, s.sampling), {
        "tails": tails, "samples": samples, "defensive": defensive, "seed": int(seed),
        "effective_sample_size": float(1 / np.square(normalized).sum()), "unique_tuples": len(np.unique(tuples, axis=0)),
        "unique_words": len(np.unique(tuples)), "log_mean_weight": log_mean_weight,
        "uniform_tail_error_bound": error, "excess_risk_bound": min(1.0, 2 * error),
        "failure_probability": failure_probability, "target": "exact_posterior_importance_corrected",
        "guide": "sequential_isotropic_residual", "readout": "explicit_MK_tail_table",
        "certificate": "fixed-sample Hoeffding across all MK tails; ESS is diagnostic only"}


def d2(scenario, *, encoder, decoder):
    s = scenario
    with torch.no_grad():
        y = torch.as_tensor(s.y.copy(), dtype=encoder.dtype)[None, :, None]
        result = decoder(encoder, y, torch.ones((1, 1), dtype=encoder.dtype), s.k, noise_var=s.noise_var)
    return result.counts[0].cpu().numpy().astype(np.int64), {
        "layers": decoder.num_layers, "target": "historical_D2_Bernoulli_effective_channel",
        "learned_parameters": sum(p.numel() for p in decoder.parameters()), "excess_risk_bound": None}


def odma(scenario, *, baseline, amplitudes):
    s = scenario
    if np.allclose(amplitudes, 1 / math.sqrt(baseline.code_length), rtol=0, atol=1e-14):
        messages, meta = baseline.decode(s.y, s.k, s.noise_var)
        counts = np.bincount(message_indices(messages), minlength=s.phi.shape[1])
        return counts, {**meta, "variant": "unchanged_repository_paper_adaptation", "excess_risk_bound": None}
    # Keep the published-method chain, replacing each symbol's amplitude in LLR/cancellation.
    residual = s.y.copy()
    decoded, seen, trials, trace = [], set(), 0, []
    for iteration in range(baseline.max_iterations):
        remaining = s.k - len(decoded)
        if remaining == 0:
            break
        score = np.abs(residual[baseline.positions]).sum(1)
        candidates = np.argsort(-score, kind="stable")[:min(baseline.num_patterns, remaining + baseline.detector_slack)]
        accepted = []
        for p in candidates:
            unresolved = remaining - (len(accepted) if baseline.sic_mode == "successive" else 0)
            variance = s.noise_var + max(0, unresolved - 1) / baseline.n
            payloads, _ = baseline.polar.decode_list(2 * amplitudes[p] * residual[baseline.positions[p]] / variance)
            trials += 1
            if not len(payloads):
                continue
            bits = np.concatenate((message_bits(np.array([p]), baseline.prefix_bits)[0], payloads[0]))
            word = int(message_indices(bits[None])[0])
            if word in seen:
                continue
            seen.add(word)
            accepted.append(word)
            if baseline.sic_mode == "successive":
                residual -= s.phi[:, word]
            if len(decoded) + len(accepted) >= s.k:
                break
        if baseline.sic_mode == "parallel" and accepted:
            residual -= s.phi[:, accepted].sum(1)
        decoded.extend(accepted)
        trace.append({"iteration": iteration + 1, "new_messages": len(accepted)})
        if not accepted:
            break
    return np.bincount(decoded, minlength=s.phi.shape[1]), {
        "variant": "balanced_energy_LLR_and_cancellation_adaptation", "polar_trials": trials, "trace": trace,
        "excess_risk_bound": None, "caveat": "adapted receiver, not the paper's uniform-energy system"}


@dataclass(frozen=True)
class DecoderSpec:
    label: str
    run: object = None
    sources: tuple = ("iid", "distinct")
    defaults: tuple = ()
    stochastic: bool = False


from framework.polar_decoder_queries import fields, fourier, gaussian_counts, prefix
from framework.polar_decoder_search import chains, configuration, particles
from framework.polar_decoder_contractions import connected, quantized, regions, signed_rank
from framework.polar_decoder_combinations import combined_importance, combined_particles, multiset_beam
from framework.polar_tractable import PolarScenario, tractable_beam, tractable_chain, tractable_importance

REGISTRY = {
    "REF": DecoderSpec("REF: exact count Bayes", reference),
    "DEC-01": DecoderSpec("DEC-01: residual Gaussian", gaussian_counts),
    "DEC-02": DecoderSpec("DEC-02: product law", product, ("iid",), (("max_iterations", 100), ("tolerance", 1e-9))),
    "DEC-03": DecoderSpec("DEC-03: product prefix", prefix, ("iid",),
                          (("splits", 128), ("tolerance", 1e-8), ("max_iterations", 100))),
    "DEC-04": DecoderSpec("DEC-04: bounded beam", configuration, defaults=(("width", 32),)),
    "DEC-05": DecoderSpec("DEC-05: guided importance", importance, defaults=(("samples", 1024), ("defensive", .1),
                                                                        ("batch_size", 256), ("failure_probability", .05)), stochastic=True),
    "DEC-06": DecoderSpec("DEC-06: corrected chain", chains, defaults=(("samples", 128), ("burn_in", 128), ("thinning", 1),
                          ("chains", 4), ("global_probability", .1), ("defensive", .1), ("block_size", 1)), stochastic=True),
    "DEC-07": DecoderSpec("DEC-07: particle batches", particles, defaults=(("batches", 4), ("particles", 64),
                          ("defensive", .1), ("guide_log_floor", -80.), ("failure_probability", .05)), stochastic=True),
    "DEC-08": DecoderSpec("DEC-08: quantized states", quantized, defaults=(("candidates", 32), ("representatives", 4), ("spacing", .25))),
    "DEC-09": DecoderSpec("DEC-09: regional counts", regions, defaults=(("region_size", 1), ("max_local_states", 20_000))),
    "DEC-10": DecoderSpec("DEC-10: connected pairs", connected, ("iid",),
                          (("candidates", 32), ("order", 2), ("max_block_tuples", 300_000))),
    "DEC-11": DecoderSpec("DEC-11: Fourier intervals", fourier, ("iid",),
                          (("frequencies", 512), ("batch_size", 64), ("failure_probability", .05)), stochastic=True),
    "DEC-12": DecoderSpec("DEC-12: signed cut SVD", signed_rank, defaults=(("candidates", 32), ("width", 32), ("rank", 8))),
    "DEC-13": DecoderSpec("DEC-13: joint fields", fields, ("iid",), (("samples", 512), ("field_rank", 2),
                          ("defensive", .1), ("failure_probability", .05)), stochastic=True),
    "D2": DecoderSpec("D2: trained, fixed encoder", d2),
    "ODMA": DecoderSpec("ODMA: CRC-SCL/SIC", odma),
}


REGISTRY["DEC-14"] = DecoderSpec("DEC-14: multiset beam", multiset_beam, ("iid",), (("width", 32),))
_IS_COMBINATION = {"samples": 1024, "defensive": .1, "product_weight": .2, "beam_weight": .2, "width": 32,
                   "max_iterations": 100, "batch_size": 128, "moves": 0, "block_size": 1}
for number, label, overrides in [
    (15, "guide + product IS", {"product_weight": .3, "beam_weight": 0.}),
    (16, "guide + beam IS", {"product_weight": 0., "beam_weight": .3}),
    (17, "guide + product + beam IS", {}),
    (18, "mixture IS + Gibbs", {"moves": 8}),
    (19, "mixture IS + guided pair MH", {"moves": 8, "block_size": 2}),
]:
    REGISTRY[f"DEC-{number:02}"] = DecoderSpec(f"DEC-{number:02}: {label}", combined_importance, ("iid",),
                                             tuple({**_IS_COMBINATION, **overrides}.items()), True)
_SMC_COMBINATION = {"batches": 4, "particles": 64, "defensive": .1, "product_weight": .25, "max_iterations": 100,
                    "guide_log_floor": -1000., "moves": 0, "block_size": 1, "batch_size": 128}
for number, label, overrides in [
    (20, "product-guided SMC", {}),
    (21, "Gaussian SMC + Gibbs", {"product_weight": 0., "moves": 8}),
    (22, "product SMC + guided pair MH", {"moves": 8, "block_size": 2}),
    (23, "product SMC + Gibbs", {"moves": 8}),
    (24, "Gaussian SMC + guided pair MH", {"product_weight": 0., "moves": 8, "block_size": 2}),
]:
    REGISTRY[f"DEC-{number:02}"] = DecoderSpec(f"DEC-{number:02}: {label}", combined_particles, ("iid",),
                                             tuple({**_SMC_COMBINATION, **overrides}.items()), True)


_TRACTABLE_BEAM = {"width": 16, "patterns": 8, "list_size": 16, "flip_order": 1, "max_refine": 32}
for number, label, method in [(25, "SCL multiset beam", "scl"), (26, "basis-flip multiset beam", "basis")]:
    REGISTRY[f"DEC-{number}"] = DecoderSpec(f"DEC-{number}: {label}", tractable_beam, ("iid",),
                                          tuple({**_TRACTABLE_BEAM, "candidate_method": method}.items()), True)
_TRACTABLE_CHAIN = {"samples": 64, "burn_in": 64, "thinning": 1, "patterns": 8, "list_size": 16,
                    "init_width": 4, "defensive": .1, "flip_order": 1, "max_refine": 32, "cache_centers": 4,
                    "init_method": "scl"}
for number, label, method in [
    (27, "independent-parity MH", "parity"), (28, "repaired-parity MH", "refined"),
    (29, "cached repaired-parity MH", "cached"), (30, "joint-pair polar MH", "joint"),
    (31, "XOR-pair polar MH", "xor"), (32, "SCL-list MH", "scl"),
]:
    REGISTRY[f"DEC-{number}"] = DecoderSpec(f"DEC-{number}: {label}", tractable_chain, ("iid",),
                                          tuple({**_TRACTABLE_CHAIN, "proposal_method": method}.items()), True)

REGISTRY["DEC-33"] = DecoderSpec("DEC-33: exchangeable parity + beam IS", tractable_importance, ("iid",),
                                (("samples", 256), ("width", 32), ("patterns", 8), ("list_size", 16),
                                 ("flip_order", 1), ("max_refine", 32), ("beam_weight", .5)), True)
REGISTRY["DEC-34"] = DecoderSpec("DEC-34: union-discovery multiset beam", tractable_beam, ("iid",),
                                tuple({**_TRACTABLE_BEAM, "candidate_method": "union"}.items()), True)


def run(name, scenario, **params):
    spec = REGISTRY[name]
    if spec.run is None:
        raise NotImplementedError(f"{name} is catalogued in Report 10 but not implemented")
    if scenario.sampling not in spec.sources:
        raise ValueError(f"{name} supports {spec.sources}, not {scenario.sampling}")
    procedural = isinstance(scenario, PolarScenario)
    if procedural and spec.run not in (tractable_beam, tractable_chain, tractable_importance):
        raise ValueError(f"{name} requires the explicit small-alphabet Scenario")
    counts, meta = spec.run(scenario, **{**dict(spec.defaults), **params})
    if isinstance(counts, dict):
        alphabet = 1 << scenario.model.B if procedural else scenario.phi.shape[1]
        if any(not isinstance(w, int) or not 0 <= w < alphabet or not isinstance(c, int) or c <= 0
               for w, c in counts.items()) or sum(counts.values()) != scenario.k:
            raise ValueError(f"{name} violated the sparse count/output-budget contract")
        meta["legacy_dense_adapter"] = not procedural
        if procedural:
            return counts, meta
        dense = np.zeros(alphabet, dtype=np.int64)
        for w, c in counts.items():
            dense[w] = c
        counts = dense
    counts = np.asarray(counts)
    if counts.shape != (scenario.phi.shape[1],) or not np.isfinite(counts).all():
        raise ValueError(f"{name} returned invalid full-message counts")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)) or counts.sum() > scenario.k:
        raise ValueError(f"{name} violated the hard-count/output-budget contract")
    if name not in ("D2", "ODMA") and counts.sum() != scenario.k:
        raise ValueError(f"{name} must allocate K copies")
    if scenario.sampling == "distinct" and np.any(counts > 1):
        raise ValueError(f"{name} returned duplicate words under the distinct-source control")
    return counts.astype(np.int64), meta
