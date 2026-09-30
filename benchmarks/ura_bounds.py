"""Finite-blocklength references for the real Gaussian MAC.

Physical Eb/N0 = 1/(2 B sigma^2) for nominal unit codeword energy.
The survey's Gaussian achievability reference is Polyanskiy's Theorem 1, not
another independent bound. The default evaluates p_t for every t and q_1,
following the numerical recipe in the original paper's Section III. The q_1
CDF integrates the correlated user minimum, not a single-user normal proxy.
Higher q_t terms are not evaluated. These are achievable-error upper bounds,
not converse floors. Numerical quadrature is checked separately by refinement.

The historical Gallager-only function remains as a regression/control curve.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, ndtr, roots_legendre
from scipy.stats import chi2, ncx2

from src.ura_bound import _E_of_t


def collision_rates(payload_bits, num_active):
    m, k = float(2 ** payload_bits), int(num_active)
    if k < 1 or payload_bits < 1:
        raise ValueError("B and K must be positive")
    per_user = -math.expm1((k - 1) * math.log1p(-1 / m))
    any_frame = 1.0 if k > m else -math.expm1(sum(math.log1p(-i / m) for i in range(k)))
    return {"per_user_duplicate_probability": per_user, "any_duplicate_probability": any_frame,
            "any_duplicate_union_bound": min(1.0, k * (k - 1) / (2 * m))}


def polyanskiy_gallager(payload_bits, n, num_active, ebn0_db, *, grid=31, power_grid=31, distinct=False, details=False):
    """Thm.1 p_t branch + clipping + original collision union bound (iid case).

    The result upper-bounds the paper's collision-as-error metric, hence also
    duplicate-tolerant PUPE. For uniformly sampled distinct messages, the same
    change-of-measure proof omits the collision term. It applies to norm<=1
    codewords, not only exact-norm codebooks. A decoder can beat this curve:
    epsilon_opt <= epsilon_achievable_bound, not epsilon_decoder >= bound.
    ``details`` exposes the minimizing grid point and each additive term.
    """
    b, k = int(payload_bits), int(num_active)
    if min(b, n, k) < 1 or k >= 2 ** b or grid < 2 or power_grid < 2 or not math.isfinite(ebn0_db):
        raise ValueError("Require positive B,n,K; K<2^B; grid sizes>=2; finite Eb/N0")
    power = 2 * b * 10.0 ** (float(ebn0_db) / 10) / n
    t = np.arange(1, k + 1)
    r1 = b * math.log(2) / n - gammaln(t + 1) / (n * t)
    r2 = (gammaln(k + 1) - gammaln(t + 1) - gammaln(k - t + 1)) / n
    collision = 0.0 if distinct else k * (k - 1) / (2 * float(2 ** b))
    best = {"unclipped_upper_bound": math.inf}
    for fraction in np.linspace(0.05, 0.99, power_grid):
        exponent = np.maximum(0.0, _E_of_t(fraction * power, t, n, r1, r2, grid))
        coding = np.sum(t / k * np.exp(-n * exponent))
        clipping = k * chi2.sf(n / fraction, n)
        total = float(coding + clipping + collision)
        if total < best["unclipped_upper_bound"]:
            best = {"unclipped_upper_bound": total, "coding_term": float(coding), "clipping_term": float(clipping),
                    "collision_union_term": collision, "auxiliary_power_fraction": float(fraction)}
    best["upper_bound"] = min(1.0, best["unclipped_upper_bound"])
    return best if details else best["upper_bound"]


@lru_cache(maxsize=32)
def _noise_quadrature(n, order):
    nodes, weights = roots_legendre(order)
    return chi2.ppf((nodes + 1) / 2, n), weights / 2


def minimum_information_cdf(gamma, auxiliary_power, n, num_active, *, order=96):
    """CDF of I_1=min_j i(X_j;X_j+Z), in nats, for Gaussian random codewords.

    Given S=||Z||^2, user information densities are independent and
    ||X_j+Z||^2/P' ~ noncentral-chi2(n,S/P'). Integrate 1-(1-F_S)^K over
    S~chi2(n). Integrating each user first and then taking the Kth power would
    incorrectly discard their shared-noise dependence.
    """
    p = float(auxiliary_power)
    if p <= 0 or not math.isfinite(p) or n < 1 or num_active < 1 or order < 8:
        raise ValueError("Require finite P'>0, positive n,K and quadrature order>=8")
    noise, weights = _noise_quadrature(n, order)
    if num_active == 1:
        # The marginal quadratic form also has the exact normal-mixture law
        # I=nC+sqrt(P'/(1+P'))*sqrt(chi2_n)*N(0,1). Avoid huge noncentralities.
        centered = np.asarray(gamma)[..., None] - n / 2 * math.log1p(p)
        return ndtr(centered / np.sqrt(p / (1 + p) * noise)) @ weights
    threshold = (1 + p) / p * (2 * (np.asarray(gamma)[..., None] - n / 2 * math.log1p(p)) + noise)
    cdf = ncx2.cdf(np.maximum(threshold, 0), n, noise / p)
    with np.errstate(divide="ignore"):
        conditional_minimum = -np.expm1(num_active * np.log1p(-cdf))
    return conditional_minimum @ weights


def q1_bound(payload_bits, n, num_active, auxiliary_power, *, order=96):
    """Numerically minimize F_{I_1}(gamma)+M*K*exp(-gamma); retain the trivial bound 1."""
    log_mk = payload_bits * math.log(2) + math.log(num_active)
    mean = n / 2 * math.log1p(auxiliary_power)
    std = math.sqrt(n * auxiliary_power / (1 + auxiliary_power))

    def objective(gamma):
        return minimum_information_cdf(gamma, auxiliary_power, n, num_active, order=order) + np.exp(log_mk - gamma)

    # The far-right tail approaches 1 and need not be unimodal: bracket the best
    # grid point before scalar refinement, rather than optimizing across a plateau.
    thresholds = np.linspace(log_mk, max(log_mk + 20, mean + 8 * std), 25)
    values = objective(thresholds)
    index = int(np.argmin(values))
    result = {"value": min(1.0, float(values[index])), "threshold_nats": float(thresholds[index])}
    if result["value"] < 1:
        lo, hi = thresholds[max(0, index - 1)], thresholds[min(len(thresholds) - 1, index + 1)]
        optimum = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
        if optimum.fun < result["value"]:
            result = {"value": float(optimum.fun), "threshold_nats": float(optimum.x)}
    return result


def polyanskiy_achievability(payload_bits, n, num_active, ebn0_db, *, grid=31, power_grid=31,
                            distinct=False, order=96, details=False):
    """Theorem 1 p_t plus q_1: the paper's numerical recipe, not every q_t.

    Power/exponent grids restrict optimization conservatively; q_1 uses exact
    conditional distributions with numerical quadrature, not a normal or Monte
    Carlo approximation. Retain the Gallager result as a no-worsening control.
    """
    if order < 8:
        raise ValueError("Quadrature order must be >=8")
    old = polyanskiy_gallager(payload_bits, n, num_active, ebn0_db, grid=grid, power_grid=power_grid,
                             distinct=distinct, details=True)
    b, k = int(payload_bits), int(num_active)
    power = 2 * b * 10.0 ** (float(ebn0_db) / 10) / n
    ts = np.arange(1, k + 1)
    r1 = b * math.log(2) / n - gammaln(ts + 1) / (n * ts)
    r2 = (gammaln(k + 1) - gammaln(ts + 1) - gammaln(k - ts + 1)) / n
    best = {**old, "q1_used": False, "q1": None, "q1_threshold_nats": None,
            "gallager_only_upper_bound": old["upper_bound"], "quadrature_order": order}
    alternatives = []
    for fraction in np.linspace(0.05, 0.99, power_grid):
        pt = np.exp(-n * np.maximum(0, _E_of_t(fraction * power, ts, n, r1, r2, grid)))
        clipping = k * chi2.sf(n / fraction, n)
        rest = float(np.sum(ts[1:] / k * pt[1:]))
        lower = rest + clipping + old["collision_union_term"]
        alternatives.append((lower, fraction, pt[0], rest, clipping))
    for lower, fraction, p1, rest, clipping in sorted(alternatives):
        if lower >= min(1.0, best["unclipped_upper_bound"]):
            continue  # Even q_1=0 cannot improve the displayed bound at this power.
        q1 = q1_bound(b, n, k, fraction * power, order=order)
        if q1["value"] >= p1:
            continue
        coding = rest + min(p1, q1["value"]) / k
        total = coding + clipping + old["collision_union_term"]
        if total < best["unclipped_upper_bound"]:
            best.update(unclipped_upper_bound=float(total), coding_term=float(coding), clipping_term=float(clipping),
                        auxiliary_power_fraction=float(fraction), q1_used=bool(q1["value"] < p1),
                        q1=q1["value"], q1_threshold_nats=q1["threshold_nats"])
    best["upper_bound"] = min(1.0, best["unclipped_upper_bound"])
    return best if details else best["upper_bound"]


def required_ebn0(payload_bits, n, num_active, target=0.05, *, distinct=False, bracket=(-8.0, 20.0), **params):
    """Energy certified by the p_t+q_1 bound; not the minimum physically achievable energy."""
    if not 0 < target < 1 or bracket[0] >= bracket[1]:
        raise ValueError("Require target in (0,1) and an increasing SNR bracket")
    # This is a limitation of the collision correction, not a PUPE floor for our decoder.
    collision = 0 if distinct else num_active * (num_active - 1) / (2 * float(2 ** payload_bits))
    if collision >= target:
        return {"ebn0_db": None, "status": "collision_correction_exceeds_target", "collision_union_term": collision}

    def gap(snr):
        return polyanskiy_achievability(payload_bits, n, num_active, snr, distinct=distinct, **params) - target

    if gap(bracket[0]) <= 0:
        return {"ebn0_db": None, "status": "below_search_bracket", "bracket_db": list(bracket)}
    if gap(bracket[1]) > 0:
        return {"ebn0_db": None, "status": "above_search_bracket", "bracket_db": list(bracket)}
    root = brentq(gap, *bracket, xtol=0.002)
    return {"ebn0_db": float(root), "status": "crossing", "numerical_root_tolerance_db": 0.002}


def reference_curves(payload_bits, n, num_active, ebn0_db, *, distinct=False):
    ebn0_db = list(ebn0_db)
    components = [polyanskiy_achievability(payload_bits, n, num_active, x, distinct=distinct, details=True) for x in ebn0_db]
    return {"ebn0_db": list(ebn0_db),
            "polyanskiy_achievability": [row["upper_bound"] for row in components],
            "polyanskiy_gallager_achievability": [row["gallager_only_upper_bound"] for row in components],
            "achievability_components": components,
            "source": "https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf",
            "achievability_label": "Polyanskiy achievable-error upper bound (p_t + q_1; not a converse)",
            "achievability_direction": "Optimal PUPE <= this value; practical decoders may perform below this curve",
            "not_evaluated": "q_t for t>=2; the original numerical recipe also used q_t only for t=1",
            "numerical_method": "Conditional noncentral-chi-square CDF and Gauss-Legendre quadrature over noise norm",
            "energy_constraint": "norm<=1; not nominal-only energy without audit",
            "message_sampling": "distinct" if distinct else "iid"}
