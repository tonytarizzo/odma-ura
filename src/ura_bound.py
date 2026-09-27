"""LEGACY approximate URA reference curves, retained for historical reproducibility.

Do not label these outputs as rigorous Polyanskiy bounds. The Gallager exponent
_E_of_t implements Theorem 1, equations (4)--(10), but _q_of_t uses a Gaussian
approximation for ONE information density, not the minimum over all t-subsets
required in equation (13). The legacy 'count' substitution K->round(E[D]) also
has no supplied achievability proof. 'strict' uses 1-E[D]/K, whereas the paper
counts EVERY user in a duplicate group as an error. Its actual per-user collision
probability is 1-(1-1/M)^(K-1); C(K,2)/M is a frame-event UNION BOUND, not equality.

Numerics/API below are unchanged so old figures can be reproduced and audited.
New comparisons use benchmarks/ura_bounds.py: the conservative, valid Gallager
branch with the original clipping/collision terms and a separately derived
genie list-Fano converse. Physical real-GMAC Eb/N0 = nP/(2B).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm

VARIANTS = ("canonical", "strict", "count")


def distinct_count(num_codewords: int, num_active: int) -> float:
    """Expected number of distinct messages E[D] = M(1-(1-1/M)^Ka).

    Evaluated as -M*expm1(Ka*log1p(-1/M)) so it stays accurate when M is huge
    (e.g. B=100): the naive form rounds (1-1/M)^Ka to exactly 1.0 in float64 and
    collapses E[D] to 0. Uses Python floats throughout to tolerate M=2^100.
    """
    M, Ka = float(num_codewords), int(num_active)
    return -M * math.expm1(Ka * math.log1p(-1.0 / M))


def collision_floor_strict(num_codewords: int, num_active: int) -> float:
    """Legacy 1-E[D]/K statistic; NOT Polyanskiy's collision-as-error floor."""
    Ka = int(num_active)
    return max(0.0, 1.0 - distinct_count(num_codewords, Ka) / Ka)


def collision_prob_union(num_codewords: int, num_active: int) -> float:
    """Union bound C(K,2)/M on P[any collision], potentially greater than one."""
    return math.comb(int(num_active), 2) / float(num_codewords)


def _E_of_t(P1: float, t: np.ndarray, n: int, R1: np.ndarray, R2: np.ndarray,
            grid: int) -> np.ndarray:
    """Vectorised E(t) for an array of t, maximised over a (rho, rho1) grid.

    P1 is P' (per-d.o.f. power of the auxiliary input). Arrays broadcast as
    (rho, rho1, t); invalid-domain points (log of nonpositive) are masked out.
    """
    r = np.linspace(1e-4, 1.0, grid)
    rho = r[:, None, None]
    rho1 = r[None, :, None]
    tt = t[None, None, :].astype(float)
    Pt = P1 * tt
    D = (Pt - 1.0) ** 2 + 4.0 * Pt * (1.0 + rho * rho1) / (1.0 + rho)
    lam = (Pt - 1.0 + np.sqrt(D)) / (4.0 * (1.0 + rho1 * rho) * Pt)
    mu = rho * lam / (1.0 + 2.0 * Pt * lam)
    a = 0.5 * rho * np.log1p(2.0 * Pt * lam) + 0.5 * np.log1p(2.0 * Pt * mu)
    b = rho * lam - mu / (1.0 + 2.0 * Pt * mu)
    inner = 1.0 - 2.0 * b * rho1
    with np.errstate(invalid="ignore", divide="ignore"):
        E0 = rho1 * a + 0.5 * np.log(inner)
        expo = E0 - rho1 * R2[None, None, :] - rho * rho1 * tt * R1[None, None, :]
    expo = np.where(inner > 0.0, expo, -np.inf)
    return np.nanmax(expo.reshape(-1, t.size), axis=0)


def _q_of_t(P1: float, t: np.ndarray, n: int, R1: np.ndarray, R2: np.ndarray,
            zgrid: int = 400) -> np.ndarray:
    """Vectorised q_t = min_gamma[ Phi((gamma-nC_t)/sigma) + exp(n(tR1+R2)-gamma) ].

    Normal approximation of the information density i_t (mean nC_t, variance
    n tP'/(1+tP')). Minimised over gamma on a normalised z-grid (gamma = nC_t + sigma z).
    """
    p = P1 * t.astype(float)
    v = 1.0 + p
    nCt = 0.5 * n * np.log(v)
    sigma = np.sqrt(n * p / v)
    Delta = nCt - n * (t * R1 + R2)  # exp term = exp(-Delta - sigma z)
    z = np.linspace(-8.0, 3.0, zgrid)[:, None]
    expo = np.clip(-Delta[None, :] - sigma[None, :] * z, -700.0, 50.0)
    f = norm.cdf(z) + np.exp(expo)
    return np.min(f, axis=0)


def _epsilon(P: float, P1: float, n: int, num_codewords: int, num_active: int,
             num_msgs: int, floor: float, grid: int) -> float:
    """Predicted per-user error at power P, auxiliary power P' = P1.

    ``num_msgs`` (= J) is the population entering the combinatorial rate R2 and the
    t-range (Ka for canonical/strict, E[D] for count). ``floor`` is the collision
    floor added to p_0. The clip term always uses the actual Ka users.
    """
    Ka = int(num_active)
    J = max(1, int(round(num_msgs)))
    t = np.arange(1, J + 1)
    logM = math.log(num_codewords)
    R1 = logM / n - np.array([math.lgamma(int(ti) + 1) for ti in t]) / (n * t)
    R2 = np.array([math.lgamma(J + 1) - math.lgamma(int(ti) + 1) - math.lgamma(J - int(ti) + 1)
                   for ti in t]) / n
    p_t = np.exp(-n * _E_of_t(P1, t, n, R1, R2, grid))
    q_t = _q_of_t(P1, t, n, R1, R2)
    sum_term = float(np.sum((t / Ka) * np.minimum(p_t, q_t)))
    clip = Ka * float(chi2.sf(n * P / P1, df=n))
    return sum_term + clip + floor


def _variant_params(variant: str, num_codewords: int, num_active: int) -> tuple[float, float]:
    """Return (J, floor) for a variant: J = R2 population, floor = collision floor."""
    if variant == "canonical":
        return float(num_active), 0.0
    if variant == "strict":
        return float(num_active), collision_floor_strict(num_codewords, num_active)
    if variant == "count":
        return distinct_count(num_codewords, num_active), 0.0
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def _min_epsilon(ebn0_db: float, n: int, payload_bits: int, num_codewords: int,
                 num_active: int, num_msgs: float, floor: float,
                 grid: int, num_pprime: int) -> float:
    """Minimum predicted error over P' in (0, P] at the given Eb/N0."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    P = 2.0 * payload_bits * ebn0 / n
    best = math.inf
    for frac in np.linspace(0.05, 1.0, num_pprime):
        best = min(best, _epsilon(P, frac * P, n, num_codewords, num_active, num_msgs, floor, grid))
    return best


def required_ebn0_db(n: int, payload_bits: int, num_active: int, target: float, *,
                     variant: str = "count",
                     ebn0_min: float = -6.0, ebn0_max: float = 20.0, tol: float = 0.02,
                     grid: int = 25, num_pprime: int = 25) -> float:
    """Smallest physical Eb/N0 (dB, GMAC convention) meeting per-user error <= target.

    Returns +inf when the (variant) collision floor alone exceeds `target`
    (unreachable at any SNR for this M) or when the bracket top still misses it.
    """
    num_codewords = 1 << int(payload_bits)
    num_msgs, floor = _variant_params(variant, num_codewords, num_active)
    if floor > target:
        return math.inf

    def eps(x: float) -> float:
        return _min_epsilon(x, n, payload_bits, num_codewords, num_active, num_msgs, floor,
                            grid, num_pprime)

    if eps(ebn0_max) > target:
        return math.inf
    if eps(ebn0_min) <= target:
        return ebn0_min
    lo, hi = ebn0_min, ebn0_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps(mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def to_experiment_axis(ebn0_db_phys: float, num_antennas: int, *, real_awgn: bool = True) -> float:
    """Map a physical GMAC Eb/N0 to this repo's reported Eb/N0 convention.

    Two offsets, made explicit so the mapping is auditable:
      * this repo defines Eb/N0 with N0 = sigma^2 (see src/signal.py, "no real-AWGN
        1/2 factor"), i.e. +3.01 dB relative to the physical N0 = 2 sigma^2;
      * the V2 common-signature model matched-filters M_ant antennas (h = 1), giving
        a 10 log10(M_ant) dB SNR gain that the reported axis excludes.
    Net: reported = phys - 10 log10(M_ant) + [3.01 if real_awgn else 0].
    For M_ant = 2 the two offsets cancel and reported == phys.
    """
    if not math.isfinite(ebn0_db_phys):
        return ebn0_db_phys
    shift = -10.0 * math.log10(num_antennas)
    if real_awgn:
        shift += 10.0 * math.log10(2.0)
    return ebn0_db_phys + shift


def required_ebn0_curve(n: int, payload_bits: int, k_values: list[int], target: float, *,
                        variants: tuple[str, ...] = VARIANTS,
                        num_antennas: int = 2, real_awgn: bool = True, **kw) -> dict:
    """Required Eb/N0 vs K for each variant on the experiment axis, plus diagnostics.

    Returns a dict keyed by K; each entry has one sub-dict per variant with the
    physical and experiment-axis Eb/N0, and the finite-M collision diagnostics.
    """
    num_codewords = 1 << int(payload_bits)
    out: dict[int, dict] = {}
    for K in k_values:
        entry: dict = {
            "distinct_count": distinct_count(num_codewords, int(K)),
            "collision_floor_strict": collision_floor_strict(num_codewords, int(K)),
            "collision_prob_union": collision_prob_union(num_codewords, int(K)),
        }
        for v in variants:
            phys = required_ebn0_db(n, payload_bits, int(K), target, variant=v, **kw)
            entry[v] = {
                "ebn0_db_phys": phys,
                "ebn0_db_experiment": to_experiment_axis(phys, num_antennas, real_awgn=real_awgn),
            }
        out[int(K)] = entry
    return out
