"""Conservative finite-blocklength references for the real Gaussian MAC.

Physical Eb/N0 = 1/(2 B sigma^2) for nominal unit codeword energy.
The survey's Gaussian achievability reference is Polyanskiy's Theorem 1, not
another independent bound. We evaluate its Gallager branch (p_t) only. Dropping
min(p_t,q_t) makes a valid but looser upper bound on achievable error, not the
tight published curve. Grid restriction also only loosens this upper bound.

The other curve is a weak genie-aided list-Fano converse, derived here; it is
not the survey's achievability or a claimed tight finite-blocklength converse.
"""

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, xlogy
from scipy.stats import chi2

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


def list_fano_converse(payload_bits, n, num_active, ebn0_db, *, distinct=False):
    """Genie reveals other messages, leaving one AWGN codeword and a list of <=K.

    log M <= C + h(e) + (1-e) log K + e log(M-K).
    C <= n/2 log(1 + 2 B Eb/N0/n). Units are nats. Under distinct
    sampling the remaining alphabet after the genie has size M-K+1.
    This lower-bounds list-miss PUPE, not multiplicity estimation error.
    """
    b, k = int(payload_bits), int(num_active)
    if min(b, n, k) < 1 or k >= 2 ** b:
        raise ValueError("Require positive B,n,K and K<2^B")
    alphabet = 2 ** b - (k - 1 if distinct else 0)
    if alphabet <= k:
        return 0.0
    log_m = math.log(alphabet)
    capacity = n / 2 * math.log1p(2 * b * 10.0 ** (float(ebn0_db) / 10) / n)

    def gap(error):
        entropy = -xlogy(error, error) - xlogy(1 - error, 1 - error)
        return log_m - capacity - entropy - (1 - error) * math.log(k) - error * math.log(alphabet - k)

    return 0.0 if gap(0.0) <= 0 else float(brentq(gap, 0, 1 - k / alphabet))


def reference_curves(payload_bits, n, num_active, ebn0_db, *, distinct=False):
    ebn0_db = list(ebn0_db)
    components = [polyanskiy_gallager(payload_bits, n, num_active, x, distinct=distinct, details=True) for x in ebn0_db]
    return {"ebn0_db": list(ebn0_db),
            "polyanskiy_gallager_achievability": [row["upper_bound"] for row in components],
            "achievability_components": components,
            "genie_list_fano_converse": [list_fano_converse(payload_bits, n, num_active, x, distinct=distinct)
                                         for x in ebn0_db],
            "source": "https://people.lids.mit.edu/yp/homepage/data/isit17_mac.pdf",
            "achievability_label": "Polyanskiy achievable-error upper bound (Gallager-only; may be loose)",
            "achievability_direction": "Optimal PUPE <= this value; practical decoders may perform below this curve",
            "not_evaluated": "Thm.1 q_t branch: minimum information density over every size-t active-user subset",
            "converse_label": "Genie list-Fano lower bound (weak)",
            "energy_constraint": "norm<=1 or average<=1 (converse); not nominal-only energy without audit",
            "message_sampling": "distinct" if distinct else "iid"}
