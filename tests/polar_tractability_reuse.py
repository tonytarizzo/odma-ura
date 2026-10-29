"""Finite checks of residual reuse; enumeration is only an independent reference."""

import json

import numpy as np

from tests.polar_tractability_checks import basis_envelope, characters
from baselines.polar import PolarCode


def main():
    rng = np.random.default_rng(27110)
    b, d, p, n = 3, 8, 4, 16
    generator = PolarCode(b, d, crc_bits=4).encode(np.eye(b, dtype=np.uint8)).T
    keys = generator @ (1 << np.arange(b))
    signs = characters(keys, b)
    positions = np.array([np.sort((np.arange(d) + 4 * j) % n) for j in range(p)])
    amplitudes = np.full(d, 1 / np.sqrt(d))
    # Each resource occurs twice: these uniform energies are exactly row-balanced.
    assert np.all(np.bincount(positions.ravel(), minlength=n) == 2)
    phi = np.zeros((n, p * (1 << b)))
    for j in range(p):
        phi[positions[j], j * (1 << b):(j + 1) * (1 << b)] = (signs * amplitudes).T
    rejection_error, invariance_error, minorization_slack, max_miss_slack = [], [], [], []
    for _ in range(40):
        sigma2 = rng.uniform(.3, 1.2)
        r0, delta = rng.normal(size=n), rng.normal(0, .2, n)
        proposal_parts, bounds = [], []
        for pos in positions:
            coefficients = r0[pos] * amplitudes / sigma2
            groups = {int(key): float(coefficients[keys == key].sum()) for key in set(keys)}
            h0 = groups.pop(0, 0.)
            _, f, q, upper, _ = basis_envelope(np.array(list(groups)), np.array(list(groups.values())), b, h0)
            proposal_parts.append((f, q, upper))
            change = delta[pos] * amplitudes / sigma2
            grouped = {int(key): float(change[keys == key].sum()) for key in set(keys)}
            constant = grouped.pop(0, 0.)
            radius = sum(abs(value) for value in grouped.values())
            bounds.append((constant - radius, constant + radius))
        u0 = sum(part[2] for part in proposal_parts)
        f0 = np.concatenate([part[0] for part in proposal_parts])
        q0 = np.concatenate([part[1] * part[2] / u0 for part in proposal_parts])
        assert np.allclose(np.log(f0), r0 @ phi / sigma2)
        change = delta @ phi / sigma2
        lo = max(min(pair[0] for pair in bounds), -np.linalg.norm(delta) / sigma2)
        hi = min(max(pair[1] for pair in bounds), np.linalg.norm(delta) / sigma2)
        assert np.all(change >= lo - 1e-12) and np.all(change <= hi + 1e-12)
        omega = hi - lo
        f = f0 * np.exp(change)
        target0, target = f0 / f0.sum(), f / f.sum()
        assert np.abs(target - target0).sum() / 2 <= np.tanh(omega / 4) + 1e-12
        selected = np.argsort(-f0)[:7]
        eps0 = min(1., 1 - target0[selected].sum() + .03)
        missed_bound = np.exp(omega) * eps0 / (1 - eps0 + np.exp(omega) * eps0)
        max_miss_slack.append(float(missed_bound - (1 - target[selected].sum())))
        assert min(max_miss_slack) >= -1e-12
        envelope = np.exp(hi) * u0
        acceptance = f / (envelope * q0)
        assert np.max(acceptance) <= 1 + 1e-12
        rejection_error.append(float(np.max(np.abs(q0 * acceptance - f / envelope))))
        z_lower = f0[selected].sum()
        c = np.exp(-omega) * z_lower / u0
        assert f.sum() / envelope >= c - 1e-12
        transition = np.minimum(q0[None, :], target[None, :] * (q0 / target)[:, None])
        transition[np.diag_indices(len(f))] += 1 - transition.sum(1)
        invariance_error.append(float(np.max(np.abs(target @ transition - target))))
        minorization_slack.append(float(np.min(transition - c * target[None, :])))
        assert min(minorization_slack) >= -1e-12
        beta = 1 - (1 - f.sum() / envelope) ** 3
        capped = beta * np.tile(target, (len(target), 1)) + (1 - beta) * np.eye(len(target))
        assert np.allclose(target @ capped, target)

    omega = 2.4
    q = np.array([np.exp(omega / 2), 1.]) / (1 + np.exp(omega / 2))
    tilted = q * np.exp([0., omega])
    tilted /= tilted.sum()
    tight_error = abs(np.abs(tilted - q).sum() / 2 - np.tanh(omega / 4))
    assert max(rejection_error + invariance_error + [tight_error]) < 1e-11
    print(json.dumps({"balanced_polar_context_pairs": 40, "words_per_context": len(f),
                      "maximum_rejection_identity_error": max(rejection_error),
                      "maximum_mh_invariance_error": max(invariance_error),
                      "minimum_minorization_slack": min(minorization_slack),
                      "minimum_list_miss_bound_slack": min(max_miss_slack),
                      "tight_two_state_tv_error": float(tight_error)}, indent=2))


if __name__ == "__main__":
    main()
