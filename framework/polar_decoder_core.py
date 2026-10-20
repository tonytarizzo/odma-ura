"""Shared physical scenario, count-tail allocation and proposal primitives."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class Scenario:
    phi: np.ndarray
    y: np.ndarray
    k: int
    noise_var: float
    sampling: str = "iid"

    def __post_init__(self):
        if self.phi.ndim != 2 or self.y.shape != (self.phi.shape[0],):
            raise ValueError("Require phi[n,M] and y[n]")
        if not np.isfinite(self.phi).all() or not np.isfinite(self.y).all():
            raise ValueError("Nonfinite observation/codebook")
        if self.k < 1 or int(self.k) != self.k or not np.isfinite(self.noise_var) or self.noise_var <= 0:
            raise ValueError("Require positive integer K and finite physical noise variance >0")
        if self.sampling not in ("iid", "distinct") or (self.sampling == "distinct" and self.k > self.phi.shape[1]):
            raise ValueError("Invalid source/alphabet")
        if not np.allclose(np.sum(self.phi**2, axis=0), 1, rtol=0, atol=1e-10):
            raise ValueError("This decoder bank requires unit-energy columns")


def allocate(tails, k, sampling="iid"):
    """Next-tail Bayes allocation; stable word-ID ties, no truth-dependent readout."""
    tails = np.asarray(tails)
    if tails.ndim != 2 or tails.shape[1] != k or not np.isfinite(tails).all():
        raise ValueError("Expected finite [M,K] count tails")
    if tails.min() < -1e-10 or tails.max() > 1 + 1e-10 or np.any(np.diff(tails, axis=1) > 1e-10):
        raise ValueError("Invalid or nonmonotone count tails")
    counts = np.zeros(len(tails), dtype=np.int64)
    for _ in range(k):
        score = tails[np.arange(len(tails)), np.minimum(counts, k - 1)].copy()
        score[counts >= (1 if sampling == "distinct" else k)] = -np.inf
        counts[np.argmax(score)] += 1
    return counts


def count_tails(tuples, weights, m):
    """One weighted update per transmitted copy, including multiplicities."""
    words = np.sort(tuples, axis=1)
    ranks = np.ones(words.shape, dtype=np.int64)
    for j in range(1, words.shape[1]):
        ranks[:, j] = np.where(words[:, j] == words[:, j - 1], ranks[:, j - 1] + 1, 1)
    tails = np.zeros((m, words.shape[1]))
    np.add.at(tails, (words.ravel(), ranks.ravel() - 1), np.repeat(weights, words.shape[1]))
    return tails



def _guide(s, residual, used, remaining, mean):
    # Isotropic Gaussian continuation is guidance only; exact tuple weights correct it.
    variance = s.noise_var + (remaining - 1) / len(s.y)
    logits = (residual - (remaining - 1) * mean) @ s.phi / variance
    if s.sampling == "distinct" and used.shape[1]:
        logits[np.arange(len(used))[:, None], used] = -np.inf
    return logits - logsumexp(logits, axis=1, keepdims=True)


