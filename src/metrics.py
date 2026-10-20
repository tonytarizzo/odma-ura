"""Detection / estimation metrics for global message-count vectors."""

from __future__ import annotations

import numpy as np

PUPE_VERSION = 2
PUPE_DEFINITION = "multiset_matching_v2"
LEGACY_PUPE_DEFINITION = "duplicate_tolerant_list_v1"


def _hard_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.ndim != 1 or np.iscomplexobj(counts) or not np.all(np.isfinite(counts)):
        raise ValueError("Expected finite real 1-D hard counts")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Expected nonnegative integer hard counts; project soft estimates before scoring")
    return counts.astype(np.float64, copy=False)


def reported_counts(counts: np.ndarray, budget: int) -> np.ndarray:
    """Hard multiset readout, capped in occurrences by descending count (stable ties).

    This is an evaluation list budget, not an estimator of the unknown count vector.
    Valid outputs already totaling at most the budget are returned unchanged.
    """
    counts = _hard_counts(counts)
    if budget < 0 or int(budget) != budget:
        raise ValueError("Multiset budget must be a nonnegative integer")
    if counts.sum() <= budget:
        return counts.copy()
    order = np.argsort(-counts.astype(float), kind="stable")
    remaining = np.maximum(budget - (np.cumsum(counts[order]) - counts[order]), 0)
    out = np.zeros_like(counts)
    out[order] = np.minimum(counts[order], remaining)
    return out


def normalized_l1_accuracy(counts_true: np.ndarray, counts_est: np.ndarray) -> float:
    """1 - normalized L1 count error, clipped to [0, 1].

    For this project counts are non-negative, so the normalizer is the true
    total number of transmitted messages.
    """
    total = float(np.sum(counts_true))
    error = float(np.sum(np.abs(counts_true - counts_est)))
    if total <= 0.0:
        return 1.0 if error == 0.0 else 0.0
    return float(max(0.0, min(1.0, 1.0 - error / total)))


def _top_positive_mask(scores: np.ndarray, max_size: int | None) -> np.ndarray:
    supp = scores > 0
    if max_size is None or int(np.sum(supp)) <= max_size:
        return supp
    idx = np.flatnonzero(supp)
    order = np.argsort(-scores[idx], kind="mergesort")
    keep = idx[order[:max_size]]
    out = np.zeros(scores.shape, dtype=bool)
    out[keep] = True
    return out


def evaluate_counts(counts_true: np.ndarray, counts_hard: np.ndarray,
                    max_list_size: int | None = None) -> dict:
    """Compare true vs estimated global message-count vectors.

    Returns raw support/count diagnostics and bounded multiset PUPE.

    Definitions:
      f1      — support detection F1 (counts_hard > 0 as the active set); ignores multiplicity.
      l1_err  — sum(|a_true - a_hat|) / sum(a_true).
      l1_acc  — clamp(1 - l1_err, 0, 1).
      nmse    — ||a_hat - a_true||_2^2 / ||a_true||_2^2.
      pupe    — 1 - sum(min(a_true, a_reported)) / K; L1/(2K) when both totals equal K.
      list_pupe — historical duplicate-tolerant membership loss (support-list budget).
      raw_list_size/list_size/list_overflow — decoded occurrences before/after/beyond the budget.
      total_count_err — absolute error in the estimated total active-user count.

    The occurrence budget defaults to K and may be reduced explicitly. Missing outputs count as misses;
    excess raw outputs are truncated by descending count. Raw L1/NMSE remain separate diagnostics.
    """
    counts_true, counts_hard = np.asarray(counts_true), np.asarray(counts_hard)
    if counts_true.shape != counts_hard.shape or counts_true.ndim != 1:
        raise ValueError("Expected matching 1-D count shapes")
    counts_true = _hard_counts(counts_true)
    counts_hard = _hard_counts(counts_hard)
    total_true = float(np.sum(counts_true))
    if total_true <= 0:
        raise ValueError("PUPE requires a positive true user count K")
    budget = int(total_true) if max_list_size is None else max_list_size
    if budget > total_true:
        raise ValueError("PUPE output budget must not exceed the true user count K")
    reported = reported_counts(counts_hard, budget)
    supp_true = counts_true > 0
    supp_hard = counts_hard > 0
    list_hard = reported > 0
    legacy_list = _top_positive_mask(counts_hard, max_list_size)
    tp = int(np.sum(supp_true & supp_hard))
    fp = int(np.sum(~supp_true & supp_hard))
    fn = int(np.sum(supp_true & ~supp_hard))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    total_true = float(np.sum(counts_true))
    norm2sq = float(np.sum(counts_true ** 2))
    diff = counts_hard - counts_true
    l1_error_abs = float(np.sum(np.abs(diff)))
    l1_err = l1_error_abs / max(total_true, 1e-12)
    l1_acc = normalized_l1_accuracy(counts_true, counts_hard)
    nmse = float(np.sum(diff ** 2)) / max(norm2sq, 1e-12)

    total_count_err = float(abs(counts_hard.sum() - counts_true.sum()))
    exact_count = float(np.all(counts_hard == counts_true))
    list_tp = int(np.sum(supp_true & list_hard))
    list_fp = int(np.sum(~supp_true & list_hard))
    list_fn = int(np.sum(supp_true & ~list_hard))
    missed_users = float(np.sum(counts_true - np.minimum(counts_true, reported)))
    false_instances = float(np.sum(reported - np.minimum(counts_true, reported)))
    list_size = int(np.sum(reported))
    pupe = missed_users / max(total_true, 1e-12)

    return dict(
        tp=tp, fp=fp, fn=fn,
        f1=float(f1),
        l1_err=l1_err,
        l1_acc=l1_acc,
        nmse=nmse,
        total_count_err=total_count_err,
        exact_count=exact_count,
        support_true=int(np.sum(supp_true)),
        pupe=pupe,
        pupe_metric_version=PUPE_VERSION,
        list_pupe=float(np.sum(counts_true[~legacy_list]) / max(total_true, 1e-12)),
        missed_users=missed_users,
        false_instances=false_instances,
        multiset_false_alarm_rate=false_instances / max(list_size, 1),
        raw_list_size=int(np.sum(counts_hard)),
        list_size=list_size,
        unique_list_size=int(np.sum(list_hard)),
        max_list_size=int(budget),
        list_overflow=max(0, int(np.sum(counts_hard)) - int(budget)),
        list_tp=list_tp,
        list_fp=list_fp,
        list_fn=list_fn,
        false_alarm_rate=float(list_fp / max(int(np.sum(list_hard)), 1)),
    )


def assemble_global_counts(block_coeffs: dict[int, np.ndarray],
                            block_to_msg_list: dict[int, list[int]],
                            num_codewords: int) -> np.ndarray:
    """Convert blockwise coefficient vectors to a global message-count vector."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


def nn_votes_to_counts(x_hat_list: list[np.ndarray],
                       codebook: np.ndarray,
                       num_codewords: int) -> np.ndarray:
    """NN-match each per-device codeword estimate against the codebook and tally votes."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for x_hat in x_hat_list:
        dists = np.sum(np.abs(x_hat[None, :] - codebook) ** 2, axis=1)
        counts[int(np.argmin(dists))] += 1.0
    return counts
