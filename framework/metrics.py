"""Per-batch URA metrics over global message-count vectors."""

from __future__ import annotations

from numbers import Number

import torch

from src.metrics import evaluate_counts as numpy_evaluate_counts


def evaluate_counts(counts_true: torch.Tensor, counts_est: torch.Tensor, *,
                    max_list_size: int | None = None) -> dict:
    """Score hard counts with the shared NumPy multiset-PUPE contract."""
    return numpy_evaluate_counts(counts_true.detach().cpu().numpy(), counts_est.detach().cpu().numpy(),
                                 max_list_size=max_list_size)


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Average each numeric field across a list of single-batch metric dicts."""
    if not per_sample:
        return {}
    versions = {d.get("pupe_metric_version", 1) for d in per_sample}
    if len(versions) != 1:
        raise ValueError("Cannot aggregate different PUPE definitions")
    keys = list(per_sample[0])
    out = {}
    for k in keys:
        vals = [d[k] for d in per_sample]
        if all(isinstance(v, Number) for v in vals):
            out[k] = float(sum(float(v) for v in vals) / len(vals))
        else:
            out[k] = vals[0]
    return out


def batch_evaluate(counts_true: torch.Tensor, counts_est: torch.Tensor, *,
                    max_list_size: int | None = None) -> tuple[list[dict], dict]:
    if counts_true.shape != counts_est.shape:
        raise ValueError(f"count shapes disagree: {tuple(counts_true.shape)} vs {tuple(counts_est.shape)}")
    if counts_true.ndim == 1:
        m = evaluate_counts(counts_true, counts_est, max_list_size=max_list_size)
        return [m], m
    if counts_true.ndim != 2:
        raise ValueError(f"expected (M,) or (B, M), got {tuple(counts_true.shape)}")
    per = [evaluate_counts(t, e, max_list_size=max_list_size)
            for t, e in zip(counts_true, counts_est)]
    return per, aggregate_metrics(per)
