"""Per-batch URA metrics over global message-count vectors."""

from __future__ import annotations

from numbers import Number

import torch


def list_mask(scores: torch.Tensor, max_list_size: int | None) -> torch.Tensor:
    mask = scores > 0
    if max_list_size is None or int(mask.sum().item()) <= int(max_list_size):
        return mask
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    order = torch.argsort(scores[idx], descending=True, stable=True)
    keep = idx[order[: int(max_list_size)]]
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[keep] = True
    return out


def evaluate_counts(counts_true: torch.Tensor, counts_est: torch.Tensor, *,
                      max_list_size: int | None = None) -> dict:
    """Standard URA detection/count metrics for one realisation.

    ``f1`` is support-only: it sees which messages are nonzero, not whether their
    multiplicities are right. Count errors are captured by ``l1_err`` and
    ``total_count_err``.
    """
    if counts_true.shape != counts_est.shape or counts_true.ndim != 1:
        raise ValueError(
            f"expected matching 1-D shapes, got {tuple(counts_true.shape)} vs {tuple(counts_est.shape)}")
    ct = counts_true.detach()
    ce = counts_est.detach()
    supp_true = ct > 0
    supp_est = ce > 0
    list_est = list_mask(ce, max_list_size)
    tp = int((supp_true & supp_est).sum().item())
    fp = int((~supp_true & supp_est).sum().item())
    fn = int((supp_true & ~supp_est).sum().item())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    total_true = float(ct.sum().item())
    diff = ce - ct
    l1_err = float(diff.abs().sum().item()) / max(total_true, 1e-12)
    nmse = float((diff ** 2).sum().item()) / max(float((ct ** 2).sum().item()), 1e-12)
    missed = float(ct[supp_true & ~list_est].sum().item())
    list_tp = int((supp_true & list_est).sum().item())
    list_fp = int((~supp_true & list_est).sum().item())
    list_fn = int((supp_true & ~list_est).sum().item())
    list_size = int(list_est.sum().item())
    raw_list_size = int(supp_est.sum().item())
    return {
        "tp": tp, "fp": fp, "fn": fn, "f1": float(f1),
        "l1_err": float(l1_err),
        "l1_acc": float(max(0.0, min(1.0, 1.0 - l1_err))),
        "nmse": float(nmse),
        "total_count_err": float(abs(ce.sum().item() - ct.sum().item())),
        "exact_count": float(torch.all(ce == ct).item()),
        "support_true": int(supp_true.sum().item()),
        "pupe": float(missed / max(total_true, 1e-12)),
        "missed_users": missed,
        "raw_list_size": raw_list_size,
        "list_size": list_size,
        "max_list_size": None if max_list_size is None else int(max_list_size),
        "list_overflow": 0 if max_list_size is None else max(0, raw_list_size - int(max_list_size)),
        "list_tp": list_tp,
        "list_fp": list_fp,
        "list_fn": list_fn,
        "false_alarm_rate": float(list_fp / max(list_size, 1)),
    }


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Average each numeric field across a list of single-batch metric dicts."""
    if not per_sample:
        return {}
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
