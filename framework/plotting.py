"""Lightweight plotting helpers for framework runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_training_curves(progress: list[dict], path: Path, *,
                          metric_keys: Iterable[str] = ("loss", "loss_dec",
                                                          "eval_pupe", "eval_f1",
                                                          "eval_l1_err")) -> None:
    if not progress:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in progress]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in metric_keys:
        if not all(key in r for r in progress):
            continue
        ax = axes[0] if "eval" not in key else axes[1]
        ax.plot(epochs, [float(r[key]) for r in progress], marker="o", label=key)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("training loss")
    axes[1].set_title("evaluation metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_count_estimate(counts_true, counts_est, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    idx = list(range(len(counts_true)))
    ax.bar(idx, counts_true, alpha=0.4, label="true")
    ax.bar(idx, counts_est, alpha=0.4, label="estimate")
    ax.set_xlabel("message index")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
