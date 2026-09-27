"""Small validation-based early stopper shared by framework training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


@dataclass
class EarlyStopping:
    """Stop after ``patience`` validation epochs without a strict improvement."""

    patience: int = 10
    min_delta: float = 0.0
    best_value: float = float("inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stopped_early: bool = False
    stop_epoch: int | None = None
    _best_states: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError(f"early-stopping patience must be nonnegative, got {self.patience}")
        if self.min_delta < 0.0:
            raise ValueError(f"early-stopping min_delta must be nonnegative, got {self.min_delta}")

    @property
    def enabled(self) -> bool: return self.patience > 0

    def update(self, value: float, epoch: int, modules: dict[str, nn.Module]) -> bool:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"validation loss must be finite, got {value}")
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self._best_states = {name: _clone_state(module) for name, module in modules.items()}
        else:
            self.bad_epochs += 1
        if self.enabled and self.bad_epochs >= self.patience:
            self.stopped_early = True
            self.stop_epoch = int(epoch)
        return self.stopped_early

    def restore(self, modules: dict[str, nn.Module]) -> bool:
        if not self._best_states:
            return False
        for name, module in modules.items():
            module.load_state_dict(self._best_states[name])
        return True

    def summary(self, epochs_run: int, restored_best: bool) -> dict:
        return {
            "enabled": self.enabled, "patience": int(self.patience), "min_delta": float(self.min_delta),
            "monitor": "validation_loss", "epochs_run": int(epochs_run), "best_epoch": int(self.best_epoch),
            "best_validation_loss": float(self.best_value), "stopped_early": bool(self.stopped_early),
            "stop_epoch": self.stop_epoch, "restored_best": bool(restored_best),
        }
