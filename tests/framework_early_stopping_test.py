"""Unit checks for the shared validation early stopper."""

from __future__ import annotations

import torch

from framework.early_stopping import EarlyStopping
from framework.training import TrainConfig


def main() -> None:
    assert TrainConfig().epochs == 200 and TrainConfig().early_stopping_patience == 10
    module = torch.nn.Linear(1, 1, bias=False)
    default = EarlyStopping()
    default.update(1.0, 0, {"model": module})
    for epoch in range(1, 11):
        assert default.update(1.0, epoch, {"model": module}) == (epoch == 10)
    stopper = EarlyStopping(patience=5)
    losses = [1.0, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
    for epoch, loss in enumerate(losses, 1):
        with torch.no_grad():
            module.weight.fill_(epoch)
        stopped = stopper.update(loss, epoch, {"model": module})
    assert stopped and stopper.stop_epoch == 7 and stopper.best_epoch == 2 and stopper.bad_epochs == 5
    assert stopper.restore({"model": module})
    assert float(module.weight.detach()) == 2.0
    summary = stopper.summary(len(losses), restored_best=True)
    assert summary["monitor"] == "validation_loss" and summary["stopped_early"]

    initial = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): initial.weight.zero_()
    zero_stopper = EarlyStopping(patience=2)
    zero_stopper.update(0.5, 0, {"model": initial})
    for epoch in (1, 2):
        with torch.no_grad(): initial.weight.fill_(epoch)
        zero_stopper.update(0.6, epoch, {"model": initial})
    assert zero_stopper.stopped_early and zero_stopper.best_epoch == 0
    assert zero_stopper.restore({"model": initial}) and float(initial.weight.detach()) == 0.0

    disabled = EarlyStopping(patience=0)
    assert not disabled.update(1.0, 1, {"model": module})
    assert not disabled.update(2.0, 2, {"model": module})
    print("early stopping: strict improvement, epoch zero, patience, disable switch, and best-state restoration passed")


if __name__ == "__main__":
    main()
