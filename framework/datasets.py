"""Dataset generation, splitting, and saving for explicit URA experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch


@dataclass(frozen=True)
class DatasetConfig:
    num_samples: int
    num_active: int
    num_codewords: int
    train_fraction: float = 0.8
    val_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if self.num_active <= 0:
            raise ValueError(f"num_active must be positive, got {self.num_active}")
        if self.num_codewords <= 1:
            raise ValueError(f"num_codewords must be >= 2, got {self.num_codewords}")
        if self.train_fraction <= 0.0 or self.val_fraction < 0.0:
            raise ValueError("train_fraction must be positive and val_fraction nonnegative")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1")


@dataclass
class CountDataset:
    counts: torch.Tensor
    active_messages: torch.Tensor
    config: DatasetConfig

    def split(self) -> dict[str, "CountDataset"]:
        n_train = int(round(self.config.num_samples * self.config.train_fraction))
        n_val = int(round(self.config.num_samples * self.config.val_fraction))
        n_train = min(max(n_train, 1), self.config.num_samples)
        n_val = min(max(n_val, 0), self.config.num_samples - n_train)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, self.config.num_samples),
        }
        return {
            name: CountDataset(self.counts[i:j], self.active_messages[i:j], self.config)
            for name, (i, j) in bounds.items()
        }

    def batches(self, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = int(self.counts.shape[0])
        order = torch.randperm(n, generator=generator, device=self.counts.device) if shuffle else torch.arange(n, device=self.counts.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            yield self.counts[idx], self.active_messages[idx]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "counts": self.counts.cpu(),
            "active_messages": self.active_messages.cpu(),
            "config": asdict(self.config),
        }, path)


def generate_uniform_count_dataset(config: DatasetConfig, generator: torch.Generator | None = None,
                                   device: torch.device | str | None = None,
                                   dtype: torch.dtype = torch.float32) -> CountDataset:
    active = torch.randint(config.num_codewords, (config.num_samples, config.num_active),
                           generator=generator, device=device)
    counts = torch.zeros(config.num_samples, config.num_codewords, dtype=dtype, device=device)
    counts.scatter_add_(1, active.long(), torch.ones_like(active, dtype=counts.dtype))
    return CountDataset(counts=counts, active_messages=active, config=config)


def load_count_dataset(path: Path | str, device: torch.device | str | None = None,
                       dtype: torch.dtype | None = None) -> CountDataset:
    payload = torch.load(Path(path), map_location=device)
    config = DatasetConfig(**payload["config"])
    counts = payload["counts"].to(device=device)
    if dtype is not None:
        counts = counts.to(dtype=dtype)
    active = payload["active_messages"].to(device=device)
    return CountDataset(counts=counts, active_messages=active, config=config)


def make_dataset_sampler(dataset: CountDataset, split: str, batch_size: int,
                         shuffle: bool, generator: torch.Generator | None = None):
    splits = dataset.split()
    if split not in splits:
        raise ValueError(f"unknown split '{split}', available {sorted(splits)}")
    batches = list(splits[split].batches(batch_size, shuffle=shuffle, generator=generator))
    if not batches:
        raise ValueError(f"split '{split}' is empty")
    index = 0

    def sample(_: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal index
        counts, active = batches[index % len(batches)]
        index += 1
        return counts, active

    return sample
