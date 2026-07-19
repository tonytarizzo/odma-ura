"""URA data generation and AWGN channel with multi-antenna known fading.

Data flow for one batch (B realisations):
    counts  ~ generator()                     # (B, M) nonnegative integers
    y       = Phi counts                      # (B, n)
    Y_clean = outer(y, h) for each b          # (B, n, M_ant)
    Y       = Y_clean + Z                     # AWGN

The receiver is assumed to know `h`. The default generator picks h = 1_{M_ant}
so the resulting Y is a stack of identical replicas - this matches the legacy
"common signature" V2 setup. Alternative `h` distributions are passed as a
callable to support future experiments without changing the call site.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from .core import URABatch
from .encoder import Encoder


# --- count-vector generators ----------------------------------------------


def uniform_counts_generator(num_active: int, num_codewords: int,
                              generator: torch.Generator | None = None,
                              device: torch.device | str | None = None
                              ) -> Callable[[int], tuple[torch.Tensor, torch.Tensor]]:
    """Each of K_a active users picks a message i.i.d. uniformly from [M]."""
    def sample(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        active = torch.randint(num_codewords, (batch_size, num_active),
                                generator=generator, device=device)
        counts = torch.zeros(batch_size, num_codewords, dtype=torch.float32, device=device)
        counts.scatter_add_(1, active.long(), torch.ones_like(active, dtype=counts.dtype))
        return counts, active
    return sample


def uniform_count_range_generator(num_active_min: int, num_active_max: int, num_codewords: int,
                                  generator: torch.Generator | None = None,
                                  device: torch.device | str | None = None
                                  ) -> Callable[[int], tuple[torch.Tensor, torch.Tensor]]:
    """Sample one K_a uniformly from an inclusive range for each generated batch."""
    if num_active_min <= 0 or num_active_max < num_active_min:
        raise ValueError(f"invalid active-user range [{num_active_min}, {num_active_max}]")
    if num_active_max > num_codewords:
        raise ValueError(f"num_active_max={num_active_max} exceeds message alphabet M={num_codewords}")

    def sample(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        K = int(torch.randint(num_active_min, num_active_max + 1, (1,), generator=generator, device=device).item())
        active = torch.randint(num_codewords, (batch_size, K), generator=generator, device=device)
        counts = torch.zeros(batch_size, num_codewords, dtype=torch.float32, device=device)
        counts.scatter_add_(1, active.long(), torch.ones_like(active, dtype=counts.dtype))
        return counts, active
    return sample


# --- channel realisation generators ---------------------------------------


def constant_fading(num_antennas: int, dtype: torch.dtype = torch.float32,
                     device: torch.device | str | None = None
                     ) -> Callable[[int], torch.Tensor]:
    """h = 1_{M_ant} for every realisation (legacy V2 convention)."""
    def sample(batch_size: int) -> torch.Tensor:
        return torch.ones(batch_size, num_antennas, dtype=dtype, device=device)
    return sample


def iid_gaussian_fading(num_antennas: int, dtype: torch.dtype = torch.float32,
                         device: torch.device | str | None = None,
                         generator: torch.Generator | None = None
                         ) -> Callable[[int], torch.Tensor]:
    """h ~ CN(0, I_{M_ant}) per realisation (assumed known at the receiver)."""
    def sample(batch_size: int) -> torch.Tensor:
        if dtype.is_complex:
            real = torch.randn(batch_size, num_antennas, generator=generator,
                               dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
                               device=device)
            imag = torch.randn(batch_size, num_antennas, generator=generator,
                               dtype=real.dtype, device=device)
            return (real + 1j * imag).to(dtype) / math.sqrt(2.0)
        return torch.randn(batch_size, num_antennas, dtype=dtype, generator=generator, device=device)
    return sample


# --- Eb/N0 convention -----------------------------------------------------


def ebn0_db_to_noise_var(ebn0_db: float, payload_bits: int,
                          energy_per_codeword: float = 1.0) -> float:
    if payload_bits <= 0:
        raise ValueError(f"payload_bits must be positive, got {payload_bits}")
    if energy_per_codeword <= 0.0:
        raise ValueError(f"energy_per_codeword must be positive, got {energy_per_codeword}")
    ebn0_lin = 10.0 ** (float(ebn0_db) / 10.0)
    return float(energy_per_codeword / (payload_bits * ebn0_lin))


def empirical_codeword_energy(encoder: Encoder) -> float:
    return encoder.mean_codeword_energy()


# --- one-batch sampler ----------------------------------------------------


def sample_batch(encoder: Encoder, batch_size: int,
                  counts_sampler: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
                  fading_sampler: Callable[[int], torch.Tensor],
                  ebn0_db: float,
                  generator: torch.Generator | None = None,
                  energy_per_codeword: float | None = None) -> URABatch:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    counts, active = counts_sampler(batch_size)
    counts = counts.to(dtype=encoder.dtype, device=encoder.device)
    num_active = counts.real.sum(dim=1).round().to(torch.long)
    actual_batch_size = int(counts.shape[0])
    y = encoder.encode(counts)                    # (B, n)
    H = fading_sampler(actual_batch_size).to(dtype=encoder.dtype, device=encoder.device)
    Y_clean = y.unsqueeze(-1) * H.unsqueeze(1)     # (B, n, M_ant)
    energy = empirical_codeword_energy(encoder) if energy_per_codeword is None else float(energy_per_codeword)
    noise_var = ebn0_db_to_noise_var(ebn0_db, encoder.spec.payload_bits, energy)
    if encoder.dtype.is_complex:
        noise = torch.randn(Y_clean.shape, dtype=Y_clean.dtype, device=Y_clean.device,
                             generator=generator) * math.sqrt(noise_var / 2.0)
    else:
        noise = torch.randn(Y_clean.shape, dtype=Y_clean.dtype, device=Y_clean.device,
                             generator=generator) * math.sqrt(noise_var)
    Y = Y_clean + noise
    return URABatch(counts=counts, y_clean=y, Y_clean=Y_clean, Y=Y, H=H,
                     noise_var=noise_var, active_messages=active, num_active=num_active,
                     ebn0_db=float(ebn0_db))


def matched_filter_collapse(Y: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Collapse the multi-antenna observation back to a scalar (B, n) signal.

    With known H, the matched filter y_mf_b = (1/||h_b||^2) sum_m h_{b,m}^* Y_{b,n,m}
    leaves a clean signal exactly equal to Phi a_b (in the noiseless case), so
    decoders can ignore antennas. The collapsed noise has variance
    sigma^2 / ||h_b||^2 per resource sample; that scaling is the caller's
    responsibility if needed.
    """
    if Y.ndim != 3:
        raise ValueError(f"Y must have shape (B, n, M_ant), got {tuple(Y.shape)}")
    if H.ndim != 2 or H.shape[0] != Y.shape[0] or H.shape[1] != Y.shape[2]:
        raise ValueError(f"H must have shape (B, M_ant), got {tuple(H.shape)}")
    h_conj = H.conj() if H.is_complex() else H
    energy = (H.conj() * H).sum(-1).real.clamp_min(1e-12) if H.is_complex() else (H * H).sum(-1).clamp_min(1e-12)
    return torch.einsum("bnm,bm->bn", Y, h_conj) / energy.unsqueeze(-1)
