"""Polar symbols with learned energy profiles or an orthogonal pre-placement mixer.

The procedural codebook has no full-message axis. PolarExplicitEncoder is the
small-B D2 certification adapter; its local payload table is not used for search.
"""

import math

import numpy as np
import torch
from torch import nn

from baselines import message_bits
from framework.core import URASpec
from framework.learned_decoders import UnrolledEffectiveChannelPGD


class PolarMappedCodebook(nn.Module):
    global_message_axis_present = False

    def __init__(self, baseline, mapping="identity", learnable=False):
        super().__init__()
        if mapping not in {"identity", "energy", "mixer"}:
            raise ValueError("mapping must be identity, energy or mixer")
        if len(set(baseline.power_levels)) != 1:
            raise ValueError("These comparisons require unit energy for every message")
        if learnable and mapping == "identity":
            raise ValueError("The identity control has no encoder parameters")
        self.payload_bits, self.n = baseline.payload_bits, baseline.n
        self.prefix_bits, self.code_length = baseline.prefix_bits, baseline.code_length
        self.polar, self.mapping = baseline.polar, mapping
        self.register_buffer("positions", torch.as_tensor(baseline.positions, dtype=torch.long).clone())
        self.register_buffer("eye", torch.eye(self.code_length))
        generator = baseline.polar.encode(np.eye(baseline.polar.payload_bits, dtype=np.uint8)).T
        self.register_buffer("symbol_equivalence", torch.as_tensor((generator[:, None] == generator[None, :]).all(-1)))
        self.register_buffer("upper", torch.triu_indices(self.code_length, self.code_length, offset=1))
        shape = (len(self.positions), self.code_length) if mapping == "energy" else (self.upper.shape[1],)
        values = torch.zeros(shape) if mapping != "identity" else torch.empty(0)
        if learnable:
            self.values = nn.Parameter(values)
        else:
            self.register_buffer("values", values)

    @property
    def dtype(self): return self.eye.dtype

    @property
    def device(self): return self.eye.device

    def mixer(self):
        if self.mapping != "mixer": return self.eye
        skew = torch.zeros_like(self.eye).index_put(tuple(self.upper), self.values)
        skew = skew - skew.T
        # Cayley coordinates contain the identity and preserve energy in forward().
        return torch.linalg.solve(self.eye - skew, self.eye + skew)

    def amplitudes(self):
        if self.mapping == "energy":
            return torch.softmax(self.values, dim=-1).sqrt()
        return self.eye.new_full(self.positions.shape, 1 / math.sqrt(self.code_length))

    def transform(self, signs, patterns):
        """Unnormalised BPSK signs (messages,Nc) -> unit-energy local symbols."""
        if self.mapping == "mixer": return (signs @ self.mixer().T) / math.sqrt(self.code_length)
        return signs * self.amplitudes()[patterns]

    def codewords(self, bits):
        bits = torch.as_tensor(bits, device=self.device)
        if bits.ndim != 2 or bits.shape[1] != self.payload_bits or bool(((bits != 0) & (bits != 1)).any()):
            raise ValueError("Expected binary (messages,B) inputs")
        weights = 1 << torch.arange(self.prefix_bits - 1, -1, -1, device=self.device)
        patterns = bits[:, :self.prefix_bits].long() @ weights
        coded = self.polar.encode(bits[:, self.prefix_bits:].detach().cpu().numpy())
        signs = 1 - 2 * torch.as_tensor(coded, dtype=self.dtype, device=self.device)
        local = self.transform(signs, patterns)
        return local.new_zeros((len(bits), self.n)).scatter(1, self.positions[patterns], local).T

    def diagnostics(self):
        with torch.no_grad():
            if self.mapping == "mixer":
                mixer = self.mixer()
                return {"orthogonality_max_error": float((mixer.T @ mixer - self.eye).abs().max()),
                        "mixer_distance_from_identity": float((mixer - self.eye).norm())}
            energy = self.amplitudes().square()
            return {"max_pattern_energy_error": float((energy.sum(-1) - 1).abs().max()),
                    "minimum_symbol_energy": float(energy.min()), "maximum_symbol_energy": float(energy.max()),
                    "energy_distance_from_uniform": float((energy - 1 / self.code_length).norm())}


class PolarExplicitEncoder(nn.Module):
    """Small-B adapter with exact structured matvec, adjoint and D2 geometry."""

    global_message_axis_present = True

    def __init__(self, baseline, loads, mapping="identity", learnable=False):
        super().__init__()
        if baseline.payload_bits > 20: raise ValueError("Full-message D2 certification requires B<=20")
        self.codebook = PolarMappedCodebook(baseline, mapping, learnable)
        self.spec = URASpec(baseline.n, 1 << baseline.payload_bits, max(loads), payload_bits=baseline.payload_bits)
        self.num_payloads = 1 << baseline.polar.payload_bits
        coded = baseline.polar.encode(message_bits(np.arange(self.num_payloads), baseline.polar.payload_bits))
        self.register_buffer("payload_signs", torch.as_tensor(1.0 - 2.0 * coded, dtype=torch.float32), persistent=False)
        self._spectral_cache, self._mean_energy_cache = {}, None

    @property
    def n(self): return self.spec.n

    @property
    def num_codewords(self): return self.spec.num_codewords

    @property
    def dtype(self): return self.codebook.dtype

    @property
    def device(self): return self.codebook.device

    def local_codebooks(self):
        bank = self.codebook
        if bank.mapping == "mixer":
            values = self.payload_signs @ bank.mixer().T / math.sqrt(bank.code_length)
            return values.T.unsqueeze(0).expand(len(bank.positions), -1, -1)
        return bank.amplitudes().unsqueeze(-1) * self.payload_signs.T.unsqueeze(0)

    def message_columns(self, indices):
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long).reshape(-1)
        if bool(((indices < 0) | (indices >= self.num_codewords)).any()): raise ValueError("Message index out of range")
        patterns, payloads = indices // self.num_payloads, indices % self.num_payloads
        values = self.codebook.transform(self.payload_signs[payloads], patterns)
        return values.new_zeros((len(indices), self.n)).scatter(1, self.codebook.positions[patterns], values).T

    def explicit_matrix(self): return self.message_columns(torch.arange(self.num_codewords, device=self.device))

    def matvec(self, counts):
        one = counts.ndim == 1
        counts = counts.reshape(-1, len(self.codebook.positions), self.num_payloads)
        local = torch.einsum("bpv,pcv->bpc", counts, self.local_codebooks())
        rows = self.codebook.positions.reshape(1, -1).expand(len(counts), -1)
        result = local.new_zeros((len(counts), self.n)).scatter_add(1, rows, local.reshape(len(counts), -1))
        return result[0] if one else result

    def rmatvec(self, residual):
        one = residual.ndim == 1
        residual = residual.reshape(-1, self.n)
        result = torch.einsum("bpc,pcv->bpv", residual[:, self.codebook.positions], self.local_codebooks())
        result = result.reshape(len(residual), -1)
        return result[0] if one else result

    def gram_geometry(self):
        bank = self.codebook
        q = bank.symbol_equivalence.to(self.dtype) * self.num_payloads
        if bank.mapping == "mixer":
            u = bank.mixer()
            local_frame = (u @ q @ u.T / bank.code_length).expand(len(bank.positions), -1, -1)
        else:
            a = bank.amplitudes()
            local_frame = a[:, :, None] * q * a[:, None, :]
        indices = bank.positions[:, :, None] * self.n + bank.positions[:, None, :]
        frame = local_frame.new_zeros(self.n * self.n).scatter_add(0, indices.reshape(-1), local_frame.reshape(-1))
        frame = frame.reshape(self.n, self.n)
        projected = frame[bank.positions[:, :, None], bank.positions[:, None, :]]
        local = self.local_codebooks()
        diagonal = local.square().sum(1).reshape(-1)
        row_energy = (local * torch.bmm(projected, local)).sum(1).reshape(-1)
        return diagonal, torch.maximum(row_energy, diagonal.square())

    def apply_constraints(self):
        # Both parameterizations enforce energy in forward, not via an identity penalty.
        self._spectral_cache.clear()
        self._mean_energy_cache = None


class PolarD2(UnrolledEffectiveChannelPGD):
    """Unchanged D2 recurrence; compute its exact geometry through polar factors."""

    def _gram_geometry(self, encoder):
        if not isinstance(encoder, PolarExplicitEncoder): return super()._gram_geometry(encoder)
        cacheable = not any(p.requires_grad for p in encoder.parameters())
        cached = self._geometry_cache
        if cacheable and cached is not None and cached[:3] == (id(encoder), encoder.device, encoder.dtype):
            return cached[3:]
        diagonal, row_energy = encoder.gram_geometry()
        if cacheable:
            self._geometry_cache = (id(encoder), encoder.device, encoder.dtype, diagonal.detach(), row_energy.detach())
        return diagonal, row_energy
