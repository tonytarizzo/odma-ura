"""Pinned-author CCS-AMP/BP with the two-pass SIC procedure in Section V.

The dense mode preserves the author's nominal (not per-message) energy policy.
The block-diagonal mode is an exact-unit-energy adaptation, not the dense paper
curve. Empirical SIC/list/iteration settings are explicit, not claimed published.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import MethodType

import numpy as np
from scipy.special import expit

from tests.ccs_amp_author import AUTHOR_COMMIT, AUTHOR_REPO, default_author_dir, load_author_modules


def _fast_fht(values):
    """Same decreasing-stride butterflies as upstream, vectorized in NumPy."""
    stride = len(values) // 2
    while stride:
        blocks = values.reshape(-1, 2, stride)
        left, right = blocks[:, 0].copy(), blocks[:, 1].copy()
        blocks[:, 0], blocks[:, 1] = left + right, left - right
        stride //= 2


def _stable_denoiser(inner, q, observation, tau):
    q = np.clip(np.asarray(q, dtype=float).reshape(-1), 1e-15, 1 - 1e-15)
    amplitude = np.sqrt(inner.getPhat())
    variance = max(float(np.asarray(tau).item()) ** 2, 1e-24)
    logits = np.log(q) - np.log1p(-q) + (amplitude * np.asarray(observation).reshape(-1) - amplitude**2 / 2) / variance
    return np.clip(expit(np.asarray(logits, dtype=float)), 1e-15, 1 - 1e-15).reshape(-1, 1)


class CCSAMPBaseline:
    def __init__(self, payload_bits: int, n: int, seed: int = 0, *, inner_mode: str = "dense",
                 information_sections: int | None = None, amp_iterations: int = 20, bp_iterations: int = 1,
                 sic_delta: int | None = None, sic_fraction: float = 0.7, list_extra: int = 10,
                 non_dc_embedding: bool = False, author_dir: str | Path | None = None):
        self.payload_bits, self.n, self.seed = int(payload_bits), int(n), int(seed)
        if min(self.payload_bits, self.n) < 1 or self.seed < 0:
            raise ValueError("Require positive B,n and a nonnegative seed")
        if information_sections is None:
            information_sections = 8 if self.payload_bits == 128 else 10 if self.payload_bits == 100 else 2
        self.information_sections = int(information_sections)
        if self.information_sections < 2 or self.payload_bits % self.information_sections:
            raise ValueError("payload_bits must be divisible by information_sections >= 2; no silent padding")
        self.section_bits = self.payload_bits // self.information_sections
        if not 1 <= self.section_bits <= 18:
            raise ValueError("section_bits must be in [1,18]; choose more information sections at large B")
        if inner_mode not in {"dense", "block_diagonal"}:
            raise ValueError("inner_mode must be dense or block_diagonal")
        if amp_iterations < 1 or bp_iterations != 1 or list_extra < 0 or not 0 < sic_fraction < 1:
            raise ValueError("positive AMP budget, exactly one extrinsic BP step, nonnegative list_extra and 0<sic_fraction<1 required")
        if sic_delta is not None and sic_delta < 1:
            raise ValueError("sic_delta must be positive")
        self.inner_mode, self.amp_iterations, self.bp_iterations = inner_mode, int(amp_iterations), int(bp_iterations)
        self.sic_delta, self.sic_fraction, self.list_extra = sic_delta, float(sic_fraction), int(list_extra)
        self.section_size = 1 << self.section_bits
        self.non_dc_embedding = bool(non_dc_embedding)
        self.modules = load_author_modules(Path(author_dir) if author_dir else default_author_dir(), transform_seed=self.seed)
        self.modules.fht.fht = _fast_fht
        transform = self.modules.fht.block_sub_fht

        def selected_embedding(n, m, l, **kwargs):
            kwargs["new_embedding"] = self.non_dc_embedding
            return transform(n, m, l, **kwargs)

        self.modules.inner.block_sub_fht = selected_embedding
        # Two fragments need only one parity check. Repeating it would create a short cycle and redundant parity.
        count = self.information_sections
        self.checks = [[1, 2, 3]] if count == 2 else [[2*i+1, 2*i+2, 2*((i+1) % count)+1] for i in range(count)]
        self.information_nodes = [1, 3] if count == 2 else list(range(1, 2*count, 2))
        with contextlib.redirect_stdout(io.StringIO()):
            self.graph = self.modules.FG.ccsfg.Encoding(self.checks, self.information_nodes, self.section_bits)
        self.graph.maxdepth = len(self.checks)
        self.sections = self.graph.varcount
        self.used_n = self.n if inner_mode == "dense" else self.n // self.sections * self.sections
        if self.used_n < self.sections:
            raise ValueError("n is too small for the outer graph")
        self._inner_type = getattr(self.modules.inner, "DenseInnerCode" if inner_mode == "dense" else "BlockDiagonalInnerCode")
        self._encoder = self._new_inner(1, 1.0)
        self._rows_per_block = self.used_n if inner_mode == "dense" else self.used_n // self.sections
        blocks = self.sections if inner_mode == "dense" else 1
        _, _, self._ordering = selected_embedding(self._rows_per_block, self.section_size, blocks, seed=self.seed)
        extent = max(self.section_size + int(self.non_dc_embedding), self._rows_per_block + 1)
        self._transform_bits = (extent - 1).bit_length()
        self._column_offset = (1 << self._transform_bits) - self.section_size if self.non_dc_embedding else 0

    def _new_inner(self, active, noise_var):
        inner = self._inner_type(self.used_n, 1.0 / self.used_n, np.sqrt(noise_var), active, self.graph)
        inner.AmpDenoiser = MethodType(_stable_denoiser, inner)
        return inner

    def section_indices(self, messages):
        messages = np.asarray(messages)
        if messages.ndim != 2 or messages.shape[1] != self.payload_bits or not np.isin(messages, [0, 1]).all():
            raise ValueError("messages must have shape [K,payload_bits] and contain binary values")
        fragments = messages.astype(np.int64).reshape(-1, self.information_sections, self.section_bits)
        labels = fragments @ (1 << np.arange(self.section_bits - 1, -1, -1))
        indices = np.zeros((len(messages), self.sections), dtype=np.int64)
        indices[:, np.asarray(self.information_nodes) - 1] = labels
        for left, parity, right in self.checks:
            indices[:, parity - 1] = (-indices[:, left - 1] - indices[:, right - 1]) % self.section_size
        return indices

    def encode(self, messages: np.ndarray) -> np.ndarray:
        indices = self.section_indices(messages)
        waveforms = np.zeros((len(indices), self.n), dtype=float)
        scale = 1.0 / np.sqrt(self.sections * self._rows_per_block)
        for start in range(0, len(indices), 32):
            batch = indices[start:start + 32]
            for section in range(self.sections):
                rows = self._ordering[section if self.inner_mode == "dense" else 0]
                products = (batch[:, section, None] + self._column_offset).astype(np.uint64) & rows[None, :].astype(np.uint64)
                parity = np.zeros(products.shape, dtype=np.uint8)
                for bit in range(self._transform_bits):
                    parity ^= ((products >> bit) & 1).astype(np.uint8)
                values = (1.0 - 2.0 * parity) * scale
                offset = 0 if self.inner_mode == "dense" else section * self._rows_per_block
                waveforms[start:start + len(batch), offset:offset + self._rows_per_block] += values
        return waveforms

    def _messages(self, codewords):
        if not len(codewords):
            return np.empty((0, self.payload_bits), dtype=np.uint8)
        indices = np.argmax(np.asarray(codewords).reshape(-1, self.sections, self.section_size), axis=2)
        labels = indices[:, np.asarray(self.information_nodes) - 1]
        shifts = np.arange(self.section_bits - 1, -1, -1)
        return ((labels[..., None] >> shifts) & 1).astype(np.uint8).reshape(-1, self.payload_bits)

    def _pass(self, observation, active, noise_var):
        inner = self._new_inner(active, noise_var)
        # The paper's Bernoulli AMP approximation requires low local occupancy, not just distinct full messages.
        if active > self.section_size:
            raise ValueError("active users exceed the section alphabet; choose larger sections")
        with contextlib.redirect_stdout(io.StringIO()):
            estimates, tau = inner.Decode(observation[:self.used_n, None], self.amp_iterations, True, 1, self.graph)
            if not np.isfinite(estimates).all() or not np.isfinite(tau).all():
                raise FloatingPointError("author AMP produced non-finite estimates; do not treat this as a failed transmission")
            codewords, likelihoods = self.graph.decoder(estimates.copy(), min(self.section_size, active + self.list_extra), True)
        return self._messages(codewords), np.asarray(likelihoods, dtype=float), np.asarray(tau).reshape(-1).tolist()

    def decode(self, y: np.ndarray, num_active: int, noise_var: float):
        observation = np.asarray(y)
        if (observation.shape != (self.n,) or np.iscomplexobj(observation) or not np.isfinite(observation).all()
                or not np.isfinite(noise_var) or noise_var <= 0):
            raise ValueError("finite real observation [n] and positive finite noise_var required")
        observation = observation.astype(float)
        active = int(num_active)
        if active < 1 or active != num_active:
            raise ValueError("num_active must be positive")
        first, first_scores, tau_first = self._pass(observation, active, noise_var)
        delta = min(active, self.sic_delta if self.sic_delta is not None else max(1, active - int(np.ceil(active*self.sic_fraction))))
        cancelled_count = min(len(first), active - delta)
        cancelled = first[:cancelled_count]
        residual = observation - self.encode(cancelled).sum(axis=0)
        remaining = active - cancelled_count
        second, second_scores, tau_second = self._pass(residual, remaining, noise_var)
        # Retain all candidates from both passes, then enforce the paper's final K-sized likelihood-ranked list.
        candidates = {}
        for message, score in zip(np.concatenate((first, second)), np.concatenate((first_scores, second_scores))):
            key = message.tobytes()
            if key not in candidates or score > candidates[key][0]:
                candidates[key] = (float(score), message)
        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)[:active]
        decoded = np.asarray([row[1] for row in ranked], dtype=np.uint8).reshape(-1, self.payload_bits)
        meta = {"passes": 2, "amp_iterations_per_pass": self.amp_iterations, "bp_iterations": 1,
                "sic_delta_requested": delta, "sic_cancelled": cancelled_count, "second_pass_active": remaining,
                "first_candidates": len(first), "second_candidates": len(second), "tau_first": tau_first, "tau_second": tau_second,
                "expected_local_colliding_pairs": active*(active-1)/(2*self.section_size),
                "oracle_candidates": False, "energy_policy": self.metadata()["energy_policy"]}
        return decoded, meta

    def metadata(self):
        native_shape = self.payload_bits == 128 and self.n == 38400 and self.information_sections == 8
        return {"source": "https://arxiv.org/abs/2010.04364", "author_repo": AUTHOR_REPO, "author_commit": AUTHOR_COMMIT,
                "variant": f"author_amp_bp_two_pass_{self.inner_mode}", "payload_bits": self.payload_bits, "n": self.n,
                "seed": self.seed, "sections": self.sections, "information_sections": self.information_sections,
                "section_bits": self.section_bits, "section_size": self.section_size, "used_channel_uses": self.used_n,
                "graph": "single_triad_small_payload_adaptation" if self.information_sections == 2 else "triadic_cycle",
                "parity_checks": len(self.checks), "parity_check_rank": len(self.checks),
                "energy_policy": "per_codeword_unit" if self.inner_mode == "block_diagonal" else "nominal_unit_not_per_codeword",
                "nominal_energy": 1.0, "amp_iterations": self.amp_iterations, "bp_iterations": 1,
                "sic_delta": self.sic_delta, "sic_fraction": self.sic_fraction, "list_extra": self.list_extra,
                "non_dc_embedding": self.non_dc_embedding,
                "paper_native_dimensions": native_shape, "published_curve_reproduced": False,
                "reproduction_caveats": ["Two-pass SIC is implemented; empirical delta and iteration budget are not supplied by the paper.",
                    "Public author's block-Hadamard operator is retained; numerical logistic evaluation and FHT are stabilized/vectorized.",
                    "Dense waveforms have message-dependent energy and must not be silently column-normalized.",
                    "Block-diagonal sensing and the small-payload single-triad graph are labelled adaptations.",
                    "Original embedding has a shared constant column; small-B zero-fragment messages can destabilize AMP.",
                    "Optional non-DC embedding is an upstream transform option, tested as a small-B adaptation only.",
                    "The native graph matches the K<200 design; the paper's extra high-load parity checks are not reconstructed.",
                    "Low section occupancy is a modelling approximation; duplicates are not multiplicity-decoded."]}
