"""Gaussian-MAC ODMA + CRC-aided polar SCL + pattern detection + iterative SIC.

Independent implementation of Ozates, Kazemi and Duman, IEEE WCL 2024, Sec. III.
This implements the complete Gaussian receiver chain, not a verified reproduction of its published curve.
The paper does not specify every implementation detail (CRC polynomial, TIN variance, detector slack).
These choices, and small-payload adaptations, are explicit in metadata().
"""

from __future__ import annotations

import time

import numpy as np

from .polar import PolarCode


class ODMAPolarBaseline:
    def __init__(self, payload_bits: int, n: int, seed: int = 0, *, prefix_bits: int | None = None,
                 code_length: int | None = None, crc_bits: int = 16, list_size: int = 128,
                 detector_slack: int = 5, max_iterations: int = 10, power_levels: tuple[float, ...] = (1.0,),
                 sic_mode: str = "parallel"):
        self.payload_bits, self.n, self.seed = int(payload_bits), int(n), int(seed)
        self.prefix_bits = int(prefix_bits if prefix_bits is not None else min(12, max(1, self.payload_bits // 2)))
        self.code_length = int(code_length if code_length is not None else (512 if self.payload_bits >= 60 else 64))
        self.detector_slack, self.max_iterations = int(detector_slack), int(max_iterations)
        self.sic_mode = sic_mode
        if not 1 <= self.prefix_bits < min(self.payload_bits, 21):
            raise ValueError("Require 1 <= prefix bits < payload bits and <= 20 (explicit pattern bank).")
        if not 2 <= self.code_length <= self.n:
            raise ValueError("Polar mother-code length must fit in the channel block.")
        if self.detector_slack < 0 or self.max_iterations < 1 or self.sic_mode not in ("parallel", "successive"):
            raise ValueError("Invalid detector/SIC configuration.")
        if crc_bits <= 0:
            raise ValueError("ODMA requires a nonzero CRC to accept/cancel a decoded packet.")
        self.polar = PolarCode(self.payload_bits - self.prefix_bits, self.code_length, crc_bits, list_size)
        self.num_patterns = 1 << self.prefix_bits
        levels = np.asarray(power_levels, dtype=np.float64)
        if levels.ndim != 1 or not len(levels) or np.any(~np.isfinite(levels)) or np.any(levels <= 0):
            raise ValueError("Power levels must be a nonempty finite positive sequence.")
        self.power_levels = tuple(map(float, levels))
        group = np.minimum(np.arange(self.num_patterns) * len(levels) // self.num_patterns, len(levels) - 1)
        energy = levels[group]
        self.pattern_energy = energy / energy.mean()
        rng = np.random.default_rng(self.seed)
        self.positions = np.stack([np.sort(rng.choice(self.n, self.code_length, replace=False))
                                   for _ in range(self.num_patterns)]).astype(np.int32)

    def _prefix_indices(self, messages: np.ndarray) -> np.ndarray:
        weights = 1 << np.arange(self.prefix_bits - 1, -1, -1)
        return messages[:, :self.prefix_bits].astype(np.int64) @ weights

    def encode(self, messages: np.ndarray) -> np.ndarray:
        messages = np.asarray(messages)
        if messages.ndim != 2 or messages.shape[1] != self.payload_bits or np.any((messages != 0) & (messages != 1)):
            raise ValueError("Expected a binary [num_messages, payload_bits] array.")
        indices = self._prefix_indices(messages)
        coded = self.polar.encode(messages[:, self.prefix_bits:])
        symbols = (1.0 - 2.0 * coded) * np.sqrt(self.pattern_energy[indices, None] / self.code_length)
        result = np.zeros((len(messages), self.n), dtype=np.float64)
        result[np.arange(len(messages))[:, None], self.positions[indices]] = symbols
        return result

    def decode(self, y: np.ndarray, num_active: int, noise_var: float) -> tuple[np.ndarray, dict]:
        """Known activity count, real AWGN variance per coordinate, no known supports/messages."""
        y = np.asarray(y)
        if y.shape != (self.n,) or np.iscomplexobj(y) or not np.all(np.isfinite(y)):
            raise ValueError("This baseline expects a finite real Gaussian-MAC observation of shape [n].")
        if int(num_active) != num_active or num_active < 0 or not np.isfinite(noise_var) or noise_var < 0:
            raise ValueError("Activity must be a nonnegative integer; real noise variance must be finite and nonnegative.")
        started = time.perf_counter()
        residual = y.astype(np.float64, copy=True)
        decoded, seen, trace = [], set(), []
        trials = crc_passes = duplicate_passes = 0
        for iteration in range(self.max_iterations):
            remaining = int(num_active) - len(decoded)
            if remaining <= 0:
                break
            scores = np.abs(residual[self.positions]).sum(axis=1)
            candidates = np.argsort(-scores, kind="stable")[:min(self.num_patterns, remaining + self.detector_slack)]
            accepted = []
            for pattern in candidates:
                amplitude = np.sqrt(self.pattern_energy[pattern] / self.code_length)
                # Uniform independent placements give (remaining-1)*E/n interference variance per resource.
                unresolved = remaining - (len(accepted) if self.sic_mode == "successive" else 0)
                variance = max(float(noise_var) + max(0, unresolved - 1) / self.n, 1e-12)
                llr = 2.0 * amplitude * residual[self.positions[pattern]] / variance
                payloads, _ = self.polar.decode_list(llr)
                trials += 1
                if not len(payloads):
                    continue
                crc_passes += 1
                prefix = ((int(pattern) >> np.arange(self.prefix_bits - 1, -1, -1)) & 1).astype(np.uint8)
                message = np.concatenate((prefix, payloads[0]))
                key = message.tobytes()
                if key in seen:
                    duplicate_passes += 1
                    continue
                seen.add(key)
                accepted.append(message)
                if self.sic_mode == "successive":
                    residual -= self.encode(message[None, :])[0]
                if len(decoded) + len(accepted) >= num_active:
                    break
            if self.sic_mode == "parallel" and accepted:
                residual -= self.encode(np.stack(accepted)).sum(axis=0)
            decoded.extend(accepted)
            trace.append({"iteration": iteration + 1, "patterns_tested": len(candidates), "new_messages": len(accepted),
                          "residual_energy": float(residual @ residual)})
            if not accepted:
                break
        result = np.stack(decoded) if decoded else np.empty((0, self.payload_bits), dtype=np.uint8)
        return result, {"iterations": len(trace), "polar_trials": trials, "crc_passes": crc_passes,
                        "duplicate_crc_passes": duplicate_passes, "trace": trace, "list_size": len(result),
                        "runtime_seconds": time.perf_counter() - started, "oracle": "activity_count_only"}

    def metadata(self) -> dict:
        return {"name": "odma_polar", "payload_bits": self.payload_bits, "n": self.n, "seed": self.seed,
                "prefix_bits": self.prefix_bits, "code_length": self.code_length, "crc_bits": self.polar.crc_bits,
                "crc_polynomial": "MSB-first zero-initialized CRC-16/0x1021" if self.polar.crc_bits == 16 else "explicit small-B CRC",
                "list_size": self.polar.list_size, "detector_slack": self.detector_slack,
                "max_iterations": self.max_iterations, "sic_mode": self.sic_mode, "power_levels": self.power_levels,
                "energy_policy": "unit_per_codeword" if len(set(self.power_levels)) == 1 else "unit_average_message_energy",
                "variant": "small_payload_adaptation" if self.payload_bits != 100 else "paper_scale_independent_implementation",
                "provenance": "Ozates, Kazemi, Duman, IEEE WCL 2024, doi:10.1109/LWC.2024.3359270, Sec.III",
                "construction": "3GPP TS 38.212 reliability sequence; mother code without NR rate matching/interleaving",
                "receiver": "l1 pattern ranking; exact-LLR CRC-aided SCL; Gaussian TIN; iterative SIC",
                "complexity": "per round O(2^Bp*nc + K_candidates*list_size*nc*log(nc)); no 2^B payload search",
                "caveats": ["Independent implementation; published performance curves not yet reproduced.",
                            "Paper omits CRC polynomial, detector slack and exact TIN-variance rule; choices are explicit here.",
                            "Gaussian channel only; known activity and real-coordinate noise variance.",
                            "CRC-only acceptance can yield false positives; no truth-aided filtering or activity support oracle."]}
