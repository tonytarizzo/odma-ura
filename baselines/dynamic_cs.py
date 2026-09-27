"""Nassaji--Truhachev dynamic CS: first-slot AMP, modulated AMP, CRC assembly and SIC.

Independent implementation of E. Nassaji's thesis, Chapter 9, equations (9.3)--(9.10).
The paper-scale profile is not a reproduced performance curve: several decoder tuning
parameters and the CRC polynomial are unspecified in the available primary source.
"""

from itertools import combinations, permutations, product
from math import ceil, factorial, log2

import numpy as np
from scipy.special import expit


SOURCE = "https://dalspace.library.dal.ca/server/api/core/bitstreams/78b29db0-2fc0-4c8a-8275-b0f8f7fbdae3/content"


def _integer(bits):
    return int(np.asarray(bits, dtype=np.uint64) @ (1 << np.arange(len(bits) - 1, -1, -1, dtype=np.uint64)))


def _bits(value, width):
    return ((int(value) >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)


def _crc(bits, width, polynomial):
    if not width:
        return np.empty(0, dtype=np.uint8)
    work = np.r_[bits, np.zeros(width, dtype=np.uint8)].copy()
    divisor = _bits(polynomial, width + 1)
    for i in range(len(bits)):
        if work[i]:
            work[i:i + width + 1] ^= divisor
    return work[-width:]


class _SignBank:
    """Counter-generated Rademacher entries; prefixes define the shared base matrix."""

    def __init__(self, rows, columns, seed, cache_bytes=16_000_000):
        self.rows, self.columns, self.seed = int(rows), int(columns), int(seed)
        self._cache = None
        if rows * columns * 4 <= cache_bytes:
            self._cache = self._generate(np.arange(columns))

    def _generate(self, columns):
        row = np.arange(self.rows, dtype=np.uint64)[:, None]
        column = np.asarray(columns, dtype=np.uint64)[None, :]
        with np.errstate(over="ignore"):
            z = row * np.uint64(0xD2B74407B1CE6E93) + column * np.uint64(0x9E3779B97F4A7C15)
            z = z + np.uint64(self.seed) + np.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
            z ^= z >> 31
        return (2 * (z >> 63).astype(np.float32) - 1)

    def columns_at(self, indices, rows=None):
        matrix = self._generate(indices) if self._cache is None else self._cache[:, indices]
        return matrix if rows is None else matrix[:rows]

    def multiply(self, x, rows, columns, transpose=False):
        x = np.asarray(x, dtype=np.float64)
        vector = x.ndim == 1
        if vector:
            x = x[:, None]
        if self._cache is not None:
            matrix = self._cache[:rows, :columns]
            result = (matrix.T @ x if transpose else matrix @ x) / np.sqrt(rows)
        else:
            result = np.zeros((columns if transpose else rows, x.shape[1]))
            for start in range(0, columns, 256):
                stop = min(columns, start + 256)
                matrix = self.columns_at(np.arange(start, stop), rows) / np.sqrt(rows)
                if transpose:
                    result[start:stop] = matrix.T @ x
                else:
                    result += matrix @ x[start:stop]
        return result[:, 0] if vector else result


class DynamicCSBaseline:
    """Dynamic-CS native receiver; small-payload and native-dimension profiles are explicit.

    For S=1 every transmitted word has exactly unit energy. S>1 superposes streams:
    the nominal energy is one, but individual word energies differ. No per-word
    projection is applied, because that would change the receiver's forward model.
    """

    def __init__(self, payload_bits, n, seed=0, *, profile="small", prefix_bits=None, data_bits=None, streams=None,
                 crc_bits=None, crc_polynomial=None, first_length=None, slot_lengths=None, amp_iterations=40,
                 global_iterations=2, list_size=None, detection_threshold=2.0, fa_threshold=0.0, damping=1.0,
                 candidate_budget=None, assembly_budget=100_000, cache_bytes=16_000_000):
        self.payload_bits, self.n, self.seed = int(payload_bits), int(n), int(seed)
        self.profile = profile
        if profile not in {"small", "native100"}:
            raise ValueError("profile must be 'small' or 'native100'")
        if self.payload_bits < 3 or self.n < 8 or self.seed < 0:
            raise ValueError("Require B>=3, n>=8 and nonnegative seed")
        if profile == "native100" and (payload_bits != 100 or n != 30000):
            raise ValueError("native100 profile requires B=100,n=30000")
        self.streams = int(streams if streams is not None else (2 if profile == "native100" else 1))
        self.prefix_bits = int(prefix_bits if prefix_bits is not None else (15 if profile == "native100" else min(8, payload_bits - 2)))
        self.crc_bits = int(crc_bits if crc_bits is not None else (5 if profile == "native100" else 4))
        self.crc_polynomial = int(crc_polynomial if crc_polynomial is not None else ((1 << self.crc_bits) | 3))
        if self.streams not in {1, 2, 3} or not 0 < self.prefix_bits < payload_bits or not 0 <= self.crc_bits <= 16:
            raise ValueError("Require S in {1,2,3}, 0<prefix_bits<B, 0<=crc_bits<=16")
        if self.crc_bits and (self.crc_polynomial.bit_length() != self.crc_bits + 1 or not self.crc_polynomial & 1):
            raise ValueError("CRC polynomial must have the declared degree and a nonzero constant term")
        remaining = self.payload_bits + self.crc_bits - self.prefix_bits
        if data_bits is None:
            data_bits = [30, 30, 30] if profile == "native100" else [remaining // 2, remaining - remaining // 2]
        self.data_bits = tuple(int(x) for x in data_bits)
        if sum(self.data_bits) != remaining or any(x <= 0 or x % self.streams for x in self.data_bits):
            raise ValueError("data_bits must sum to B+CRC-prefix_bits and each be divisible by S")
        self.order_width = ceil(log2(factorial(self.streams)))
        self.orders = tuple(permutations(range(self.streams)))
        self.header_bits = self.prefix_bits + len(self.data_bits) * self.order_width
        if self.header_bits > 24 or max(self.data_bits) // self.streams > 20:
            raise ValueError("This receiver limits header and per-stream alphabets to 24 and 20 bits")
        if slot_lengths is None:
            first = int(first_length if first_length is not None else (3000 if profile == "native100" else max(8, n // 2)))
            quotient, remainder = divmod(n - first, len(self.data_bits))
            slot_lengths = [first] + [quotient + (i < remainder) for i in range(len(self.data_bits))]
        self.slot_lengths = tuple(int(x) for x in slot_lengths)
        if len(self.slot_lengths) != 1 + len(self.data_bits) or sum(self.slot_lengths) != n or min(self.slot_lengths) <= 0:
            raise ValueError("slot_lengths must be positive, one per slot, and sum to n")
        self.amp_iterations, self.global_iterations = int(amp_iterations), int(global_iterations)
        self.list_size = int(list_size if list_size is not None else self.streams + (self.crc_bits > 0))
        self.detection_threshold, self.fa_threshold = float(detection_threshold), float(fa_threshold)
        self.damping, self.candidate_budget, self.assembly_budget = float(damping), candidate_budget, int(assembly_budget)
        if min(self.amp_iterations, self.global_iterations, self.assembly_budget) < 1 or self.list_size < self.streams:
            raise ValueError("Iteration/budget counts must be positive and list_size>=S")
        if not 0 < self.damping <= 1 or candidate_budget is not None and candidate_budget < 1:
            raise ValueError("Require 0<damping<=1 and positive candidate_budget when supplied")
        self._offsets = np.cumsum((0,) + self.slot_lengths)
        # Preserve the source's equal per-stream amplitudes, including the header.
        # This is nominal energy: repeated substreams and cross terms make S>1 norms vary.
        self.nominal_raw_energy = self.slot_lengths[0] + self.streams * sum(self.slot_lengths[1:])
        self._a = _SignBank(self.slot_lengths[0], 1 << self.header_bits, seed + 17011, cache_bytes)
        self._b = _SignBank(max(self.slot_lengths[1:]), 1 << (max(self.data_bits) // self.streams), seed + 34019, cache_bytes)

    def _parts(self, message):
        message = np.asarray(message, dtype=np.uint8)
        augmented = np.r_[message, _crc(message, self.crc_bits, self.crc_polynomial)]
        position, values, order_bits = self.prefix_bits, [], []
        for width in self.data_bits:
            chunk = augmented[position:position + width].reshape(self.streams, width // self.streams)
            indices = [_integer(x) for x in chunk]
            values.append(indices)
            if self.order_width:
                permutation = tuple(np.argsort(indices, kind="stable"))
                order_bits.extend(_bits(self.orders.index(permutation), self.order_width))
            position += width
        header = _integer(np.r_[augmented[:self.prefix_bits], np.asarray(order_bits, dtype=np.uint8)])
        return header, values

    def encode(self, messages):
        messages = np.asarray(messages)
        if messages.ndim != 2 or messages.shape[1] != self.payload_bits or not np.isin(messages, [0, 1]).all():
            raise ValueError("messages must be binary [K,B]")
        out = np.zeros((len(messages), self.n))
        for i, message in enumerate(messages):
            header, values = self._parts(message)
            signature = self._a.columns_at([header])[:, 0]
            out[i, :self.slot_lengths[0]] = signature / np.sqrt(self.nominal_raw_energy)
            for slot, indices in enumerate(values, 1):
                length = self.slot_lengths[slot]
                modulation = np.resize(signature, length)
                waveform = modulation * self._b.columns_at(indices, length).sum(axis=1)
                out[i, self._offsets[slot]:self._offsets[slot + 1]] = waveform / np.sqrt(self.nominal_raw_energy)
        return out

    def _amp(self, y, bank, columns, amplitude, prior, noise_var, signatures=None):
        length = len(y)
        shape = (columns,) if signatures is None else (columns, signatures.shape[1])
        estimate = np.zeros(shape)
        residual = y.copy() if signatures is None else y[:, None] * signatures
        for iteration in range(self.amp_iterations):
            variance = np.maximum(np.mean(residual * residual, axis=0), max(noise_var, 1e-12))
            observation = bank.multiply(residual, length, columns, transpose=True) + estimate
            llr = (amplitude * observation - 0.5 * amplitude ** 2) / variance
            probability = expit(llr + np.log(prior) - np.log1p(-prior))
            updated = amplitude * probability
            derivative = amplitude ** 2 * probability * (1 - probability) / variance
            reconstruction = bank.multiply(updated, length, columns)
            if signatures is None:
                next_residual = y - reconstruction + residual * derivative.sum() / length
            else:
                common = y - np.sum(signatures * reconstruction, axis=1)
                # Eq. (9.10), with the arithmetic mean taken down each user column.
                next_residual = common[:, None] * signatures + residual * derivative.sum(axis=0) / length
            estimate = self.damping * updated + (1 - self.damping) * estimate
            residual = self.damping * next_residual + (1 - self.damping) * residual
            if not np.isfinite(residual).all():
                raise FloatingPointError("Dynamic-CS AMP diverged; do not silently substitute another receiver")
        variance = np.maximum(np.mean(residual * residual, axis=0), max(noise_var, 1e-12))
        observation = bank.multiply(residual, length, columns, transpose=True) + estimate
        return (amplitude * observation - 0.5 * amplitude ** 2) / variance

    def _assemble(self, header, slot_llrs):
        header_bits = _bits(header, self.header_bits)
        choices = []
        for slot, llrs in enumerate(slot_llrs):
            count = min(self.list_size, len(llrs))
            top = np.argsort(llrs)[-count:][::-1]
            # The source's Bernoulli decoder selects S distinct entries. Repeated
            # substreams are a modelling failure, not extra copies of the top LLR.
            choices.append([(tuple(sorted(indices)), float(llrs[list(indices)].sum()))
                            for indices in combinations(top, self.streams)])
        size = int(np.prod([len(x) for x in choices], dtype=np.int64))
        if size > self.assembly_budget:
            raise ValueError(f"CRC assembly requires {size} paths; increase assembly_budget explicitly")
        best, best_score, tested = None, -np.inf, 0
        for path in product(*choices):
            assembled = list(header_bits[:self.prefix_bits])
            score, invalid = 0., False
            for slot, (indices, value) in enumerate(path):
                if self.order_width:
                    start = self.prefix_bits + slot * self.order_width
                    order = _integer(header_bits[start:start + self.order_width])
                    if order >= len(self.orders):
                        invalid = True
                        break
                    ordered = np.empty(self.streams, dtype=int)
                    ordered[list(self.orders[order])] = indices
                else:
                    ordered = indices
                for index in ordered:
                    assembled.extend(_bits(index, self.data_bits[slot] // self.streams))
                score += value
            if invalid:
                continue
            tested += 1
            candidate = np.asarray(assembled, dtype=np.uint8)
            message = candidate[:self.payload_bits]
            if not np.array_equal(candidate[self.payload_bits:], _crc(message, self.crc_bits, self.crc_polynomial)):
                continue
            if self._parts(message)[0] == header and score > best_score:
                best, best_score = message, score
        return best, best_score, tested

    def decode(self, y, num_active, noise_var):
        y = np.asarray(y)
        if (y.shape != (self.n,) or np.iscomplexobj(y) or not np.isfinite(y).all()
                or int(num_active) != num_active or num_active < 0 or not np.isfinite(noise_var) or noise_var < 0):
            raise ValueError("Require finite real y[n], K>=0 and noise_var>=0")
        y = y.astype(np.float64)
        if num_active == 0:
            return np.empty((0, self.payload_bits), np.uint8), {"header_candidates": 0, "decoded": 0}
        length = self.slot_lengths[0]
        prior = -np.expm1(num_active * np.log1p(-1 / (1 << self.header_bits)))
        llrs = self._amp(y[:length], self._a, 1 << self.header_bits, np.sqrt(length / self.nominal_raw_energy),
                         np.clip(prior, 1e-12, 1 - 1e-12), noise_var)
        headers = np.flatnonzero(llrs > self.detection_threshold)
        before_cap = len(headers)
        if self.candidate_budget is not None and len(headers) > self.candidate_budget:
            headers = headers[np.argsort(llrs[headers])[-self.candidate_budget:]]
        initial_headers = headers.copy()
        residual, recovered, scores = y.copy(), [], []
        paths, fa_removed, iterations = 0, 0, 0
        for iteration in range(self.global_iterations):
            if not len(headers):
                break
            iterations += 1
            signature = self._a.columns_at(headers)
            evidence = []
            for slot, width in enumerate(self.data_bits, 1):
                length = self.slot_lengths[slot]
                signs = signature[np.arange(length) % self.slot_lengths[0]]
                alphabet = 1 << (width // self.streams)
                signal = residual[self._offsets[slot]:self._offsets[slot + 1]]
                evidence.append(self._amp(signal, self._b, alphabet, np.sqrt(length / self.nominal_raw_energy),
                                          min(self.streams / alphabet, 1 - 1e-12), noise_var, signs))
            fa_score = sum(x.max(axis=0) for x in evidence)
            keep = fa_score >= self.fa_threshold
            fa_removed += int((~keep).sum())
            next_headers, accepted = [], []
            for index, header in enumerate(headers):
                if not keep[index]:
                    continue
                message, score, tested = self._assemble(int(header), [x[:, index] for x in evidence])
                paths += tested
                if message is None:
                    next_headers.append(header)
                else:
                    recovered.append(message)
                    scores.append(score)
                    accepted.append(message)
            if accepted:
                residual -= self.encode(np.asarray(accepted)).sum(axis=0)
            headers = np.asarray(next_headers, dtype=int)
            # FA removal also changes A': rerun matrix AMP even without a CRC/SIC success.
            if not accepted and keep.all():
                break
        decoded = np.asarray(recovered, dtype=np.uint8).reshape(-1, self.payload_bits)
        if len(decoded) > num_active:
            decoded = decoded[np.argsort(scores)[-num_active:]]
        return decoded, {"header_candidates": len(initial_headers), "header_candidates_before_cap": before_cap,
                         "decoded": len(decoded), "crc_paths_tested": paths, "fa_removed": fa_removed,
                         "global_iterations": iterations, "candidate_cap_applied": before_cap > len(initial_headers),
                         "native_receiver": True, "oracle_candidates": False}

    def metadata(self):
        return {"name": "dynamic_cs", "source": SOURCE, "doi": "10.1109/LCOMM.2024.3403501",
                "variant": f"independent_dynamic_cs_{self.profile}", "paper_curve_reproduced": False,
                "payload_bits": self.payload_bits, "n": self.n, "prefix_bits": self.prefix_bits,
                "header_bits": self.header_bits, "data_bits": list(self.data_bits), "streams": self.streams,
                "slot_lengths": list(self.slot_lengths), "crc_bits": self.crc_bits,
                "nominal_raw_energy": self.nominal_raw_energy, "power_allocation": "equal amplitude per substream",
                "crc_polynomial": self.crc_polynomial, "amp_iterations": self.amp_iterations,
                "global_iterations": self.global_iterations, "list_size": self.list_size,
                "detection_threshold": self.detection_threshold, "fa_threshold": self.fa_threshold,
                "damping": self.damping, "candidate_budget": self.candidate_budget,
                "energy_policy": "exact_unit" if self.streams == 1 else "nominal_unit_multistream_not_peak_constrained",
                "caveats": ["Independent implementation, not author code or validated paper-curve reproduction.",
                            "CRC polynomial, thresholds and iteration budgets are exposed implementation choices.",
                            "Bernoulli AMP approximates repeated headers and repeated within-slot substreams.",
                            "More than one message sharing a header cannot generally be recovered by one header list.",
                            "S>1 waveform norms vary; per-word normalization would change the stated AMP model.",
                            "The small-payload profile is an adaptation, not the native B=100 operating point."]}
