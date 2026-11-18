"""Rate-zero plus repetition subtree collapse, exact in real arithmetic.

Direct metric accumulation can change floating-point pruning near ties. Encoding, CRC and the
ordinary stable branch order remain unchanged; no general finite-precision equivalence is claimed.
"""

import numpy as np

from baselines.polar import append_crc
from framework.polar_fast_scl import RateZeroPolarCode


def _repetition_update(alpha, metrics, list_size):
    both = np.empty((len(metrics), 2), dtype=np.float64)
    both[:, 0] = metrics + np.logaddexp(0., -alpha).sum(axis=1)
    both[:, 1] = metrics + np.logaddexp(0., alpha).sum(axis=1)
    chosen = np.argsort(both.ravel(), kind="stable")[:list_size]
    bits = (chosen % 2).astype(np.uint8)
    beta = np.broadcast_to(bits[:, None], (len(bits), alpha.shape[1])).copy()
    u = np.zeros_like(beta)
    u[:, -1] = bits
    return beta, u, both.ravel()[chosen], chosen // 2


class RepetitionPolarCode(RateZeroPolarCode):
    def __init__(self, payload_bits, n, crc_bits=16, list_size=128):
        super().__init__(payload_bits, n, crc_bits, list_size)
        self._repetition = np.zeros(2 * self.n, dtype=bool)
        width = 2
        while width <= self.n:
            for start in range(0, self.n, width):
                frozen = self.frozen[start:start + width]
                self._repetition[(self.n + start) // width] = frozen[:-1].all() and not frozen[-1]
            width *= 2

    def decode_list(self, llr):
        llr = np.asarray(llr, dtype=np.float64)
        if llr.shape != (self.n,) or not np.all(np.isfinite(llr)):
            raise ValueError("Expected one finite LLR per coded bit.")

        def visit(alpha, metrics, start):
            width, paths = alpha.shape[1], alpha.shape[0]
            if width > 1 and self._rate_zero[(self.n + start) // width]:
                bits = np.zeros((paths, width), dtype=np.uint8)
                return bits, bits, metrics + np.logaddexp(0., -alpha).sum(axis=1), np.arange(paths)
            if width > 1 and self._repetition[(self.n + start) // width]:
                return _repetition_update(alpha, metrics, self.list_size)
            if width == 1:
                if self.frozen[start]:
                    bits = np.zeros((paths, 1), dtype=np.uint8)
                    return bits, bits, metrics + np.logaddexp(0.0, -alpha[:, 0]), np.arange(paths)
                both = metrics[:, None] + np.logaddexp(0.0, -alpha * np.array([1.0, -1.0]))
                chosen = np.argsort(both.ravel(), kind="stable")[:self.list_size]
                bits = (chosen % 2).astype(np.uint8)[:, None]
                return bits, bits, both.ravel()[chosen], chosen // 2
            half = width // 2
            a, b = alpha[:, :half], alpha[:, half:]
            f = np.logaddexp(0.0, a + b) - np.logaddexp(a, b)
            left, left_u, left_metrics, origins = visit(f, metrics, start)
            a, b = a[origins], b[origins]
            right_alpha = b + (1.0 - 2.0 * left) * a
            right, right_u, final_metrics, right_origins = visit(right_alpha, left_metrics, start + half)
            beta = np.concatenate((left[right_origins] ^ right, right), axis=1)
            u = np.concatenate((left_u[right_origins], right_u), axis=1)
            return beta, u, final_metrics, origins[right_origins]

        _, u, metrics, _ = visit(llr[None, :], np.zeros(1), 0)
        data = u[:, self.information]
        payload = data[:, :self.payload_bits]
        valid = np.all(append_crc(payload, self.crc_bits) == data, axis=1)
        selected = np.flatnonzero(valid)
        selected = selected[np.argsort(metrics[selected], kind="stable")]
        return payload[selected], metrics[selected]
