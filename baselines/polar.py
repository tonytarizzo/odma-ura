"""Binary polar mother code with exact-LLR, CRC-aided successive-cancellation list decoding.

Reliability indices are the N <= 512 subset of 3GPP TS 38.212 Table 5.3.1.2-1.
This is not the full NR transport chain: no rate matching, input interleaver, or NR parity-check bits.
The decoder follows the ordinary f/g recursion with exact log-domain path metrics; it never enumerates payloads.
"""

from __future__ import annotations

import numpy as np


# Standardized numerical table, in ascending reliability (not source code from an external decoder).
_RELIABILITY = np.fromstring("""
0 1 2 4 8 16 32 3 5 64 9 6 17 10 18 128 12 33 65 20 256 34 24 36
7 129 66 11 40 68 130 19 13 48 14 72 257 21 132 35 258 26 80 37 25 22 136 260
264 38 96 67 41 144 28 69 42 49 74 272 160 288 192 70 44 131 81 50 73 15 320 133
52 23 134 384 76 137 82 56 27 97 39 259 84 138 145 261 29 43 98 88 140 30 146 71
262 265 161 45 100 51 148 46 75 266 273 104 162 53 193 152 77 164 268 274 54 83 57 112
135 78 289 194 85 276 58 168 139 99 86 60 280 89 290 196 141 101 147 176 142 321 31 200
90 292 322 263 149 102 105 304 296 163 92 47 267 385 324 208 386 150 153 165 106 55 328 113
154 79 269 108 224 166 195 270 275 291 59 169 114 277 156 87 197 116 170 61 281 278 177 293
388 91 198 172 120 201 336 62 282 143 103 178 294 93 202 323 392 297 107 180 151 209 284 94
204 298 400 352 325 155 210 305 300 109 184 115 167 225 326 306 157 329 110 117 212 171 330 226
387 308 216 416 271 279 158 337 118 332 389 173 121 199 179 228 338 312 390 174 393 283 122 448
353 203 63 340 394 181 295 285 232 124 205 182 286 299 354 211 401 185 396 344 240 206 95 327
402 356 307 301 417 213 186 404 227 418 302 360 111 331 214 309 188 449 217 408 229 159 420 310
333 119 339 218 368 230 391 313 450 334 233 175 123 341 220 314 424 395 355 287 183 234 125 342
316 241 345 452 397 403 207 432 357 187 236 126 242 398 346 456 358 405 303 244 189 361 215 348
419 406 464 362 409 219 311 421 410 231 248 369 190 364 335 480 315 221 370 422 425 451 235 412
343 372 317 222 426 453 237 433 347 243 454 318 376 428 238 359 457 399 434 349 245 458 363 127
191 407 436 465 246 350 460 249 411 365 440 374 423 466 250 371 481 413 366 468 429 252 373 482
427 414 223 472 455 377 435 319 484 430 488 239 378 459 437 380 461 496 351 467 438 251 462 442
441 469 247 367 253 375 444 470 483 415 485 473 474 254 379 431 489 486 476 439 490 463 381 497
492 443 382 498 445 471 500 446 475 487 504 255 477 491 478 383 493 499 502 494 501 447 505 506
479 508 495 503 507 509 510 511
""", dtype=np.int64, sep=" ")

_CRC_POLYNOMIALS = {0: 0, 4: 0x3, 8: 0x07, 12: 0x80F, 16: 0x1021}


def polar_transform(bits: np.ndarray) -> np.ndarray:
    """Return u F^{tensor log2(N)}, without bit reversal, over GF(2)."""
    result = np.asarray(bits, dtype=np.uint8).copy()
    n = result.shape[-1]
    if n < 2 or n & (n - 1):
        raise ValueError("Polar length must be a power of two >= 2.")
    width = 1
    while width < n:
        view = result.reshape(*result.shape[:-1], -1, 2 * width)
        view[..., :width] ^= view[..., width:]
        width *= 2
    return result


def append_crc(bits: np.ndarray, crc_bits: int) -> np.ndarray:
    """MSB-first polynomial division, zero initial register, no final XOR."""
    if crc_bits not in _CRC_POLYNOMIALS:
        raise ValueError(f"CRC length must be one of {tuple(_CRC_POLYNOMIALS)}.")
    bits = np.asarray(bits, dtype=np.uint8)
    if crc_bits == 0:
        return bits.copy()
    polynomial = (1 << crc_bits) | _CRC_POLYNOMIALS[crc_bits]
    taps = ((polynomial >> np.arange(crc_bits, -1, -1)) & 1).astype(np.uint8)
    work = np.pad(bits, [(0, 0)] * (bits.ndim - 1) + [(0, crc_bits)])
    for i in range(bits.shape[-1]):
        work[..., i:i + crc_bits + 1] ^= work[..., i:i + 1].copy() * taps
    return np.concatenate((bits, work[..., -crc_bits:]), axis=-1)


class PolarCode:
    def __init__(self, payload_bits: int, n: int, crc_bits: int = 16, list_size: int = 128):
        self.payload_bits, self.n = int(payload_bits), int(n)
        self.crc_bits, self.list_size = int(crc_bits), int(list_size)
        self.k = self.payload_bits + self.crc_bits
        if self.n < 2 or self.n > 512 or self.n & (self.n - 1):
            raise ValueError("Supported mother-code lengths are powers of two in [2, 512].")
        if not 1 <= self.k <= self.n or self.payload_bits < 1 or self.list_size < 1:
            raise ValueError("Require 1 <= payload+CRC <= polar length and a positive list size.")
        if self.crc_bits not in _CRC_POLYNOMIALS:
            raise ValueError("Unsupported CRC polynomial length.")
        reliability = _RELIABILITY[_RELIABILITY < self.n]
        self.information = np.sort(reliability[-self.k:])
        self.frozen = np.ones(self.n, dtype=bool)
        self.frozen[self.information] = False

    def encode(self, messages: np.ndarray) -> np.ndarray:
        messages = np.asarray(messages)
        if messages.ndim < 1 or messages.shape[-1] != self.payload_bits or np.any((messages != 0) & (messages != 1)):
            raise ValueError("Expected binary payloads with the configured final dimension.")
        u = np.zeros((*messages.shape[:-1], self.n), dtype=np.uint8)
        u[..., self.information] = append_crc(messages, self.crc_bits)
        return polar_transform(u)

    def decode_list(self, llr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CRC-valid payload candidates in increasing path metric; empty means CRC failure."""
        llr = np.asarray(llr, dtype=np.float64)
        if llr.shape != (self.n,) or not np.all(np.isfinite(llr)):
            raise ValueError("Expected one finite LLR per coded bit.")

        def visit(alpha, metrics, start):
            width, paths = alpha.shape[1], alpha.shape[0]
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
