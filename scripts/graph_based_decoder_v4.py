"""
ODMA + URA Decoder Testbed — V4: Full MU-MIMO with per-user Rayleigh fading (no CSI)
=====================================================================================
True physical uplink model — no channel inversion, no CSI at the devices.

Signal model:
    Y = sum_u (P_{b_u} c_{m_u}) h_u^T + Z       ∈ C^{n × M}

  - Global codebook of unit-norm codewords, shape (num_codewords, d).
  - Each global message index m deterministically selects codebook[m]
    and ODMA pattern msg_to_block[m].
  - P_b ∈ {0,1}^{n×d}, P_b^T P_b = I_d.
  - Each active device u independently picks a message m_u, draws an
    i.i.d. Rayleigh fading channel h_u ~ CN(0, I_M) (not normalized),
    and contributes a rank-1 term (P_{b_u} c_{m_u}) h_u^T.
  - Z is i.i.d. complex circular AWGN (or real AWGN in real-valued mode).
  - Unlike V1/V2/V3, the physical signal depends on raw per-user channels
    with no pre-equalization.  This is the hardest decoding scenario.

Decoder target:
    message_counts ∈ Z_+^{num_codewords}.  The decoder must recover the
    multiset of transmitted messages.  User identities and user channels
    are NOT part of the decoder output.

Run:
    python graph_based_decoder_v4.py --seed 42 --n 128 --d 16 --num-blocks 8 \
        --num-codewords 64 --num-devices-active 10 --num-antennas 4 --esn0-db 10
"""

from __future__ import annotations
import argparse
import numpy as np


def make_codebook(num_codewords: int, d: int, rng: np.random.Generator, complex_valued: bool = False) -> np.ndarray:
    """Random Gaussian codebook with unit-normalized rows.  Returns (num_codewords, d)."""
    if complex_valued:
        raw = rng.standard_normal((num_codewords, d)) + 1j * rng.standard_normal((num_codewords, d))
    else:
        raw = rng.standard_normal((num_codewords, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def make_odma_blocks(num_blocks: int, n: int, d: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Random ODMA blocks. Each block is a sorted array of d resource indices drawn without replacement from {0, ..., n-1}."""
    return [np.sort(rng.choice(n, size=d, replace=False)) for _ in range(num_blocks)]


def build_pattern_matrices(blocks: list[np.ndarray], n: int) -> dict[int, np.ndarray]:
    """Build per-block embedding matrices P_b ∈ {0,1}^{n×d}. Column j of P_b is e_{S_b[j]}. Satisfies P_b^T P_b = I_d."""
    P_mats: dict[int, np.ndarray] = {}
    for b, S_b in enumerate(blocks):
        d = len(S_b)
        P = np.zeros((n, d), dtype=np.float64)
        P[S_b, np.arange(d)] = 1.0
        P_mats[b] = P
    return P_mats


def make_message_block_mapping(num_codewords: int, num_blocks: int):
    """Deterministic mapping: message m → pattern msg_to_block[m] = m % num_blocks."""
    msg_to_block: dict[int, int] = {}
    block_to_msg_list: dict[int, list[int]] = {b: [] for b in range(num_blocks)}
    for m in range(num_codewords):
        b = m % num_blocks
        msg_to_block[m] = b
        block_to_msg_list[b].append(m)
    return msg_to_block, block_to_msg_list


def sample_active_messages(num_devices_active: int, num_codewords: int, rng: np.random.Generator) -> np.ndarray:
    """Each active device independently picks a message index uniformly at random. Duplicates are allowed."""
    return rng.integers(0, num_codewords, size=num_devices_active)


def build_message_counts(active_msgs: np.ndarray, num_codewords: int) -> np.ndarray:
    """Global message count vector — the direct decoder target.  Returns (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for m in active_msgs:
        counts[int(m)] += 1.0
    return counts


def build_block_coefficients(active_msgs: np.ndarray, msg_to_block: dict[int, int], block_to_msg_list: dict[int, list[int]], num_blocks: int) -> dict[int, np.ndarray]:
    """Blockwise view of the decoder target: for each block b, a sparse vector of message multiplicities."""
    coeffs: dict[int, np.ndarray] = {}
    for b in range(num_blocks):
        msg_list = block_to_msg_list[b]
        a_b = np.zeros(len(msg_list), dtype=np.float64)
        msg_to_local = {m: i for i, m in enumerate(msg_list)}
        for m in active_msgs:
            if msg_to_block[m] == b:
                a_b[msg_to_local[m]] += 1.0
        coeffs[b] = a_b
    return coeffs


def build_block_dictionaries(codebook: np.ndarray, block_to_msg_list: dict[int, list[int]], num_blocks: int) -> dict[int, np.ndarray]:
    """Gather global codewords for each block's assigned messages.  Returns dict block_idx -> (L_b, d)."""
    return {b: codebook[block_to_msg_list[b]] for b in range(num_blocks)}


def make_user_channels(num_users: int, M: int, rng: np.random.Generator, complex_valued: bool = False) -> np.ndarray:
    """I.i.d. Rayleigh fading channel per active device. Complex: h_u ~ CN(0, I_M), not normalized. Real: h_u ~ N(0, I_M).  Returns (num_users, M)."""
    if complex_valued:
        return (rng.standard_normal((num_users, M)) + 1j * rng.standard_normal((num_users, M))) / np.sqrt(2)
    return rng.standard_normal((num_users, M))


def esn0_db_to_noise_var(esn0_db: float, d: int) -> float:
    """Convert Es/N0 (dB) to per-entry per-antenna noise variance. Es = 1/d (unit-norm codewords). N0 = 1/(d * esn0_lin)."""
    esn0_lin = 10.0 ** (esn0_db / 10.0)
    return 1.0 / (d * esn0_lin)


def synthesize_received_signal(active_msgs: np.ndarray, msg_to_block: dict[int, int], P_mats: dict[int, np.ndarray], codebook: np.ndarray, user_channels: np.ndarray, noise_var: float, rng: np.random.Generator, complex_valued: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize Y = sum_u (P_{b_u} c_{m_u}) h_u^T + Z.  Per-user fading channels.

    Each active device u contributes a rank-1 term weighted by its own channel h_u.
    The physical signal depends on individual channels, not just on message multiplicities.

    Returns:  (Y_noisy, Y_clean)  both (n, M)
    """
    n = next(iter(P_mats.values())).shape[0]
    M = user_channels.shape[1]
    dtype = np.complex128 if complex_valued else np.float64
    Y_clean = np.zeros((n, M), dtype=dtype)

    for u in range(len(active_msgs)):
        m_u = int(active_msgs[u])
        P_u = P_mats[msg_to_block[m_u]]       # (n, d)
        c_u = codebook[m_u]                    # (d,)
        h_u = user_channels[u]                 # (M,)
        s_u = P_u @ c_u                        # (n,)  embedded codeword
        Y_clean += np.outer(s_u, h_u)

    if complex_valued:
        noise = np.sqrt(noise_var / 2) * (rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M)))
    else:
        noise = np.sqrt(noise_var) * rng.standard_normal((n, M))

    return Y_clean + noise, Y_clean


def placeholder_decoder(Y: np.ndarray, P_mats: dict[int, np.ndarray], block_dicts: dict[int, np.ndarray], noise_var: float, *, max_iter: int = 50, damping: float = 0.3) -> tuple[dict[int, np.ndarray], dict]:
    """Placeholder for the iterative graph-based decoder.

    TODO — implement two-step iterative message passing:
        Step 1: resource-node local MMSE / Gaussian belief propagation
        Step 2: blockwise sparse coefficient posterior inference / denoiser
        Iterate until convergence or max_iter.

    Returns:  (coeffs_hat, meta)
        coeffs_hat : dict  block_idx -> estimated multiplicity vector (L_b,)
        meta       : dict  convergence info
    """
    coeffs_hat: dict[int, np.ndarray] = {}
    for b in P_mats:
        L_b = block_dicts[b].shape[0]
        coeffs_hat[b] = np.zeros(L_b)
    meta = {"converged": False, "iterations": 0, "note": "placeholder — not implemented"}
    return coeffs_hat, meta


def assemble_global_counts(block_coeffs: dict[int, np.ndarray], block_to_msg_list: dict[int, list[int]], num_codewords: int) -> np.ndarray:
    """Convert blockwise coefficient vectors back to a single global message count vector (num_codewords,)."""
    counts = np.zeros(num_codewords, dtype=np.float64)
    for b, a_b in block_coeffs.items():
        for local_idx, global_msg in enumerate(block_to_msg_list[b]):
            counts[global_msg] = a_b[local_idx]
    return counts


def evaluate_counts(counts_true: np.ndarray, counts_hat: np.ndarray) -> dict:
    """Compare true vs estimated global message count vectors. Returns l1, mse, support counts."""
    diff = counts_true - counts_hat
    return dict(
        l1=float(np.sum(np.abs(diff))),
        mse=float(np.mean(diff ** 2)),
        support_true=int(np.count_nonzero(counts_true)),
        support_hat=int(np.count_nonzero(counts_hat)),
    )


def print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, user_channels, message_counts, block_coeffs, Y_clean, Y_noisy, noise_var, esn0_db):
    num_blocks = len(P_mats)
    n, d = next(iter(P_mats.values())).shape
    M = Y_noisy.shape[1]
    K = len(active_msgs)

    print("\n" + "=" * 60)
    print("ODMA + URA Simulation — V4 (per-user Rayleigh fading, no CSI)")
    print("=" * 60)

    print(f"\nCodebook          : {codebook.shape}  (num_codewords × d)")
    print(f"Resource grid     : n = {n}")
    print(f"Block size        : d = {d}")
    print(f"Num blocks        : {num_blocks}")
    print(f"Pattern matrices  : {num_blocks} × ({n}, {d})")
    print(f"Num antennas      : M = {M}")
    print(f"Active devices    : K = {K}")
    print(f"Es/N0             : {esn0_db:.1f} dB")
    print(f"Noise variance    : {noise_var:.6f}")
    print(f"User channels     : {user_channels.shape}  (K × M)")

    print(f"\nActive messages (per user) : {active_msgs.tolist()}")
    active_blocks = [msg_to_block[int(m)] for m in active_msgs]
    print(f"Active blocks (per user)  : {active_blocks}")

    num_unique = int(np.count_nonzero(message_counts))
    max_mult = int(message_counts.max())
    print(f"\nDecoder target (global message counts):")
    print(f"  Unique messages    : {num_unique} / {len(message_counts)}")
    print(f"  Max multiplicity   : {max_mult}")
    active_indices = np.nonzero(message_counts)[0]
    print(f"  Active msg → count : {dict(zip(active_indices.tolist(), message_counts[active_indices].astype(int).tolist()))}")

    print(f"\nPer-block coefficient summary:")
    for b in range(num_blocks):
        a_b = block_coeffs[b]
        nnz = int(np.count_nonzero(a_b))
        total = int(a_b.sum())
        max_mult_b = int(a_b.max()) if nnz > 0 else 0
        print(f"  block {b:3d}: {len(block_to_msg_list[b]):4d} msgs, nnz={nnz:3d}, total_users={total:3d}, max_mult={max_mult_b}")

    supports = {b: np.any(P_mats[b], axis=1) for b in range(num_blocks)}
    overlap_counts = []
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            ov = int(np.sum(supports[i] & supports[j]))
            if ov > 0:
                overlap_counts.append((i, j, ov))
    resource_usage = np.zeros(n, dtype=int)
    for b in range(num_blocks):
        resource_usage += supports[b].astype(int)

    print(f"\nBlock overlap statistics:")
    print(f"  Pairs with overlap : {len(overlap_counts)} / {num_blocks * (num_blocks - 1) // 2}")
    if overlap_counts:
        ovs = [c for _, _, c in overlap_counts]
        print(f"  Overlap sizes      : min={min(ovs)}, max={max(ovs)}, mean={np.mean(ovs):.1f}")
    print(f"  Max blocks/resource: {resource_usage.max()}")
    print(f"  Resources unused   : {int((resource_usage == 0).sum())} / {n}")

    print(f"\nReceived signal:")
    print(f"  Y shape           : {Y_noisy.shape}")
    print(f"  ||Y_clean||_F     : {np.linalg.norm(Y_clean):.4f}")
    print(f"  ||Y_noisy||_F     : {np.linalg.norm(Y_noisy):.4f}")
    print(f"  ||noise||_F       : {np.linalg.norm(Y_noisy - Y_clean):.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ODMA + URA scaffold — V4 (per-user Rayleigh fading, no CSI)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=128, help="total resource grid size")
    parser.add_argument("--d", type=int, default=16, help="codeword length / block size")
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-codewords", type=int, default=64)
    parser.add_argument("--num-devices-active", type=int, default=10)
    parser.add_argument("--num-antennas", type=int, default=4)
    parser.add_argument("--esn0-db", type=float, default=10.0, help="Es/N0 in dB (active-symbol energy to noise spectral density)")
    parser.add_argument("--complex-valued", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    codebook = make_codebook(args.num_codewords, args.d, rng, complex_valued=args.complex_valued)
    blocks = make_odma_blocks(args.num_blocks, args.n, args.d, rng)
    P_mats = build_pattern_matrices(blocks, args.n)
    msg_to_block, block_to_msg_list = make_message_block_mapping(args.num_codewords, args.num_blocks)
    block_dicts = build_block_dictionaries(codebook, block_to_msg_list, args.num_blocks)

    active_msgs = sample_active_messages(args.num_devices_active, args.num_codewords, rng)
    user_channels = make_user_channels(args.num_devices_active, args.num_antennas, rng, complex_valued=args.complex_valued)
    message_counts = build_message_counts(active_msgs, args.num_codewords)
    block_coeffs = build_block_coefficients(active_msgs, msg_to_block, block_to_msg_list, args.num_blocks)
    noise_var = esn0_db_to_noise_var(args.esn0_db, args.d)

    # Y = sum_u (P_{b_u} c_{m_u}) h_u^T + Z
    Y_noisy, Y_clean = synthesize_received_signal(active_msgs, msg_to_block, P_mats, codebook, user_channels, noise_var, rng, complex_valued=args.complex_valued)

    coeffs_hat, meta = placeholder_decoder(Y_noisy, P_mats, block_dicts, noise_var)
    counts_hat = assemble_global_counts(coeffs_hat, block_to_msg_list, args.num_codewords)
    metrics = evaluate_counts(message_counts, counts_hat)

    print_diagnostics(codebook, P_mats, msg_to_block, block_to_msg_list, active_msgs, user_channels, message_counts, block_coeffs, Y_clean, Y_noisy, noise_var, args.esn0_db)
    print(f"Decoder meta: {meta}")
    print(f"Decoder eval: {metrics}")

    ground_truth = dict(
        codebook=codebook,
        P_mats=P_mats,
        msg_to_block=msg_to_block,
        block_to_msg_list=block_to_msg_list,
        block_dicts=block_dicts,
        active_msgs=active_msgs,
        active_blocks=np.array([msg_to_block[int(m)] for m in active_msgs]),
        user_channels=user_channels,
        message_counts=message_counts,
        block_coeffs=block_coeffs,
        Y_clean=Y_clean,
        Y_noisy=Y_noisy,
        noise_var=noise_var,
    )
    return ground_truth


if __name__ == "__main__":
    main()
