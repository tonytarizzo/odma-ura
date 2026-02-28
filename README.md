# ODMA-URA

Research repository for decoders that jointly exploit **ODMA** (On-Off Division Multiple Access) and **URA** (Unsourced Random Access) structure. The goal is to recover **message counts** (which messages were sent and how often) from a superposition of codewords over ODMA resource blocks, rather than per-user identities.

## Concepts

- **ODMA**: Each message uses a sparse subset of \(n\) resources (block of \(d\) indices). Pattern matrices \(P_b \in \{0,1\}^{n \times d}\) embed block signals into the resource grid.
- **URA**: Shared codebook of unit-norm codewords; devices pick a message index and transmit the corresponding codeword. Collisions are multiple devices sending the same message.
- **Decoder target**: Global **message-count vector** \(\in \mathbb{Z}_+^{\text{num\_codewords}}\), aligned with the sparse coefficient structure of the received signal.

## Scripts (decoder testbeds)

Four progressively harder signal models in `scripts/`:

| Script | Model | Notes |
|--------|--------|--------|
| **v1** | Single stream, no fading | \(y = \sum_b P_b C_b^T a_b + z\); baseline for message-count decoder. |
| **v2** | Multi-antenna, no fading | \(Y = (\sum_b P_b C_b^T a_b)\,\mathbf{1}^T + Z\); \(M\) antennas, same spatial signature. |
| **v3** | Multi-antenna, first-antenna inversion | Devices pre-equalize so \(g_u[0]=1\); remaining antennas see ratio channels. |
| **v4** | Full MU-MIMO, Rayleigh fading | \(Y = \sum_u (P_{b_u} c_{m_u}) h_u^T + Z\); no CSI at devices. |

Each script builds codebook, ODMA blocks, message–block mapping, and synthetic received signal; the decoder itself is a **placeholder** (graph-based / AMP-style message passing is intended to be implemented).

## Run

From the repo root:

```bash
# V1 — single stream, no fading
python scripts/graph_based_decoder_v1.py --seed 42 --n 128 --d 16 --num-blocks 8 \
  --num-codewords 64 --num-devices-active 10 --esn0-db 10

# V2 — multi-antenna, no fading
python scripts/graph_based_decoder_v2.py --seed 42 --n 128 --d 16 --num-blocks 8 \
  --num-codewords 64 --num-devices-active 10 --num-antennas 4 --esn0-db 10

# V3 / V4 — same pattern, add --num-antennas and (for V3) --min-first-ant-power
python scripts/graph_based_decoder_v3.py ...
python scripts/graph_based_decoder_v4.py ...
```

Common args: `--n` (resources), `--d` (codeword/block size), `--num-blocks`, `--num-codewords`, `--num-devices-active`, `--esn0-db`. Use `--help` on each script for full options.

## Setup

```bash
pip install -r requirements.txt
```
