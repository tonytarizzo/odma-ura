"""Generate the preregistered job-027 sparse-density manifest."""

from __future__ import annotations

import csv
from pathlib import Path


SUPPORTS = [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]
SEEDS = [2701, 2702]
DECODERS = ["d0", "d1"]


def main() -> None:
    rows = []
    for support in SUPPORTS:
        for seed in SEEDS:
            for decoder in DECODERS:
                name = f"B12_n256_sparse_s{support}_{decoder}_seed{seed}"
                rows.append((name, "sparse_global_fixed", decoder, 12, 256, support, 1, seed))
    for encoder, label, support, Q in (("dense_fixed", "dense", 256, 1), ("odma_fixed", "odma_Q4", 64, 4)):
        for seed in SEEDS:
            for decoder in DECODERS:
                name = f"B12_n256_{label}_{decoder}_seed{seed}"
                rows.append((name, encoder, decoder, 12, 256, support, Q, seed))
    path = Path(__file__).with_name("manifest.tsv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "encoder", "decoder", "B", "n", "support", "Q", "seed")); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
