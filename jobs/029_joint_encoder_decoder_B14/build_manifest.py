"""Generate the focused job-029 joint encoder/decoder manifest."""

from __future__ import annotations

import csv
from pathlib import Path


SEEDS = [2801, 2802]
DECODERS = ["d0", "d1"]
SPARSE_CONFIGS = [
    ("sparse_iid_fixed", 16),
    ("hash_linear_selected_fixed", 16),
    ("sparse_iid_fixed", 32),
    ("hash_linear_selected_fixed", 32),
]


def main() -> None:
    rows = []
    for encoder, support in SPARSE_CONFIGS:
        for seed in SEEDS:
            for decoder in DECODERS:
                name = f"B14_n256_{encoder.removesuffix('_fixed')}_s{support}_{decoder}_seed{seed}_joint"
                candidates = 128 if encoder == "hash_linear_selected_fixed" else 1
                rows.append((name, encoder, decoder, 14, 256, support, seed, candidates))
    for seed in SEEDS:
        for decoder in DECODERS:
            rows.append((f"B14_n256_dense_{decoder}_seed{seed}_joint", "dense_fixed", decoder, 14, 256, 256, seed, 1))
    path = Path(__file__).with_name("manifest.tsv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "encoder", "decoder", "B", "n", "support", "seed", "search_candidates"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
