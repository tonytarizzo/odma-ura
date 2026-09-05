"""Generate the preregistered job-028 hash-skeleton certification manifest."""

from __future__ import annotations

import csv
from pathlib import Path


FAMILIES = ["sparse_iid_fixed", "hash_table_random_fixed", "hash_linear_random_fixed", "hash_linear_selected_fixed"]
SUPPORTS = [16, 32]
SEEDS = [2801, 2802]
DECODERS = ["d0", "d1"]


def main() -> None:
    rows = []
    for support in SUPPORTS:
        for family in FAMILIES:
            for seed in SEEDS:
                for decoder in DECODERS:
                    name = f"B14_n256_{family.removesuffix('_fixed')}_s{support}_{decoder}_seed{seed}"
                    candidates = 128 if family == "hash_linear_selected_fixed" else 1
                    rows.append((name, family, decoder, 14, 256, support, seed, candidates))
    for seed in SEEDS:
        for decoder in DECODERS:
            rows.append((f"B14_n256_dense_{decoder}_seed{seed}", "dense_fixed", decoder, 14, 256, 256, seed, 1))
    path = Path(__file__).with_name("manifest.tsv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "encoder", "decoder", "B", "n", "support", "seed", "search_candidates"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
