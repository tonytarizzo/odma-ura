"""Generate the matched fixed-versus-learned amplitude-frontier manifest."""

from __future__ import annotations

import csv
from pathlib import Path


SEEDS = [2801, 2802]
DECODERS = ["d0", "d1"]
BASE_LABEL_BITS = [0, 2, 4, 8, 14]
EXTENSION_LABEL_BITS = [10, 12]


def main() -> None:
    rows = []
    for seed in SEEDS:
        for decoder in DECODERS:
            for J in BASE_LABEL_BITS:
                for mode in ("fixed", "learned"):
                    name = f"B14_n256_hash_T32_J{J}_{decoder}_{mode}_seed{seed}"
                    rows.append((name, decoder, 14, 256, 32, J, mode, "gaussian", seed, 128))
            rows.append((f"B14_n256_hash_T32_equal_{decoder}_fixed_seed{seed}", decoder, 14, 256, 32,
                         0, "fixed", "equal", seed, 128))
            rows.append((f"B14_n256_hash_T32_rademacher_{decoder}_fixed_seed{seed}", decoder, 14, 256, 32,
                         14, "fixed", "rademacher", seed, 128))
    # Keep the completed base sweep at rows 1--48 so its PBS indices remain stable.
    for seed in SEEDS:
        for decoder in DECODERS:
            for J in EXTENSION_LABEL_BITS:
                for mode in ("fixed", "learned"):
                    name = f"B14_n256_hash_T32_J{J}_{decoder}_{mode}_seed{seed}"
                    rows.append((name, decoder, 14, 256, 32, J, mode, "gaussian", seed, 128))
    path = Path(__file__).with_name("manifest.tsv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "decoder", "B", "n", "support", "J", "mode", "amplitude_init", "seed",
                         "search_candidates"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
