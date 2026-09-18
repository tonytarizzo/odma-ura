"""A focused, paired coordinate-bank experiment; keep row order stable after submission."""

import csv
from pathlib import Path


def main():
    rows = []
    # Main frontier; raw J4 ablation; shared-label diagnostic; unrestricted endpoint.
    cases = [("coordinate", J, "balanced", mode) for J in (4, 8, 10) for mode in ("fixed", "learned")]
    cases += [("coordinate", 4, "raw", mode) for mode in ("fixed", "learned")]
    cases += [("shared", 4, init, "fixed") for init in ("raw", "balanced")]
    cases += [("coordinate", 14, "raw", mode) for mode in ("fixed", "learned")]
    for seed in (2801, 2802, 2803):
        for decoder in ("d0", "d1"):
            for mapping, J, init, mode in cases:
                name = f"B14_n256_T32_{mapping}_J{J}_{init}_{decoder}_{mode}_seed{seed}"
                rows.append((name, decoder, 14, 256, 32, J, mode, mapping, init, seed, 128))
    path = Path(__file__).with_name("manifest.tsv")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("name", "decoder", "B", "n", "support", "J", "mode", "mapping", "initialization", "seed",
                         "search_candidates"))
        writer.writerows(rows)
    assert len(rows) == len({row[0] for row in rows}) == 72
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
