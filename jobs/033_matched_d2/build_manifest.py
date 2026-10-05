"""Extend 032 without changing its rows or its frozen experimental contract."""

import json
from pathlib import Path


def manifest():
    path = Path(__file__).resolve().parents[1] / "032_published_baselines/comparison.jsonl"
    previous = [json.loads(line) for line in path.read_text().splitlines()]
    rows = []
    for old in previous:
        if old["decoder"] == "d1" and "selected_profile" in old:
            rows.append({**old, "name": old["name"].replace("_d1_", "_d2_"), "decoder": "d2"})
    for old in previous:
        if (old["family"], old["mode"], old["decoder"]) != ("sparse", "fixed", "d2"):
            continue
        for j in (old["B"], 4, 8, 10):
            full = j == old["B"]
            family = "hash_full" if full else "hash_coordinate"
            label = "hash_full" if full else f"hash_coordinate_J{j}"
            for mode in ("fixed", "joint"):
                rows.append({**old, "name": f"B{old['B']}_n{old['n']}_{label}_{mode}_d2_s{old['seed']}",
                             "family": family, "mode": mode, "amplitude_label_bits": j,
                             "hash_search_candidates": 128})
    return rows


if __name__ == "__main__":
    rows = manifest()
    Path(__file__).with_name("manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} rows: 18 published-codebook controls + 48 hash/amplitude runs")
