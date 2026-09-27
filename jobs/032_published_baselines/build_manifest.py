"""Deterministic native tuning, matched small-B comparisons and optional native-scale checks."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.ura_comparison import DEFAULTS


def manifests():
    pilot, comparison, native = [], [], []
    for b in (12, 14):
        n = 256
        profiles = {
            "odma_polar": [{"prefix_bits": prefix, "code_length": length, "crc_bits": crc, "list_size": size}
                           for prefix, length, (crc, size) in product((4, 6, 8), (32, 64), ((8, 8), (12, 32), (16, 128)))],
            "dynamic_cs": [{"prefix_bits": prefix, "first_length": first, "amp_iterations": 40}
                           for prefix, first in product((6, 8, 10), (85, 128))],
            "ccs_amp": [{"amp_iterations": iterations, "sic_fraction": fraction, "non_dc_embedding": non_dc}
                        for iterations, fraction, non_dc in product((20, 40), (0.5, 0.8), (False, True))],
            "ccs_block": [{"amp_iterations": iterations, "sic_fraction": fraction, "non_dc_embedding": non_dc}
                          for iterations, fraction, non_dc in product((20, 40), (0.5, 0.8), (False, True))],
        }
        for family, variants in profiles.items():
            for index, params in enumerate(variants, 1):
                pilot.append({"name": f"B{b}_n{n}_{family}_v{index}", "B": b, "n": n, "seed": 3299,
                              "family": family, "mode": "native", "decoder": "native", "baseline_params": params,
                              "loads": [7, 15, 26], "eval_ebn0": [0, 4, 8], "eval_sampling": ["distinct"], "eval_frames": 64})
        for seed in (3201, 3202, 3203):
            base = {"B": b, "n": n, "seed": seed}
            for family, decoder in product(profiles, ("native", "d0", "d1")):
                comparison.append({**base, "name": f"B{b}_n{n}_{family}_{decoder}_s{seed}", "family": family,
                                   "decoder": decoder, "mode": "native" if decoder == "native" else "fixed",
                                   "selected_profile": f"B{b}_n{n}_{family}"})
            for family in ("dense", "sparse"):
                for mode, decoder in product(("fixed", "joint"), ("d0", "d1")):
                    comparison.append({**base, "name": f"B{b}_n{n}_{family}_{mode}_{decoder}_s{seed}", "family": family,
                                       "decoder": decoder, "mode": mode, "support": 32})
                for decoder in ("d2", "d3", "d4"):
                    comparison.append({**base, "name": f"B{b}_n{n}_{family}_fixed_{decoder}_s{seed}", "family": family,
                                       "decoder": decoder, "mode": "fixed", "support": 32})
    # Preserve the first 156 row indices; append joint D2--D4 controls.
    for b, seed, family, decoder in product((12, 14), (3201, 3202, 3203), ("dense", "sparse"), ("d2", "d3", "d4")):
        comparison.append({"B": b, "n": 256, "seed": seed, "name": f"B{b}_n256_{family}_joint_{decoder}_s{seed}",
                           "family": family, "decoder": decoder, "mode": "joint", "support": 32})
    for seed, k, family in product((3251, 3252), (50, 100), ("odma_polar", "dynamic_cs", "ccs_amp")):
        if family == "odma_polar":
            b, n, snrs = 100, 30000, [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.5]
            params = {"prefix_bits": 12 if k == 50 else 13, "code_length": 512, "crc_bits": 16, "list_size": 128}
            source, figure, approximate = "https://doi.org/10.1109/LWC.2024.3359270", "Fig. 3 (GMAC)", 0.25 if k == 50 else 0.4
        elif family == "dynamic_cs":
            b, n, snrs = 100, 30000, [0.5, 1, 1.25, 1.5, 1.75, 2, 2.5]
            params = {"profile": "native100", "detection_threshold": 6.0, "cache_bytes": 4_000_000_000,
                      "amp_iterations": 40, "global_iterations": 2, "list_size": 3}
            source = "https://dalspace.library.dal.ca/server/api/core/bitstreams/78b29db0-2fc0-4c8a-8275-b0f8f7fbdae3/content"
            figure, approximate = "Thesis Fig. 9.3 / Table 9.1", 1.4 if k == 50 else 1.55
        else:
            b, n, snrs = 128, 38400, [1, 1.5, 1.75, 2, 2.25, 2.5, 3]
            params = {"amp_iterations": 40, "sic_fraction": 0.7, "list_extra": 10, "non_dc_embedding": False}
            source, figure, approximate = "https://arxiv.org/abs/2010.04364", "Fig. 8 (enhanced AMP+Tree)", 2.1 if k == 50 else 2.4
        reference = {"source": source, "figure": figure, "target_pupe": 0.05,
                     "approx_required_ebn0_db": approximate, "reading_uncertainty_db": 0.15,
                     "provenance": "Manual visual reading, not author-tabulated data; a comparison guide, not a pass/fail tolerance"}
        for shard, start in enumerate(range(0, len(snrs), 3), 1):
            native.append({"name": f"B{b}_{family}_K{k}_s{seed}_part{shard}", "B": b, "n": n, "seed": seed,
                           "family": family, "mode": "native", "decoder": "native", "loads": [k],
                           "baseline_params": params, "eval_frames": 64, "eval_ebn0": snrs[start:start+3],
                           "eval_sampling": ["iid"], "paper_reference": reference,
                           "purpose": "native-dimension paper-alignment check; reproduction remains unverified"})
    return {phase: [{**DEFAULTS, **row} for row in rows]
            for phase, rows in (("pilot", pilot), ("comparison", comparison), ("native", native))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for phase, rows in manifests().items():
        (args.out_dir / f"{phase}.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(f"{phase}: {len(rows)} rows")


if __name__ == "__main__": main()
