"""Common-D2 encoder comparison: four signature/energy arms and five archived encoder controls."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import normalized_config


def manifest():
    rows = []
    for b in (12, 14):
        for seed in (3201, 3202, 3203):
            for signature in ("reference", "repair3"):
                for mapping in ("uniform", "balanced"):
                    name = f"B{b}_{signature}_{mapping}_s{seed}"
                    config = normalized_config({"profile": "research", "name": name, "B": b, "prefix_bits": b - 6,
                                                "seed": seed, "mapping": mapping, "threads": 2, "decoders": ["D2"],
                                                "loads": [7, 26], "eval_ebn0": [4, 10], "eval_frames": 128,
                                                "eval_sampling": ["iid"], "bounds": False,
                                                "decoder_params": {"DEC-26": {"width": 32, "patterns": 8,
                                                                               "list_size": 16, "max_refine": 32}}})
                    rows.append({"name": name, "signature": signature, "config": config, "pair_frames": 32,
                                 "reference_policy": "archived_job035",
                                 "encoder_id": f"polar_{signature}_{mapping}",
                                 "scope": "exploratory_fixed_resource_comparison_existing_seeds"})
    for b in (12, 14):
        for seed in (3201, 3202, 3203):
            template = next(r for r in rows if (r["config"]["B"], r["config"]["seed"]) == (b, seed))
            for encoder_id in ("dense_fixed", "dense_joint", "sparse_fixed", "sparse_joint", "polar_energy_joint"):
                name = f"B{b}_{encoder_id}_s{seed}"
                config = {**template["config"], "name": name, "decoder_params": {}}
                rows.append({"name": name, "encoder_id": encoder_id, "config": config,
                             "scope": "exploratory_archived_encoder_and_D2_fresh_count_evaluation"})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    args = parser.parse_args()
    rows = manifest()
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print("54 rows; 27,648 D2-frame evaluations; 3,072 oracle pair diagnostics; 12 new D2 calibrations.")
