"""One decoder/B/seed per array row; mappings run sequentially, with all loads together."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import normalized_config
from framework.polar_decoders import REGISTRY


def manifest(profile="research", mappings=("balanced", "uniform"), sampling="iid"):
    rows = []
    bits, seeds = ([12, 14], [3201, 3202, 3203]) if profile == "research" else ([4], [3217])
    names = [name for name in ["REF", *[f"DEC-{i:02}" for i in range(1, 14)], "D2", "ODMA"]
             if sampling in REGISTRY[name].sources and (profile == "smoke" or name != "REF")]
    for b in bits:
        for seed in seeds:
            for name in names:
                runs = []
                for mapping in mappings:
                    # At n256, Gaussian continuation log potentials are routinely below the tiny-smoke floor -80.
                    params = {name: {"guide_log_floor": -1000.}} if profile == "research" and name == "DEC-07" else {}
                    config = normalized_config({"profile": profile, "B": b, "prefix_bits": b - 6 if b >= 12 else 2,
                                                "seed": seed, "mapping": mapping, "threads": 2, "decoders": [name],
                                                "eval_sampling": [sampling], "decoder_params": params})
                    config["name"] = f"B{b}_n{config['n']}_{mapping}_{sampling}_{name}_s{seed}"
                    runs.append(config)
                rows.append({"name": f"B{b}_{sampling}_{name}_s{seed}", "runs": runs})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "research"), default="research")
    parser.add_argument("--mappings", nargs="+", choices=("uniform", "balanced"), default=["balanced", "uniform"])
    parser.add_argument("--sampling", choices=("iid", "distinct"), default="iid")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    args = parser.parse_args()
    rows = manifest(args.profile, tuple(args.mappings), args.sampling)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} jobs containing {sum(len(r['runs']) for r in rows)} sequential runs to {args.out}; "
          f"submit with qsub -J '1-{len(rows)}'")


if __name__ == "__main__":
    main()
