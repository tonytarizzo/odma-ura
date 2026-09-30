"""Keep receiver variants separate; these diagnostic rows must not be pooled as seeds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.ura_merge import load_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results") / "checks")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    results, audit = load_results(Path(__file__).with_name("checks.jsonl"), args.results, args.allow_incomplete)
    rows = []
    for run in results:
        config = run["config"]
        for cell in run["evaluation"]:
            ds = cell["native_diagnostics"]
            row = {"name": config["name"], "variant": config["variant"], "seed": config["seed"],
                   "K": cell["K"], "ebn0_db": cell["ebn0_db"], "frames": cell["frames"],
                   "pupe": cell["means"]["pupe"], "seconds_per_frame": cell["decoder_seconds_per_frame"]}
            if config["family"] == "odma_polar":
                cap = config["baseline_params"]["max_iterations"]
                row["active_cap_hits"] = sum(d["iterations"] == cap and d["list_size"] < cell["K"]
                                             and d["trace"][-1]["new_messages"] > 0 for d in ds)
            if config["family"] == "dynamic_cs":
                row["mean_header_candidates"] = sum(d["header_candidates"] for d in ds)/len(ds)
            rows.append(row)
            print(json.dumps(row))
    (args.results / "analysis.json").write_text(json.dumps({"audit": audit, "rows": rows}, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__": main()
