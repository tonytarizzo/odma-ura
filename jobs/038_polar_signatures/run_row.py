"""Run one signature/energy arm; reference D2 reuses an audited Job035 checkpoint by default."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment import run_experiment, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "1")))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path(__file__).with_name("results_signatures"))
    parser.add_argument("--baseline-results", type=Path, default=ROOT / "jobs/035_polar_decoders/results_all_decoders")
    parser.add_argument("--fresh-reference", action="store_true", help="Regenerate amplitudes and train reference D2; apply to all arms")
    parser.add_argument("--smoke", action="store_true", help="Actual b6 signature, two frames/cell, one D2 epoch; never a PUPE result")
    parser.add_argument("--probe", action="store_true", help="One K26/4dB frame, DEC-26 only; no training, for timing")
    args = parser.parse_args()
    if args.smoke and args.probe:
        parser.error("Choose smoke or probe")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if not 1 <= args.index <= len(rows):
        parser.error(f"Index must be 1..{len(rows)}")
    row = copy.deepcopy(rows[args.index - 1])
    c = row["config"]
    if args.fresh_reference or args.smoke or args.probe:
        row["reference_policy"] = "regenerated"
    if args.smoke or args.probe:
        row["scope"] = "cost_probe" if args.probe else "execution_smoke_not_performance"
        row["pair_frames"] = c["eval_frames"] = 1 if args.probe else 2
        c.update(loads=[26] if args.probe else [2, 7], eval_ebn0=[4], max_epochs=1,
                 batches_per_epoch=1, validation_batches=1)
        if args.probe:
            c["decoders"] = ["DEC-26"]
        row["name"] += "_probe" if args.probe else "_smoke"
        c["name"] = row["name"]
    checkpoint, archive = None, None
    if row["reference_policy"] == "archived_job035":
        archive = args.baseline_results / f"B{c['B']}_n256_{c['mapping']}_iid_D2_s{c['seed']}"
        if not (archive / "encoder.npz").is_file() or not (archive / "summary.json").is_file():
            parser.error(f"Missing reference archive {archive}; use --baseline-results or explicitly --fresh-reference")
    if row["signature"] == "reference" and archive is not None:
        checkpoint = archive / "D2/decoder.pt"
        if not checkpoint.is_file():
            parser.error(f"Missing audited reference checkpoint {checkpoint}; use --baseline-results or explicitly --fresh-reference")
    out = args.out_root / row["name"]
    if (out / "summary.json").is_file():
        saved = json.loads((out / "summary.json").read_text())
        if saved["status"] != "complete" or saved["row"] != row or saved["source_sha256"] != source_hashes():
            raise ValueError(f"Existing run differs from requested row/source: {out}")
        print(f"Already complete: {out}")
        return
    result = run_experiment(row, out, checkpoint, archive)
    print(json.dumps({"output": str(out), "setup_seconds": result["encoder"]["seconds"],
                      "training_seconds_this_run": result["training"].get("seconds_this_run", 0),
                      "evaluation": result["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
