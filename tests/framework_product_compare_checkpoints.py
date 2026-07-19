"""Evaluate two product-experiment checkpoints on identical fresh batches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.channel import constant_fading  # noqa: E402
from tests.framework_product_experiment import (build_experiment_encoder, evaluate_one, make_decoder,
                                                parse_float_grid, parse_int_grid)  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", required=True)
    p.add_argument("--second", required=True)
    p.add_argument("--eval-k", type=parse_int_grid, default=parse_int_grid("5"))
    p.add_argument("--eval-ebn0", type=parse_float_grid, default=parse_float_grid("-2,0,2,4,6,8"))
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=91001)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def load_checkpoint(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    run_args = argparse.Namespace(**payload["metadata"]["args"])
    generator = torch.Generator().manual_seed(int(run_args.seed))
    encoder, _, _ = build_experiment_encoder(run_args, generator)
    encoder.load_state_dict(payload["encoder"])
    decoder = make_decoder(run_args)
    decoder.load_state_dict(payload["decoder"])
    encoder.eval(); decoder.eval()
    return payload, run_args, encoder, decoder


def assert_same_encoder(first, second) -> None:
    first_state, second_state = first.state_dict(), second.state_dict()
    if first_state.keys() != second_state.keys():
        raise ValueError("checkpoints do not contain the same encoder structure")
    for key in first_state:
        if not torch.equal(first_state[key], second_state[key]):
            raise ValueError(f"encoder states differ at {key}; paired decoder comparison would be confounded")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    first_payload, first_args, first_encoder, first_decoder = load_checkpoint(args.first)
    second_payload, second_args, second_encoder, second_decoder = load_checkpoint(args.second)
    assert_same_encoder(first_encoder, second_encoder)
    if first_args.payload_bits != second_args.payload_bits or first_args.n != second_args.n:
        raise ValueError("checkpoint URA dimensions differ")
    eval_args = argparse.Namespace(**vars(first_args))
    eval_args.eval_batches = int(args.eval_batches)
    eval_args.batch_size = int(args.batch_size)
    first_fading = constant_fading(first_encoder.spec.num_antennas, first_encoder.dtype, first_encoder.device)
    second_fading = constant_fading(second_encoder.spec.num_antennas, second_encoder.dtype, second_encoder.device)
    rows = []
    for k_index, K in enumerate(args.eval_k):
        for snr_index, ebn0_db in enumerate(args.eval_ebn0):
            eval_seed = int(args.eval_seed) + 1000 * k_index + snr_index
            first_gen = torch.Generator().manual_seed(eval_seed)
            second_gen = torch.Generator().manual_seed(eval_seed)
            first_metrics, matched = evaluate_one(first_encoder, first_decoder, K, ebn0_db, eval_args,
                                                  first_gen, first_fading)
            second_metrics, second_matched = evaluate_one(second_encoder, second_decoder, K, ebn0_db, eval_args,
                                                          second_gen, second_fading)
            for key in ["pupe", "l1_acc", "f1", "exact_count"]:
                if matched[key] != second_matched[key]:
                    raise RuntimeError(f"paired evaluation batches diverged at metric {key}")
            row = {"K": K, "ebn0_db": ebn0_db, "first": first_metrics,
                   "second": second_metrics, "matched_filter": matched}
            rows.append(row)
            print(f"K={K:3d} Eb/N0={ebn0_db:5.1f} first PUPE={first_metrics['pupe']:.4f} "
                  f"second PUPE={second_metrics['pupe']:.4f} matched={matched['pupe']:.4f}")
    result = {"first": {"path": args.first, "decoder": first_payload["metadata"]["args"]["decoder"]},
              "second": {"path": args.second, "decoder": second_payload["metadata"]["args"]["decoder"]},
              "eval_seed": args.eval_seed, "eval_batches": args.eval_batches,
              "batch_size": args.batch_size, "rows": rows}
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
