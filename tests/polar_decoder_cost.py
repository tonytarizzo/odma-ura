"""CPU feasibility timings at B12/B14, not PUPE evidence or a scaling guarantee."""

import argparse
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from benchmarks.polar_decoders import build_encoder, fixture, normalized_config, source_hashes, write_json
from benchmarks.ura_comparison import learned_batch, noise_variance, objective
from baselines import message_bits
from framework.polar_decoders import REGISTRY, Scenario, hypothesis_count, run
from framework.polar_decoder_queries import geometry
from framework.polar_mapping import PolarD2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", nargs="+", type=int, default=[12, 14])
    parser.add_argument("--loads", nargs="+", type=int, default=[7, 26])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decoders", nargs="+", default=[name for name in REGISTRY if name != "REF"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing cost profile")
    torch.set_num_threads(args.threads)
    rows = []
    for b in args.bits:
        c = normalized_config({"profile": "research", "B": b, "prefix_bits": b - 6, "eval_frames": max(8, args.repeats),
                               "threads": args.threads, "eval_ebn0": [4.], "loads": args.loads, "decoders": args.decoders})
        start = time.perf_counter()
        base, encoder, phi, _ = build_encoder(c)
        encoder_seconds = time.perf_counter() - start
        decoder = PolarD2(c["layers"]).double()
        start = time.perf_counter()
        decoder._gram_geometry(encoder)
        geometry_seconds = time.perf_counter() - start
        start = time.perf_counter()
        query_cache = geometry(phi)
        query_geometry_seconds = time.perf_counter() - start
        for k in args.loads:
            ids, noise, signals = fixture(c, phi, k, "iid")
            variance = noise_variance(b, 4.)
            times, details = {}, {}
            for name in args.decoders:
                elapsed, meta_rows = [], []
                for repeat in range(args.repeats):
                    s = Scenario(phi, signals[repeat] + np.sqrt(variance) * noise[repeat], k, variance)
                    params = ({"samples": args.samples, "seed": 991 + repeat} if name == "DEC-05" else
                              {"encoder": encoder, "decoder": decoder} if name == "D2" else
                              {"baseline": base, "amplitudes": encoder.codebook.amplitudes().numpy()} if name == "ODMA" else {})
                    if name in ("DEC-01", "DEC-13"):
                        params["geometry_cache"] = query_cache
                    if REGISTRY[name].stochastic:
                        params["seed"] = 991 + repeat
                    start = time.perf_counter()
                    _, meta = run(name, s, **params)
                    elapsed.append(time.perf_counter() - start)
                    meta_rows.append({key: meta[key] for key in ("iterations", "optimizer_converged", "effective_sample_size",
                                                                "polar_trials", "excess_risk_bound", "candidate_count",
                                                                "maximum_tail_interval_width") if key in meta})
                times[name] = {"mean": float(np.mean(elapsed)), "min": min(elapsed), "max": max(elapsed),
                               "repeats": args.repeats}
                details[name] = meta_rows
                print(f"B={b} K={k} {name}: {np.mean(elapsed):.4f} s/frame", flush=True)
            training_times = []
            training_config = {**c, "decoder": "d2"}
            optimizer = torch.optim.Adam(decoder.parameters(), lr=c["lr"])
            for _ in range(args.repeats):
                start = time.perf_counter()
                counts, y, nv = learned_batch(encoder, message_bits(ids[:8], b), noise[:8], 4.)
                output = decoder(encoder, y[..., None], torch.ones((8, 1), dtype=encoder.dtype), k, noise_var=nv)
                loss = objective(training_config, output, counts, None)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
                optimizer.step()
                training_times.append(time.perf_counter() - start)
            rows.append({"B": b, "n": c["n"], "K": k, "ebn0_db": 4., "mapping": c["mapping"],
                         "encoder_seconds": encoder_seconds, "D2_geometry_seconds": geometry_seconds,
                         "query_geometry_seconds": query_geometry_seconds,
                         "matrix_bytes": phi.nbytes, "REF_count_states_iid": str(hypothesis_count(2**b, k, "iid")),
                         "seconds_per_frame": times, "details": details,
                         "D2_seconds_per_training_batch_8": float(np.mean(training_times)),
                         "estimated_full_grid_inference_hours_per_seed": {
                             name: value["mean"] * 4 * 8 * 256 / 3600 for name, value in times.items()},
                         "estimated_one_load_job_inference_hours": {
                             name: value["mean"] * 8 * 256 / 3600 for name, value in times.items()}})
    result = {"scope": "timing only; these few observations do not establish PUPE or worst-case runtime",
              "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "threads": args.threads, "importance_samples": args.samples, "source_sha256": source_hashes(),
              "extrapolation": "8192 frames = 4 loads x 8 SNRs x 256 frames, one source/seed/encoder; excludes training/setup",
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
