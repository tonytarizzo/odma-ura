"""Reevaluate frozen 032/034 encoder+D2 checkpoints on the Job038 physical frames and count loss.

Historical IID aggregate scores are not imported. Float32 waveforms are promoted and explicitly
normalized to unit power for common inference; original error must satisfy the historical 1e-5 tolerance.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from baselines import message_bits
from benchmarks import polar_mapping, ura_comparison
from framework.polar_decoders import Scenario, run
from framework.polar_mapping import PolarD2
from src.metrics import PUPE_DEFINITION
from experiment import TRAIN_KEYS, common, digest, source_hashes


def load_checkpoint(row, root32, root34):
    c, label = row["config"], row["encoder_id"]
    family, mode = label.rsplit("_", 1)
    root = root34 if family == "polar_energy" else root32
    path = root / f"B{c['B']}_n256_{family}_{mode}_d2_s{c['seed']}"
    summary = json.loads((path / "summary.json").read_text())
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    old = summary["config"]
    expected_training = {**c, "loads": c["train_loads"]}
    if (summary["status"] != "complete" or checkpoint["config"] != old or old["family"] != family
            or old["mode"] != mode or old["B"] != c["B"] or old["n"] != c["n"] or old["decoder"] != "d2"
            or any(old[key] != expected_training[key] for key in TRAIN_KEYS)):
        raise ValueError(f"Checkpoint/config/training contract mismatch: {path}")
    if family == "polar_energy":
        _, encoder = polar_mapping.make_encoder(old)
    else:
        encoder = ura_comparison.make_encoder(old)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    # Freeze the exact stored float32 physical matrix before changing arithmetic precision.
    with torch.no_grad():
        phi = encoder.explicit_matrix().detach().numpy().astype(np.float64)
    energy = np.square(phi).sum(0)
    if not np.isfinite(phi).all() or np.max(np.abs(energy - 1)) > 1e-5:
        raise ValueError(f"Historical codebook fails per-message power tolerance: {path}")
    if family != "polar_energy" and mode == "fixed":
        if digest(phi.astype(np.float32)) != summary["initial_matrix_sha256"]:
            raise ValueError(f"Frozen matrix does not reproduce the archived initial matrix: {path}")
    normalization_change = float(np.max(np.abs(1 / np.sqrt(energy) - 1)))
    phi /= np.sqrt(energy)[None, :]
    encoder = ura_comparison.make_encoder({**old, "family": "dense", "mode": "fixed"}).double()
    with torch.no_grad():
        encoder.components[0].C.copy_(torch.from_numpy(phi))
    encoder.requires_grad_(False)
    decoder = PolarD2(c["layers"]).double()
    decoder.load_state_dict(checkpoint["decoder"], strict=True)
    decoder.eval()
    provenance = {"path": str(path), "checkpoint_sha256": hashlib.sha256((path / "checkpoint.pt").read_bytes()).hexdigest(),
                  "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
                  "original_revision": summary["revision"], "original_source_sha256": summary["source_sha256"],
                  "original_training": summary["training"], "encoder_learned_parameters": summary["encoder_learned_parameters"],
                  "precision": "float64 inference; stored float32 columns explicitly unit-normalized; D2 parameters reused",
                  "original_per_message_energy": [float(energy.min()), float(energy.max())],
                  "maximum_column_rescaling": normalization_change,
                  "per_message_energy": [float(np.square(phi).sum(0).min()), float(np.square(phi).sum(0).max())]}
    return encoder, decoder, phi, provenance


def run_comparator(row, out, root32, root34):
    c, out = row["config"], Path(out)
    if c["decoders"] != ["D2"] or c["eval_sampling"] != ["iid"]:
        raise ValueError("Archived comparators use common D2 and fresh IID count evaluation")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    fingerprints = source_hashes()
    torch.set_num_threads(c["threads"])
    before = time.perf_counter()
    encoder, decoder, phi, provenance = load_checkpoint(row, root32, root34)
    setup_seconds = time.perf_counter() - before
    before = time.perf_counter()
    decoder._gram_geometry(encoder)
    geometry_seconds = time.perf_counter() - before
    out.mkdir(parents=True)
    (out / "inputs").mkdir()
    common.write_json(out / "config.json", row)
    np.savez_compressed(out / "encoder.npz", phi=phi)
    records = []
    with (out / "frames.jsonl").open("w") as stream:
        for k in c["loads"]:
            ids, standard_noise, signals = common.fixture(c, phi, k, "iid")
            for snr in c["eval_ebn0"]:
                variance = ura_comparison.noise_variance(c["B"], snr)
                observations = signals + math.sqrt(variance) * standard_noise
                np.savez_compressed(out / "inputs" / f"iid_K{k}_EbN0{float(snr):g}.npz", message_ids=ids,
                                    standard_noise=standard_noise, y=observations, noise_var=variance)
                for frame, y in enumerate(observations):
                    before = time.perf_counter()
                    counts, meta = run("D2", Scenario(phi, y, k, variance, "iid"), encoder=encoder, decoder=decoder)
                    seconds = time.perf_counter() - before
                    selected = np.flatnonzero(counts)
                    decoded = message_bits(np.repeat(selected, counts[selected]), c["B"])
                    record = {"decoder": "D2", "sampling": "iid", "K": k, "ebn0_db": snr, "frame": frame,
                              "seed": c["seed"], "seconds": seconds, "meta": meta,
                              "metrics": ura_comparison.frame_metrics(message_bits(ids[frame], c["B"]), decoded),
                              "counts": [[int(w), int(counts[w])] for w in selected], "input_sha256": digest(y),
                              "truth_sha256": digest(ids[frame]), "standard_noise_sha256": digest(standard_noise[frame])}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(record)
                stream.flush()
    if source_hashes() != fingerprints:
        raise RuntimeError("Sources changed during comparator evaluation")
    result = {"status": "complete", "row": row, "config": c, "encoder_id": row["encoder_id"],
              "source_sha256": fingerprints, "pupe_definition": PUPE_DEFINITION, "matrix_sha256": digest(phi),
              "archive": provenance, "setup_seconds": setup_seconds, "D2_geometry_seconds": geometry_seconds,
              "training": {"seconds_this_run": 0., "checkpoint_reused": provenance["path"]},
              "evaluation": common.summarize(records), "scope": row["scope"],
              "resource_boundary": "n256/unit energy/B/load/channel matched; dense occupies 256 versus sparse/polar 32 resources",
              "calibration_boundary": "same D2 recurrence/training budget; historical joint training optimizes encoder too"}
    common.write_json(out / "summary.json", result)
    return result
