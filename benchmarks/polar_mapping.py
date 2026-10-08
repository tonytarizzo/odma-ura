"""Job-033-matched polar mapping experiments; historical benchmark code is unchanged."""

import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch

from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.ura_comparison import DEFAULTS, _atomic_json, evaluate, materialize, train
from framework.polar_mapping import PolarD2, PolarExplicitEncoder


FAMILIES = {"polar_energy": "energy", "polar_mixer": "mixer", "odma_polar": "identity"}


def source_hashes():
    paths = sorted(Path("baselines").glob("*.py")) + sorted(Path("benchmarks").glob("*.py"))
    paths += sorted(Path("framework").glob("*.py")) + [Path("src/ura_bound.py"), Path("tests/ccs_amp_author.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_encoder(config):
    baseline = ODMAPolarBaseline(config["B"], config["n"], config["seed"], **config["baseline_params"])
    encoder = PolarExplicitEncoder(baseline, config["loads"], FAMILIES[config["family"]], config["mode"] == "joint")
    return baseline, encoder


def energy_audit(encoder):
    with torch.no_grad(): energies = encoder.explicit_matrix().double().square().sum(0)
    return {"scope": "exhaustive", "min": float(energies.min()), "mean": float(energies.mean()),
            "max": float(energies.max()), "std": float(energies.std(unbiased=False))}


def run_experiment(config, out_dir):
    config = {**DEFAULTS, **config}
    if config["family"] not in FAMILIES or config["decoder"] != "d2" or config["mode"] not in {"fixed", "joint"}:
        raise ValueError("Expected a polar mapping with fixed/joint full-alphabet D2")
    if not 1 <= min(config["loads"]) <= max(config["loads"]) < 2 ** config["B"]:
        raise ValueError("Require 1<=K<2^B")
    if min(config[k] for k in ("max_epochs", "batch_size", "eval_frames", "validation_batches", "batches_per_epoch")) < 1:
        raise ValueError("Training and evaluation budgets must be positive")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"] + 400_000)
    fingerprints = source_hashes()
    baseline, encoder = make_encoder(config)
    initial_matrix = encoder.explicit_matrix().detach().numpy()
    if not np.array_equal(initial_matrix, materialize(baseline)):
        raise AssertionError("Mapping initialization must exactly reproduce the original ODMA-polar matrix")
    initial_hash = hashlib.sha256(initial_matrix.tobytes()).hexdigest()
    initial_energy, initial_diagnostics = energy_audit(encoder), encoder.codebook.diagnostics()
    del initial_matrix
    decoder = PolarD2(config["layers"])
    training = train(config, encoder, decoder, out_dir)
    cells = evaluate(config, baseline, encoder, decoder, out_dir)
    final_energy = energy_audit(encoder)
    if not all(np.isfinite(final_energy[k]) and abs(final_energy[k] - 1) <= 1e-5 for k in ("min", "max")):
        raise FloatingPointError("The learned construction violates per-message unit energy")
    if source_hashes() != fingerprints: raise RuntimeError("Source changed during this run")
    summary = {"status": "complete", "config": config,
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_sha256": fingerprints, "baseline": baseline.metadata(), "training": training, "evaluation": cells,
               "initial_matrix_sha256": initial_hash, "energy_audit_initial": initial_energy, "energy_audit_final": final_energy,
               "mapping_diagnostics_initial": initial_diagnostics, "mapping_diagnostics_final": encoder.codebook.diagnostics(),
               "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
               "encoder_learned_parameters": sum(p.numel() for p in encoder.parameters()),
               "physical_noise_convention": "real AWGN sigma^2=1/(2*B*10^(EbN0_dB/10)); nominal E=1",
               "known_to_receiver": ["K", "noise variance", "codebook/construction", "unit real channel"],
               "candidate_policy": None, "candidate_training": None,
               "encoder_construction": {"mapping": encoder.codebook.mapping, "prefix_bits": baseline.prefix_bits,
                                        "polar_length": baseline.code_length, "patterns": baseline.num_patterns,
                                        "initialization": "exact original equal-energy ODMA-polar",
                                        "energy_constraint": "per-message identity in forward, no energy penalty",
                                        "decoder_geometry": "exact polar-factor Gram diagonal/row energy; unchanged D2 recurrence",
                                        "global_decoder_state": True}}
    _atomic_json(out_dir / "summary.json", summary)
    return summary
