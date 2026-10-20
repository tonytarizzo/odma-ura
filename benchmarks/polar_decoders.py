"""Paired Report-10 decoder benchmark: fixed encoder, physical noise, count-PUPE.

Run directly, through tests.polar_decoder_benchmark, or from a JSON job row.
Decoder selection never changes the transmitted frames or D2 training data.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch

from baselines import message_bits, message_indices
from baselines.odma_polar import ODMAPolarBaseline
from benchmarks.polar_energy_geometry import balanced_energy
from benchmarks.ura_bounds import list_fano_converse, polyanskiy_achievability
from benchmarks.ura_comparison import DEFAULTS as TRAIN_DEFAULTS, frame_metrics, noise_variance, sample_messages, train
from framework.polar_decoders import REGISTRY, Scenario, hypothesis_count, prepare_reference, run
from framework.polar_decoder_queries import geometry
from framework.polar_mapping import PolarD2, PolarExplicitEncoder
from src.metrics import PUPE_DEFINITION


PROFILES = {
    "smoke": {"B": 4, "n": 12, "seed": 3217, "prefix_bits": 2, "code_length": 8, "crc_bits": 4,
              "loads": [1, 2, 3, 4], "train_loads": [1, 2, 3, 4], "eval_ebn0": [-4., 4.], "eval_frames": 128,
              "max_epochs": 30, "batches_per_epoch": 16, "validation_batches": 8, "patience": 8,
              "decoders": ["REF", *[f"DEC-{i:02}" for i in range(1, 14)], "D2", "ODMA"]},
    "research": {"B": 12, "n": 256, "seed": 3201, "prefix_bits": 6, "code_length": 32, "crc_bits": 8,
                 "train_loads": [7, 15, 22, 26], "decoders": [*[f"DEC-{i:02}" for i in range(1, 14)], "D2", "ODMA"]},
}
EXTRA_DEFAULTS = {
    "profile": "smoke", "mapping": "balanced", "name": "polar_decoders", "eval_sampling": ["iid"],
    "list_size": 8, "detector_slack": 5, "native_iterations": 30, "sic_mode": "parallel",
    "decoder_params": {}, "reference_max_states": 200_000, "reference_max_bytes": 512_000_000,
    "balance_tolerance": 1e-10, "balance_iterations": 10_000,
    "bound_grid": 31, "bound_power_grid": 31, "bound_order": 96, "bounds": True, "d2_checkpoint": None,
}
ROOT = Path(__file__).resolve().parents[1]


def normalized_config(config):
    profile = config.get("profile", "smoke")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile}")
    defaults = {**TRAIN_DEFAULTS, **EXTRA_DEFAULTS, **PROFILES[profile]}
    unknown = set(config) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    c = {**defaults, **config}
    c["decoders"] = [name.upper() for name in c["decoders"]]
    if not c["decoders"] or len(set(c["decoders"])) != len(c["decoders"]):
        raise ValueError("Select a nonempty, nonduplicated decoder list")
    for name in c["decoders"]:
        if name not in REGISTRY or REGISTRY[name].run is None:
            raise ValueError(f"{name} is unknown or catalogued but not implemented; use --list-decoders")
        if not set(c["eval_sampling"]) <= set(REGISTRY[name].sources):
            raise ValueError(f"{name} supports {REGISTRY[name].sources}; split the source-specific runs")
    if not 2 <= c["B"] <= 20 or c["mapping"] not in ("uniform", "balanced"):
        raise ValueError("Explicit benchmark requires 2<=B<=20 and uniform/balanced mapping")
    for key in ("loads", "train_loads"):
        if not c[key] or any(int(k) != k or not 1 <= k < 2**c["B"] for k in c[key]):
            raise ValueError(f"Require positive integer {key} below M")
        if len(set(c[key])) != len(c[key]):
            raise ValueError(f"Duplicate {key}")
    if not c["eval_ebn0"] or not np.isfinite(c["eval_ebn0"]).all() or len(set(c["eval_ebn0"])) != len(c["eval_ebn0"]):
        raise ValueError("Use distinct finite Eb/N0 points")
    for key in ("eval_frames", "batch_size", "max_epochs", "batches_per_epoch", "validation_batches", "threads", "layers"):
        if int(c[key]) != c[key] or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["train_sampling"] not in ("iid", "distinct") or not c["eval_sampling"] or not set(c["eval_sampling"]) <= {"iid", "distinct"}:
        raise ValueError("Invalid training/evaluation source")
    for name, params in c["decoder_params"].items():
        if name not in REGISTRY or not set(params) <= dict(REGISTRY[name].defaults).keys():
            raise ValueError(f"Unknown decoder parameters for {name}")
        for key, value in params.items():
            default = dict(REGISTRY[name].defaults)[key]
            if isinstance(default, int) and (not isinstance(value, int) or value < (0 if key == "field_rank" else 1)):
                raise ValueError(f"Invalid integer budget {name}.{key}")
            if isinstance(default, float) and not np.isfinite(value):
                raise ValueError(f"Nonfinite parameter {name}.{key}")
            if key in ("defensive", "global_probability") and not 0 < value <= 1:
                raise ValueError(f"{name}.{key} must be in (0,1]")
            if key == "failure_probability" and not 0 < value < 1:
                raise ValueError(f"{name}.{key} must be in (0,1)")
            if key in ("block_size", "order") and value not in (1, 2, 3):
                raise ValueError(f"{name}.{key} supports only 1, 2 or 3")
            if key in ("spacing", "tolerance") and value < 0 or key == "guide_log_floor" and value >= 0:
                raise ValueError(f"Invalid {name}.{key}")
    p = {**dict(REGISTRY["DEC-02"].defaults), **c["decoder_params"].get("DEC-02", {})}
    q = {**dict(REGISTRY["DEC-05"].defaults), **c["decoder_params"].get("DEC-05", {})}
    if p["max_iterations"] < 1 or p["tolerance"] <= 0:
        raise ValueError("DEC-02 requires a positive iteration budget and tolerance")
    if q["samples"] < 1 or q["batch_size"] < 1 or not 0 < q["defensive"] <= 1 or not 0 < q["failure_probability"] < 1:
        raise ValueError("Invalid DEC-05 budget/mixture/failure probability")
    if c["bounds"] and (min(c["bound_grid"], c["bound_power_grid"]) < 2 or c["bound_order"] < 8):
        raise ValueError("Invalid bound quadrature/grid budget")
    if "REF" in c["decoders"]:
        for source in c["eval_sampling"]:
            for k in c["loads"]:
                states = hypothesis_count(2**c["B"], k, source)
                if states > c["reference_max_states"]:
                    raise ValueError(f"REF K={k} needs {states:,} states; omit REF or explicitly increase its cap")
    return c


def source_hashes():
    paths = set()
    for folder in ("framework", "baselines", "benchmarks"):
        paths.update((ROOT / folder).glob("*.py"))
    paths.update(ROOT / p for p in ("src/metrics.py", "src/ura_bound.py", "uv.lock"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_encoder(c):
    base = ODMAPolarBaseline(c["B"], c["n"], c["seed"], prefix_bits=c["prefix_bits"], code_length=c["code_length"],
                             crc_bits=c["crc_bits"], list_size=c["list_size"], detector_slack=c["detector_slack"],
                             max_iterations=c["native_iterations"], sic_mode=c["sic_mode"])
    encoder = PolarExplicitEncoder(base, c["train_loads"], "energy" if c["mapping"] == "balanced" else "identity").double()
    balance = None
    if c["mapping"] == "balanced":
        energies, balance = balanced_energy(base.positions, c["n"], c["balance_tolerance"], c["balance_iterations"])
        with torch.no_grad():
            encoder.codebook.values.copy_(torch.from_numpy(np.log(energies)))
    phi = encoder.explicit_matrix().detach().numpy().copy()
    if not np.allclose(np.square(phi).sum(0), 1, rtol=0, atol=1e-12):
        raise ValueError("Encoder failed exhaustive unit-energy audit")
    phi.setflags(write=False)
    return base, encoder, phi, balance


def fixture(c, phi, k, source):
    """Per-frame keys preserve pairing across decoder selection, order, SNR and job splits."""
    ids, noise = [], []
    for frame in range(c["eval_frames"]):
        rng = np.random.default_rng(np.random.SeedSequence([c["seed"], 300_000, k, int(source == "iid"), frame]))
        ids.append(message_indices(sample_messages(rng, 1, k, c["B"], source))[0])
        noise.append(rng.standard_normal(c["n"]))
    ids, noise = np.asarray(ids), np.asarray(noise)
    signals = phi[:, ids].sum(2).T
    return ids, noise, signals


def interval(values, bounded=True):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    half = float(student_t.ppf(.975, len(values) - 1) * se) if se is not None else None
    ci = None if half is None else [mean - half, mean + half]
    if bounded and ci is not None:
        ci = [max(0., ci[0]), min(1., ci[1])]
    h = math.sqrt(math.log(40) / (2 * len(values)))
    return {"mean": mean, "standard_error": se, "frame_t95": ci,
            "frame_hoeffding95": [max(0., mean - h), min(1., mean + h)] if bounded else None}


def summarize(records):
    cells = []
    keys = sorted({(r["decoder"], r["sampling"], r["K"], r["ebn0_db"]) for r in records})
    for name, source, k, snr in keys:
        rows = [r for r in records if (r["decoder"], r["sampling"], r["K"], r["ebn0_db"]) == (name, source, k, snr)]
        seeds = sorted({r["seed"] for r in rows})
        unit_values = ([np.mean([r["metrics"]["pupe"] for r in rows if r["seed"] == seed]) for seed in seeds]
                       if len(seeds) > 1 else [r["metrics"]["pupe"] for r in rows])
        pupe = interval(unit_values)
        times = np.array([r["seconds"] for r in rows])
        cell = {"decoder": name, "sampling": source, "K": k, "ebn0_db": snr, "frames": len(rows),
                "pupe_definition": PUPE_DEFINITION, "pupe": pupe, "seeds": seeds,
                "confidence_unit": "seed" if len(seeds) > 1 else "frame",
                "means": {key: float(np.mean([r["metrics"][key] for r in rows])) for key in rows[0]["metrics"]},
                "decoder_seconds_per_frame": float(times.mean()), "median_seconds": float(np.median(times)),
                "p95_seconds": float(np.quantile(times, .95))}
        for key in ("conditional_excess_risk", "tail_max_error", "effective_sample_size", "excess_risk_bound", "posterior_risk",
                    "candidate_recall", "true_configuration_in_candidates", "maximum_tail_interval_width",
                    "positive_evidence_certified", "mixing_failure_bound", "support_omitted_mass_bound"):
            values = [r["meta"][key] for r in rows if r["meta"].get(key) is not None]
            if values:
                cell[key] = float(np.mean(values))
        for comparator in ("REF", "D2", "ODMA"):
            pairs = {(r["seed"], r["frame"]): r for r in records
                     if (r["decoder"], r["sampling"], r["K"], r["ebn0_db"]) == (comparator, source, k, snr)}
            if pairs and all((r["seed"], r["frame"]) in pairs for r in rows):
                differences = [r["metrics"]["pupe"] - pairs[(r["seed"], r["frame"])]["metrics"]["pupe"] for r in rows]
                if len(seeds) > 1:
                    differences = [np.mean([v for r, v in zip(rows, differences) if r["seed"] == seed]) for seed in seeds]
                cell[f"paired_delta_vs_{comparator}"] = interval(differences, bounded=False)
        cells.append(cell)
    return cells


def make_bounds(c):
    result = []
    if c["bounds"]:
        for source in c["eval_sampling"]:
            for k in c["loads"]:
                for snr in c["eval_ebn0"]:
                    upper = polyanskiy_achievability(c["B"], c["n"], k, snr, distinct=source == "distinct",
                                                    grid=c["bound_grid"], power_grid=c["bound_power_grid"],
                                                    order=c["bound_order"], details=True)
                    lower = list_fano_converse(c["B"], c["n"], k, snr, distinct=source == "distinct")
                    result.append({"sampling": source, "K": k, "ebn0_db": snr, "achievability": upper,
                                   "genie_list_fano_lower": lower})
    return result


def run_experiment(config, out_dir):
    c, out = normalized_config(config), Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(c["threads"])
    fingerprints = source_hashes()
    write_json(out / "config.json", c)
    started = time.perf_counter()
    base, encoder, phi, balance = build_encoder(c)
    encoder_seconds = time.perf_counter() - started
    matrix_hash = hashlib.sha256(phi.tobytes()).hexdigest()
    np.savez_compressed(out / "encoder.npz", phi=phi, positions=base.positions,
                        amplitudes=encoder.codebook.amplitudes().detach().numpy())
    training, prepared_global = {}, {}
    before = time.perf_counter()
    if any(name in c["decoders"] for name in ("DEC-01", "DEC-13")):
        cache = geometry(phi)
        for name in ("DEC-01", "DEC-13"):
            prepared_global[name] = {"geometry_cache": cache}
    query_setup = time.perf_counter() - before
    if "D2" in c["decoders"]:
        torch.manual_seed(c["seed"] + 400_000)
        decoder = PolarD2(c["layers"]).double()
        d2_config = {**c, "loads": c["train_loads"], "decoder": "d2", "mode": "fixed", "family": "polar_decoder_bank"}
        d2_dir = out / "D2"; d2_dir.mkdir()
        if c["d2_checkpoint"]:
            checkpoint = torch.load(c["d2_checkpoint"], map_location="cpu", weights_only=True)
            if checkpoint.get("matrix_sha256") != matrix_hash:
                raise ValueError("D2 checkpoint does not match this exact fixed encoder")
            training_keys = ("loads", "train_sampling", "train_ebn0", "max_epochs", "batches_per_epoch", "batch_size",
                             "validation_batches", "patience", "layers", "lr", "seed")
            if any(checkpoint["training_config"][key] != d2_config[key] for key in training_keys):
                raise ValueError("D2 checkpoint training contract differs; use its recorded training settings")
            decoder.load_state_dict(checkpoint["decoder"], strict=True)
            training["D2"] = {**checkpoint["training"], "checkpoint_reused": str(c["d2_checkpoint"])}
        else:
            training["D2"] = train(d2_config, encoder, decoder, d2_dir)
        decoder.eval()
        training["D2"]["trainable_parameters"] = sum(p.numel() for p in decoder.parameters())
        torch.save({"decoder": decoder.state_dict(), "matrix_sha256": matrix_hash, "training": training["D2"],
                    "training_config": d2_config}, d2_dir / "decoder.pt")
        before = time.perf_counter()
        decoder._gram_geometry(encoder)
        prepared_global["D2"] = {"encoder": encoder, "decoder": decoder}
        d2_setup = time.perf_counter() - before
    else:
        d2_setup = 0.0
    for name in c["decoders"]:
        training.setdefault(name, {"status": "not_applicable", "trainable_parameters": 0, "seconds": 0.0})
    prepared_global["ODMA"] = {"baseline": base, "amplitudes": encoder.codebook.amplitudes().detach().numpy()}
    records, setup = [], []
    (out / "inputs").mkdir()
    with (out / "frames.jsonl").open("w") as stream:
        for source in c["eval_sampling"]:
            for k in c["loads"]:
                ids, standard_noise, signals = fixture(c, phi, k, source)
                prepared = dict(prepared_global)
                if "REF" in c["decoders"]:
                    before = time.perf_counter()
                    ref_bank = prepare_reference(phi, k, source, c["reference_max_states"], c["reference_max_bytes"])
                    prepared["REF"] = {"prepared": ref_bank}
                    setup.append({"decoder": "REF", "sampling": source, "K": k, "seconds": time.perf_counter() - before,
                                  "states": len(ref_bank["tuples"]), "estimated_bytes": ref_bank["estimated_bytes"]})
                for snr in c["eval_ebn0"]:
                    variance = noise_variance(c["B"], snr)
                    observations = signals + math.sqrt(variance) * standard_noise
                    cell_name = f"{source}_K{k}_EbN0{float(snr):g}"
                    np.savez_compressed(out / "inputs" / f"{cell_name}.npz", message_ids=ids,
                                        standard_noise=standard_noise, y=observations, noise_var=variance)
                    order = sorted(c["decoders"], key=lambda name: name != "REF")
                    for frame, y in enumerate(observations):
                        y.setflags(write=False)
                        scenario = Scenario(phi, y, k, variance, source)
                        ref_tails = ref_risk = ref_log_z = None
                        for name in order:
                            params = {**dict(REGISTRY[name].defaults), **c["decoder_params"].get(name, {}),
                                      **prepared.get(name, {})}
                            if REGISTRY[name].stochastic:
                                params["seed"] = int(np.random.SeedSequence([c["seed"], 600_000 + int(name[4:]), k,
                                                                           int(source == "iid"), frame]).generate_state(1)[0])
                            before = time.perf_counter()
                            counts, meta = run(name, scenario, **params)
                            elapsed = time.perf_counter() - before
                            tails = meta.pop("tails", None)
                            upper_tails = meta.pop("tail_upper", None)
                            if upper_tails is not None:
                                meta["maximum_tail_interval_width"] = float(np.max(upper_tails - tails))
                            if name == "REF":
                                ref_tails, ref_risk, ref_log_z = tails, meta["posterior_risk"], meta["log_evidence"]
                            if ref_tails is not None:
                                risk = 1 - sum(ref_tails[w, :count].sum() for w, count in enumerate(counts) if count) / k
                                excess = float(risk - ref_risk)
                                if excess < -1e-9:
                                    raise AssertionError("A reported action beats the exact conditional Bayes risk")
                                meta["conditional_excess_risk"] = max(0., excess)
                                if tails is not None:
                                    meta["tail_max_error"] = float(np.max(np.abs(tails - ref_tails)))
                                if "elbo" in meta:
                                    meta["exact_evidence_variational_bound"] = min(
                                        1., 2 * math.sqrt(max(0., ref_log_z - meta["elbo"]) / 2))
                            selected = np.flatnonzero(counts)
                            decoded = message_bits(np.repeat(selected, counts[selected]), c["B"])
                            metrics = frame_metrics(message_bits(ids[frame], c["B"]), decoded)
                            if "candidate_ids" in meta:
                                meta["candidate_recall"] = float(np.isin(ids[frame], meta["candidate_ids"]).mean())
                                meta["true_configuration_in_candidates"] = bool(np.isin(ids[frame], meta["candidate_ids"]).all())
                            record = {"decoder": name, "sampling": source, "K": k, "ebn0_db": snr,
                                      "seed": c["seed"], "frame": frame, "seconds": elapsed, "metrics": metrics,
                                      "counts": [[int(w), int(counts[w])] for w in selected], "meta": meta,
                                      "input_sha256": hashlib.sha256(y.tobytes()).hexdigest()}
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            records.append(record)
                    stream.flush()
                    values = {name: np.mean([r["metrics"]["pupe"] for r in records
                                             if (r["decoder"], r["sampling"], r["K"], r["ebn0_db"]) == (name, source, k, snr)])
                              for name in c["decoders"]}
                    print(json.dumps({"sampling": source, "K": k, "ebn0_db": snr, "pupe": values}), flush=True)
    before = time.perf_counter()
    bounds = make_bounds(c)
    bound_seconds = time.perf_counter() - before
    if fingerprints != source_hashes():
        raise RuntimeError("Source changed during the run; preserve partial files and rerun in a stable checkout")
    result = {"schema_version": 1, "status": "complete", "config": c, "pupe_definition": PUPE_DEFINITION,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": fingerprints, "matrix_sha256": matrix_hash, "baseline": base.metadata(),
              "encoder": {"mapping": c["mapping"], "balance": balance, "dtype": str(phi.dtype),
                          "min_energy": float(np.square(phi).sum(0).min()), "max_energy": float(np.square(phi).sum(0).max()),
                          "seconds": encoder_seconds, "explicit_matrix_bytes": phi.nbytes},
              "training": training, "setup": setup, "D2_geometry_seconds": d2_setup, "bound_seconds": bound_seconds,
              "query_geometry_seconds": query_setup,
              "evaluation": summarize(records), "bounds": bounds,
              "decoder_params": {name: {**dict(REGISTRY[name].defaults), **c["decoder_params"].get(name, {})} for name in c["decoders"]},
              "physical_noise_convention": "sigma²=1/(2 B 10^(Eb/N0/10)); unit-energy real AWGN",
              "frame_policy": "SeedSequence(seed,300000,K,iid,frame); shared truth/noise across decoders and SNR",
              "confidence": "95% t intervals over frames for one seed, seed means for multiple seeds; Hoeffding also stored",
              "scope": "tiny_smoke_not_production_ranking" if c["profile"] == "smoke" else "research_run"}
    write_json(out / "summary.json", result)
    save_tables(result, out)
    plots(result, out)
    return result


def save_tables(result, out):
    columns = ["decoder", "sampling", "K", "ebn0_db", "frames", "pupe", "pupe_t95_low", "pupe_t95_high",
               "decoder_seconds_per_frame", "median_seconds", "p95_seconds", "conditional_excess_risk", "tail_max_error",
               "effective_sample_size", "excess_risk_bound", "candidate_recall", "true_configuration_in_candidates",
               "maximum_tail_interval_width", "positive_evidence_certified", "mixing_failure_bound"]
    with (out / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for cell in result["evaluation"]:
            row = {k: cell.get(k) for k in columns}
            row["pupe"] = cell["pupe"]["mean"]
            row["pupe_t95_low"], row["pupe_t95_high"] = cell["pupe"]["frame_t95"] or (None, None)
            writer.writerow(row)


def plots(result, out):
    c, cells = result["config"], result["evaluation"]
    colors = {name: plt.get_cmap("tab20")(i) for i, name in enumerate(REGISTRY)}
    colors.update(dict(zip(("REF", "DEC-02", "DEC-05", "D2", "ODMA"),
                           ("#222222", "#b64a00", "#007c91", "#6b4ba1", "#277c38"))))
    for source in sorted({r["sampling"] for r in cells}):
        snrs = sorted({r["ebn0_db"] for r in cells if r["sampling"] == source})
        for metric, filename in (("pupe", "pupe_vs_users"), ("runtime", "runtime_vs_users")):
            cols = min(3, len(snrs)); rows = math.ceil(len(snrs) / cols)
            legend_cols = 4 if len(c["decoders"]) > 8 else 3
            legend_rows = math.ceil((len(c["decoders"]) + 2) / legend_cols)
            height = 3.6 * rows + max(1.5, .24 * legend_rows + .6)
            fig, axes = plt.subplots(rows, cols, figsize=(max(10.0, 5.0 * cols), height), squeeze=False)
            for ax, snr in zip(axes.flat, snrs):
                for name in c["decoders"]:
                    points = sorted((r for r in cells if (r["decoder"], r["sampling"], r["ebn0_db"]) == (name, source, snr)),
                                    key=lambda r: r["K"])
                    if not points:
                        continue
                    xs = [r["K"] for r in points]
                    ys = [r["pupe"]["mean"] if metric == "pupe" else 1000 * r["decoder_seconds_per_frame"] for r in points]
                    label = REGISTRY[name].label
                    if name == "ODMA":
                        label += " (energy adaptation)" if c["mapping"] == "balanced" else " (paper adaptation)"
                    if metric == "pupe" and all(r["pupe"]["frame_t95"] is not None for r in points):
                        errors = np.array([[max(0, r["pupe"]["mean"] - r["pupe"]["frame_t95"][0]),
                                            max(0, r["pupe"]["frame_t95"][1] - r["pupe"]["mean"])] for r in points]).T
                        ax.errorbar(xs, ys, yerr=errors, fmt="o-", markersize=4, capsize=2, color=colors.get(name), label=label)
                    else:
                        ax.plot(xs, ys, "o-", markersize=4, color=colors.get(name), label=label)
                if metric == "pupe":
                    bs = sorted((r for r in result["bounds"] if (r["sampling"], r["ebn0_db"]) == (source, snr)),
                                key=lambda r: r["K"])
                    if bs:
                        ax.plot([r["K"] for r in bs], [r["achievability"]["upper_bound"] for r in bs], "--",
                                color="#7a6250", label="Achievability upper (not a floor)")
                        ax.plot([r["K"] for r in bs], [r["genie_list_fano_lower"] for r in bs], ":",
                                color="#777777", linewidth=2, label="Genie/list-Fano lower (weak)")
                    ax.set(ylim=(-.025, 1.025), ylabel="Multiset PUPE")
                else:
                    ax.set(yscale="log", ylabel="Mean decode time / frame (ms)")
                ax.set(title=f"Physical Eb/N0 = {snr:g} dB", xlabel="Active users K")
                ax.set_xticks(sorted({r["K"] for r in cells})); ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(snrs):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=legend_cols, fontsize=8)
            fig.suptitle(f"{c['mapping'].capitalize()} polar/CRC | B={c['B']}, n={c['n']} | {source} | {result['scope'].replace('_', ' ')}",
                         fontsize=12)
            unit = "seed" if any(len(r["seeds"]) > 1 for r in cells) else "frame"
            footer = (f"95% {unit} t intervals; zero observed errors do not prove zero PUPE. Bounds concern coding limits, REF the fixed bank."
                      if metric == "pupe" else "Decode timing includes readout; encoder, REF preparation and D2 training costs are stored separately.")
            fig.text(.5, .015, footer, ha="center", fontsize=8)
            fig.tight_layout(rect=(0, max(.20, (.55 + .17 * legend_rows) / height), 1, .94))
            fig.savefig(out / f"{filename}_{source}.png", dpi=170)
            fig.savefig(out / f"{filename}_{source}.pdf")
            plt.close(fig)
    learned = result["training"].get("D2", {})
    if learned.get("history"):
        history = learned["history"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        epochs = [0] + [r["epoch"] for r in history]
        for ax, key, label in zip(axes, ("loss", "pupe"), ("Validation training objective", "Validation multiset PUPE")):
            values = [learned["initial"][key]] + [r["validation"][key] for r in history]
            ax.plot(epochs, values, "o-", markersize=3, color="#6b4ba1")
            ax.set(xlabel="D2 training epoch", ylabel=label); ax.grid(alpha=.2)
        fig.suptitle("D2 learning on separate training/validation data; encoder frozen")
        fig.tight_layout()
        fig.savefig(out / "learning_D2.png", dpi=170); plt.close(fig)
    if len(c["decoders"]) > 8 and any("conditional_excess_risk" in r for r in cells):
        names = [name for name in c["decoders"] if name != "REF"]
        for source in c["eval_sampling"]:
            columns = sorted({(r["ebn0_db"], r["K"]) for r in cells if r["sampling"] == source})
            fig, axes = plt.subplots(1, 2, figsize=(13, max(4., .28 * len(names) + 1.5)))
            for ax, metric, title in zip(axes, ("conditional_excess_risk", "tail_max_error"),
                                         ("Conditional PUPE regret", "Maximum count-tail error")):
                values = np.full((len(names), len(columns)), np.nan)
                for r in cells:
                    if r["sampling"] == source and r["decoder"] in names and metric in r:
                        values[names.index(r["decoder"]), columns.index((r["ebn0_db"], r["K"]))] = r[metric]
                color = plt.get_cmap("magma").copy(); color.set_bad("#dddddd")
                chart = ax.imshow(np.ma.masked_invalid(values), aspect="auto", vmin=0, vmax=1, cmap=color)
                ax.set_yticks(range(len(names)), names)
                ax.set_xticks(range(len(columns)), [f"{snr:g} dB\nK{k}" for snr, k in columns], fontsize=7)
                ax.set_title(title); fig.colorbar(chart, ax=ax, shrink=.8)
            fig.suptitle(f"Exact-reference diagnostics | {source} | gray = not reported; evaluated after inference")
            fig.tight_layout(); fig.savefig(out / f"reference_diagnostics_{source}.png", dpi=170); plt.close(fig)
    elif any(r["decoder"] != "REF" and "conditional_excess_risk" in r for r in cells):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(axes, ("conditional_excess_risk", "tail_max_error"),
                                     ("Conditional PUPE regret against REF", "Maximum count-tail error against REF")):
            for name in c["decoders"]:
                if name == "REF":
                    continue
                for snr in sorted({r["ebn0_db"] for r in cells}):
                    points = sorted((r for r in cells if r["decoder"] == name and r["ebn0_db"] == snr
                                     and r["sampling"] == c["eval_sampling"][0] and metric in r), key=lambda r: r["K"])
                    if points:
                        ax.plot([r["K"] for r in points], [r[metric] for r in points], "o-", markersize=3,
                                label=f"{name}, {snr:g} dB")
            ax.set(xlabel="Active users K", ylabel=title); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "Not reported by selected decoders", ha="center", transform=ax.transAxes)
        fig.suptitle(f"Exact-posterior diagnostics | {c['eval_sampling'][0]} | evaluation only, never supplied to decoders")
        fig.tight_layout(); fig.savefig(out / "reference_diagnostics.png", dpi=170); plt.close(fig)


def merge_runs(paths, out_dir):
    if not paths:
        raise ValueError("Provide at least one original run directory")
    results = [json.loads((Path(p) / "summary.json" if Path(p).is_dir() else Path(p)).read_text()) for p in paths]
    first = results[0]
    ignored = {"name", "seed", "decoders", "decoder_params", "loads", "eval_ebn0", "d2_checkpoint"}
    contract = {k: v for k, v in first["config"].items() if k not in ignored}
    seen, records, parameters, matrices = set(), [], {}, {}
    bounds = {}
    for path, result in zip(paths, results):
        if "merged_from" in result:
            raise ValueError("Merge original run directories; nested aggregates are not supported")
        if result["status"] != "complete" or result["pupe_definition"] != PUPE_DEFINITION:
            raise ValueError("Only complete results with the current metric may be merged")
        if contract != {k: v for k, v in result["config"].items() if k not in ignored}:
            raise ValueError("Incompatible experiment/training contracts")
        if result["source_sha256"] != first["source_sha256"]:
            raise ValueError("Source hashes differ; do not silently mix implementations")
        seed = result["config"]["seed"]
        if matrices.setdefault(seed, result["matrix_sha256"]) != result["matrix_sha256"]:
            raise ValueError("Unmatched codebooks for the same seed")
        for name, params in result["decoder_params"].items():
            if parameters.setdefault(name, params) != params:
                raise ValueError(f"Different {name} budgets/variants need separate comparison labels")
        folder = Path(path) if Path(path).is_dir() else Path(path).parent
        config = result["config"]
        frame_count = 0
        for line in (folder / "frames.jsonl").read_text().splitlines():
            row = json.loads(line)
            if (row["decoder"] not in config["decoders"] or row["seed"] != seed
                    or row["sampling"] not in config["eval_sampling"] or row["K"] not in config["loads"]
                    or row["ebn0_db"] not in config["eval_ebn0"] or row["frame"] not in range(config["eval_frames"])):
                raise ValueError("Frame record does not match its saved configuration")
            key = (row["decoder"], row["seed"], row["sampling"], row["K"], row["ebn0_db"], row["frame"])
            if key in seen:
                raise ValueError(f"Duplicate result frame {key}")
            seen.add(key); records.append(row)
            frame_count += 1
        expected = math.prod(len(config[key]) for key in ("decoders", "eval_sampling", "loads", "eval_ebn0"))
        if frame_count != expected * config["eval_frames"]:
            raise ValueError(f"Incomplete frame records in {folder}")
        for b in result["bounds"]:
            key = (b["sampling"], b["K"], b["ebn0_db"])
            if bounds.setdefault(key, b) != b:
                raise ValueError("Different bounds for the same cell")
    for source, k, snr in {(r["sampling"], r["K"], r["ebn0_db"]) for r in records}:
        by_decoder = [{r["seed"] for r in records if (r["decoder"], r["sampling"], r["K"], r["ebn0_db"]) == (name, source, k, snr)}
                      for name in parameters]
        nonempty = [s for s in by_decoder if s]
        if any(s != nonempty[0] for s in nonempty):
            raise ValueError("Decoder comparisons in a cell must use the same seed set")
    paired = {}
    for row in records:
        key = (row["seed"], row["sampling"], row["K"], row["ebn0_db"], row["frame"])
        if paired.setdefault(key, row["input_sha256"]) != row["input_sha256"]:
            raise ValueError("Unpaired observations in decoder job rows")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = {**first["config"], "decoders": [n for n in REGISTRY if n in parameters],
              "loads": sorted({r["K"] for r in records}), "eval_ebn0": sorted({r["ebn0_db"] for r in records}),
              "seed": next(iter(matrices)) if len(matrices) == 1 else None}
    result = {**first, "config": config, "evaluation": summarize(records), "bounds": list(bounds.values()),
              "decoder_params": parameters, "merged_from": [str(p) for p in paths],
              "seeds": sorted(matrices), "matrix_sha256_by_seed": {str(seed): value for seed, value in matrices.items()},
              "matrix_sha256": first["matrix_sha256"] if len(matrices) == 1 else None,
              "encoder": first["encoder"] if len(matrices) == 1 else None,
              "setup": [], "D2_geometry_seconds": None, "bound_seconds": None,
              "training": ({name: row for r in results for name, row in r["training"].items()} if len(matrices) == 1 else {}),
              "run_metadata": [{"seed": r["config"]["seed"], "matrix_sha256": r["matrix_sha256"], "encoder": r["encoder"],
                                "training": r["training"], "setup": r["setup"], "D2_geometry_seconds": r["D2_geometry_seconds"],
                                "query_geometry_seconds": r.get("query_geometry_seconds", 0.),
                                "bound_seconds": r["bound_seconds"]}
                               for r in results]}
    (out / "frames.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    write_json(out / "summary.json", result); save_tables(result, out); plots(result, out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-decoders", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--decoders", "--decoder", nargs="+")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--plot-only", type=Path, help="Existing run directory; regenerate its plots from saved summaries")
    parser.add_argument("--mapping", choices=("uniform", "balanced"))
    parser.add_argument("--sampling", nargs="+", choices=("iid", "distinct"), dest="eval_sampling")
    parser.add_argument("--loads", nargs="+", type=int)
    parser.add_argument("--train-loads", nargs="+", type=int)
    parser.add_argument("--ebn0", nargs="+", type=float, dest="eval_ebn0")
    parser.add_argument("--train-ebn0", nargs=2, type=float)
    parser.add_argument("--decoder-params", type=json.loads, help='JSON, e.g. {"DEC-05":{"samples":4096}}')
    parser.add_argument("--d2-checkpoint")
    parser.add_argument("--no-bounds", action="store_true")
    for flag, dest in (("frames", "eval_frames"), ("B", "B"), ("n", "n"), ("seed", "seed"), ("prefix-bits", "prefix_bits"),
                       ("code-length", "code_length"), ("crc-bits", "crc_bits"), ("list-size", "list_size"),
                       ("native-iterations", "native_iterations"), ("max-epochs", "max_epochs"), ("patience", "patience"),
                       ("batches-per-epoch", "batches_per_epoch"), ("batch-size", "batch_size"), ("layers", "layers"),
                       ("validation-batches", "validation_batches"), ("threads", "threads"),
                       ("reference-max-states", "reference_max_states")):
        parser.add_argument("--" + flag, dest=dest, type=int)
    args = parser.parse_args(argv)
    if args.list_decoders:
        print(json.dumps({name: {"label": s.label, "implemented": s.run is not None, "sources": s.sources,
                                "parameters": dict(s.defaults)} for name, s in REGISTRY.items()}, indent=2)); return
    if args.plot_only:
        result = json.loads((args.plot_only / "summary.json").read_text()); plots(result, args.plot_only); return
    if args.out_dir is None:
        parser.error("--out-dir is required")
    if args.merge:
        merge_runs(args.merge, args.out_dir); return
    config = json.loads(args.config.read_text()) if args.config else {}
    excluded = {"list_decoders", "config", "out_dir", "merge", "plot_only", "no_bounds"}
    config.update({k: v for k, v in vars(args).items() if k not in excluded and v is not None})
    if args.no_bounds:
        config["bounds"] = False
    run_experiment(config, args.out_dir)
    print(f"Results and plots: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
