"""Matched fixed-resource signature/energy comparison; reuse the existing decoder bank.

This job-local extension leaves historical benchmark sources and their hash contracts unchanged.
The explicit D2 adapter and the two-user genie are small-alphabet research instruments.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import logsumexp
import torch

from baselines import message_bits
from benchmarks import polar_decoders as common
from benchmarks.polar_energy_geometry import diagnostics
from benchmarks.ura_comparison import frame_metrics, noise_variance, train
from framework.polar_decoders import REGISTRY, Scenario, run
from framework.polar_mapping import PolarD2, PolarExplicitEncoder, PolarMappedCodebook
from framework.polar_tractable import PolarModel
from src.metrics import PUPE_DEFINITION


ROOT = Path(__file__).resolve().parents[2]
JOB = Path(__file__).resolve().parent
REFERENCE_KEYS = [0, 36, 27, 49, 16, 12, 59, 41, 39, 2, 62, 21, 51, 46, 26, 9,
                  56, 12, 3, 57, 40, 36, 35, 33, 31, 42, 38, 29, 11, 6, 2, 1]
REPLACEMENTS = {17: 8, 21: 20, 30: 61}
TRAIN_KEYS = ("loads", "train_sampling", "train_ebn0", "max_epochs", "batches_per_epoch", "batch_size",
              "validation_batches", "patience", "layers", "lr", "seed")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_hashes():
    paths = [*JOB.glob("*.py"), JOB / "038_polar_signatures.sh", JOB / "manifest.jsonl",
             ROOT / "jobs/037_polar_tractable/reference_contract.json"]
    return {**common.source_hashes(), **{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(paths) if p.is_file()}}


class RepairedSignature:
    """Six-bit linear signature with three repeated character rows replaced; no native SCL decoder."""

    def __init__(self, reference):
        generator = reference.encode(np.eye(reference.payload_bits, dtype=np.uint8)).T
        keys = (generator @ (1 << np.arange(reference.payload_bits - 1, -1, -1))).tolist()
        if keys != REFERENCE_KEYS:
            raise ValueError("The certified repair applies only to the recorded b6/Nc32/CRC8 generator")
        self.n, self.payload_bits = 32, 6
        # DEC-26 constructs an unused PolarCode even in basis-only mode. Keep its provenance field;
        # the job rejects SCL/union methods, and all scored words use this modified encode operation.
        self.crc_bits = reference.crc_bits
        for index, key in REPLACEMENTS.items():
            keys[index] = key
        self.generator = message_bits(np.array(keys), self.payload_bits)

    def encode(self, payload):
        payload = np.asarray(payload)
        if payload.shape[-1] != self.payload_bits or np.any((payload != 0) & (payload != 1)):
            raise ValueError("Expected binary six-bit payloads")
        return (payload.astype(np.uint8) @ self.generator.T) % 2

    def decode_list(self, *args, **kwargs):
        raise ValueError("The repaired signature is not compatible with the original CRC/SCL receiver")


class ArchivedAmplitudeBook(PolarMappedCodebook):
    """Preserve historical float64 amplitudes exactly across platforms and both signature arms."""

    def __init__(self, base, mapping, amplitudes):
        super().__init__(base, mapping)
        self.register_buffer("archived_amplitudes", torch.from_numpy(amplitudes.copy()))

    def amplitudes(self):
        return self.archived_amplitudes


def audited_summary(directory):
    summary = json.loads((directory / "summary.json").read_text())
    contract = json.loads((ROOT / "jobs/037_polar_tractable/reference_contract.json").read_text())
    allowed = [contract["current_source_sha256"], *[v["source_sha256"] for v in contract["historical_snapshots"]]]
    if common.source_hashes() != contract["current_source_sha256"]:
        raise ValueError("Shared runtime changed since the historical compatibility audit")
    if summary["status"] != "complete" or summary["source_sha256"] not in allowed:
        raise ValueError("Historical run source/status is outside the audited contract")
    return summary


def build_encoder(config, signature, archive=None):
    base, encoder, phi, balance = common.build_encoder(config)
    amplitudes = None
    if archive is not None:
        summary = audited_summary(archive)
        if any(summary["config"][k] != config[k] for k in ("B", "n", "seed", "prefix_bits", "code_length", "crc_bits", "mapping")):
            raise ValueError("Historical encoder configuration differs")
        saved = np.load(archive / "encoder.npz")
        amplitudes = saved["amplitudes"]
        if digest(saved["phi"]) != summary["matrix_sha256"] or not np.array_equal(saved["positions"], base.positions):
            raise ValueError("Historical encoder archive failed matrix/placement audit")
        if not np.allclose(amplitudes, encoder.codebook.amplitudes().numpy(), rtol=0, atol=1e-14):
            raise ValueError("Historical amplitudes differ by more than numerical roundoff from the declared construction")
        encoder.codebook = ArchivedAmplitudeBook(base, encoder.codebook.mapping, amplitudes).double()
        phi = encoder.explicit_matrix().detach().numpy().copy()
        if not np.array_equal(phi, saved["phi"]):
            raise ValueError("Historical amplitudes do not reproduce the reference waveform matrix exactly")
        balance = summary["encoder"]["balance"]
    if signature == "repair3":
        base.polar = RepairedSignature(base.polar)
        original = encoder
        encoder = PolarExplicitEncoder(base, config["train_loads"], original.codebook.mapping).double()
        with torch.no_grad():
            encoder.codebook.values.copy_(original.codebook.values)
        if amplitudes is not None:
            encoder.codebook = ArchivedAmplitudeBook(base, encoder.codebook.mapping, amplitudes).double()
        phi = encoder.explicit_matrix().detach().numpy().copy()
        phi.setflags(write=False)
    elif signature != "reference":
        raise ValueError("Unknown signature")
    if not np.allclose(np.square(phi).sum(0), 1, rtol=0, atol=1e-12):
        raise ValueError("Per-message energy differs from one")
    phi.setflags(write=False)
    return base, encoder, phi, balance


def pair_posterior(left, right, y, variance, same_pattern):
    """Ordered IID uniform payload pair, with both patterns and all outside users known."""
    cross = left.T @ right
    norm = np.square(left).sum(0)[:, None] + np.square(right).sum(0)[None, :] + 2 * cross
    score = ((y @ left)[:, None] + (y @ right)[None, :] - norm / 2) / variance
    weights = np.exp(score - logsumexp(score))
    if same_pattern:
        diagonal = np.diag(weights)
        tails = np.column_stack((weights.sum(0) + weights.sum(1) - diagonal, diagonal))
    else:
        tails = np.column_stack((np.r_[weights.sum(1), weights.sum(0)], np.zeros(left.shape[1] + right.shape[1])))
    # Tail gains decrease with rank; stable flatten ordering also handles ties legally.
    chosen = np.argsort(-tails.ravel(), kind="stable")[:2] // 2
    return weights, tails, np.bincount(chosen, minlength=len(tails)), norm


def pair_diagnostic(phi, positions, ids, noise, variance, payloads=64):
    """First two IID slots: source-weighted, no selection using payloads, noise or decoded output."""
    before = time.perf_counter()
    p, q = ids[:2] // payloads
    u, v = ids[:2] % payloads
    left = phi[:, p * payloads:(p + 1) * payloads]
    right = phi[:, q * payloads:(q + 1) * payloads]
    signal = left[:, u] + right[:, v]
    weights, tails, counts, norm = pair_posterior(left, right, signal + noise, variance, p == q)
    distances = np.maximum(0., norm + signal @ signal - 2 * ((signal @ left)[:, None] + (signal @ right)[None, :]))
    allowed = np.ones(weights.shape, dtype=bool) if p != q else np.triu(np.ones(weights.shape, dtype=bool))
    allowed[u, v] = False
    if p == q:
        allowed[v, u] = False
    truth = Counter([int(u), int(v if p == q else payloads + v)])
    gain = sum(min(c, int(counts[w])) for w, c in truth.items())
    probability = weights[u, v] + (weights[v, u] if p == q and u != v else 0.)
    return {"scope": "oracle_outside_and_two_patterns_IID_ordered_uniform_payloads", "same_pattern": bool(p == q),
            "overlap": int(len(np.intersect1d(positions[p], positions[q]))),
            "minimum_competing_distance2": float(distances[allowed].min()),
            "competitors_within_4sigma2": int(np.sum(distances[allowed] <= 4 * variance)),
            "true_pair_posterior": float(probability), "pair_count_pupe": 1 - gain / 2,
            "conditional_bayes_pair_risk": float(1 - sum(tails[w, :c].sum() for w, c in enumerate(counts)) / 2),
            "ordered_payload_states": weights.size, "seconds": time.perf_counter() - before}


def prepare_d2(c, encoder, matrix_hash, out, checkpoint_path):
    torch.manual_seed(c["seed"] + 400_000)
    decoder = PolarD2(c["layers"]).double()
    training_config = {**c, "loads": c["train_loads"], "decoder": "d2", "mode": "fixed", "family": "polar_decoder_bank"}
    out.mkdir()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        summary = audited_summary(checkpoint_path.parent.parent)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("matrix_sha256") != matrix_hash or summary["matrix_sha256"] != matrix_hash:
            raise ValueError("Historical D2 checkpoint does not match this physical matrix")
        if any(checkpoint["training_config"][key] != training_config[key] for key in TRAIN_KEYS):
            raise ValueError("Historical D2 training contract differs")
        decoder.load_state_dict(checkpoint["decoder"], strict=True)
        training = {**checkpoint["training"], "checkpoint_reused": str(checkpoint_path),
                    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "seconds_this_run": 0.}
    else:
        training = train(training_config, encoder, decoder, out)
        training["seconds_this_run"] = training["seconds"]
    decoder.eval()
    torch.save({"decoder": decoder.state_dict(), "matrix_sha256": matrix_hash, "training": training,
                "training_config": training_config}, out / "decoder.pt")
    before = time.perf_counter()
    decoder._gram_geometry(encoder)
    return decoder, training, time.perf_counter() - before


def run_experiment(row, out, checkpoint_path=None, archive=None):
    c, signature, out = common.normalized_config(row["config"]), row["signature"], Path(out)
    if c["B"] - c["prefix_bits"] != 6 or c["code_length"] != 32 or c["crc_bits"] != 8:
        raise ValueError("This comparison fixes the certified six-bit payload fixture")
    if c["eval_sampling"] != ["iid"] or not set(c["decoders"]) <= {"D2", "DEC-26"}:
        raise ValueError("Use IID counts with D2 and/or the compatible parity-basis DEC-26")
    if c["decoder_params"].get("DEC-26", {}).get("candidate_method", "basis") != "basis":
        raise ValueError("The modified signature requires basis-only proposals; native SCL is incompatible")
    if min(c["loads"]) < 2 or not 1 <= row["pair_frames"] <= c["eval_frames"]:
        raise ValueError("Pair diagnostics require K>=2 and a positive frame prefix within the evaluated frames")
    if row["reference_policy"] == "archived_job035" and archive is None:
        raise ValueError("This row requires the audited historical encoder archive")
    if row["reference_policy"] not in ("archived_job035", "regenerated"):
        raise ValueError("Unknown reference construction policy")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    fingerprints = source_hashes()
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "config.json", row)
    torch.set_num_threads(c["threads"])
    before = time.perf_counter()
    base, encoder, phi, balance = build_encoder(c, signature, archive)
    model = PolarModel(base.polar, base.positions, encoder.codebook.amplitudes().detach().numpy(), c["n"])
    matrix_hash = digest(phi)
    geometry = diagnostics(encoder.codebook)
    encoder_seconds = time.perf_counter() - before
    np.savez_compressed(out / "encoder.npz", phi=phi, positions=base.positions, amplitudes=model.amplitudes,
                        generator=base.polar.encode(np.eye(6, dtype=np.uint8)).T)
    prepared, training, d2_setup = {"DEC-26": {"model": model}}, {}, 0.
    if "D2" in c["decoders"]:
        decoder, training, d2_setup = prepare_d2(c, encoder, matrix_hash, out / "D2", checkpoint_path)
        prepared["D2"] = {"encoder": encoder, "decoder": decoder}
    records, pairs = [], []
    (out / "inputs").mkdir()
    with (out / "frames.jsonl").open("w") as stream, (out / "pairs.jsonl").open("w") as pair_stream:
        for k in c["loads"]:
            ids, standard_noise, signals = common.fixture(c, phi, k, "iid")
            for snr in c["eval_ebn0"]:
                variance = noise_variance(c["B"], snr)
                observations = signals + math.sqrt(variance) * standard_noise
                np.savez_compressed(out / "inputs" / f"iid_K{k}_EbN0{float(snr):g}.npz", message_ids=ids,
                                    standard_noise=standard_noise, y=observations, noise_var=variance)
                for frame, y in enumerate(observations):
                    key = {"sampling": "iid", "K": k, "ebn0_db": snr, "seed": c["seed"], "frame": frame}
                    if frame < row["pair_frames"]:
                        pair = {**key, **pair_diagnostic(phi, base.positions, ids[frame],
                                                        math.sqrt(variance) * standard_noise[frame], variance)}
                        pair_stream.write(json.dumps(pair, allow_nan=False) + "\n")
                        pairs.append(pair)
                    scenario = Scenario(phi, y, k, variance, "iid")
                    for name in c["decoders"]:
                        params = {**c["decoder_params"].get(name, {}), **prepared[name]}
                        if REGISTRY[name].stochastic:
                            params["seed"] = int(np.random.SeedSequence([c["seed"], 600_000 + int(name[4:]), k,
                                                                       1, frame]).generate_state(1)[0])
                        before = time.perf_counter()
                        counts, meta = run(name, scenario, **params)
                        seconds = time.perf_counter() - before
                        if "candidate_ids" in meta:
                            hits = np.isin(ids[frame], meta.pop("candidate_ids"))
                            meta["candidate_recall"] = float(hits.mean())
                            meta["true_configuration_in_candidates"] = bool(hits.all())
                        meta.pop("tail_probabilities", None)
                        selected = np.flatnonzero(counts)
                        decoded = message_bits(np.repeat(selected, counts[selected]), c["B"])
                        record = {**key, "decoder": name, "seconds": seconds, "meta": meta,
                                  "metrics": frame_metrics(message_bits(ids[frame], c["B"]), decoded),
                                  "counts": [[int(w), int(counts[w])] for w in selected], "input_sha256": digest(y),
                                  "truth_sha256": digest(ids[frame]), "standard_noise_sha256": digest(standard_noise[frame])}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        records.append(record)
                stream.flush(); pair_stream.flush()
                print(json.dumps({"signature": signature, "mapping": c["mapping"], "K": k, "ebn0_db": snr,
                                  "completed_frames": c["eval_frames"]}), flush=True)
    if fingerprints != source_hashes():
        raise RuntimeError("Source changed during run; preserve partial output and use a stable checkout")
    result = {"status": "complete", "row": row, "config": c, "signature": signature,
              "source_sha256": fingerprints, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                                cwd=ROOT, text=True).strip(),
              "pupe_definition": PUPE_DEFINITION, "matrix_sha256": matrix_hash, "training": training,
              "reference_archive": None if archive is None else {"path": str(archive),
                  "encoder_sha256": hashlib.sha256((archive / "encoder.npz").read_bytes()).hexdigest(),
                  "summary_sha256": hashlib.sha256((archive / "summary.json").read_bytes()).hexdigest()},
              "decoder_params": {name: {**dict(REGISTRY[name].defaults), **c["decoder_params"].get(name, {})}
                                 for name in c["decoders"]},
              "encoder": {"balance": balance, "moments": geometry, "seconds": encoder_seconds,
                          "explicit_matrix_bytes": phi.nbytes, "generator_keys": model.keys,
                          "positions_sha256": digest(base.positions), "amplitudes_sha256": digest(model.amplitudes),
                          "per_message_energy": [float(np.square(phi).sum(0).min()), float(np.square(phi).sum(0).max())]},
              "D2_geometry_seconds": d2_setup, "pair_diagnostic_seconds": sum(p["seconds"] for p in pairs),
              "evaluation": common.summarize(records), "scope": row["scope"],
              "assumptions": "IID uniform full messages; known K/noise; real AWGN; unit power; fixed n,B,Nc,placements",
              "receiver_boundary": "D2 is explicit-M; DEC-26 is procedural inside the explicit small-B harness; no native SCL",
              "pair_boundary": "oracle outside and fixed two patterns; O(n V^2) time/O(V^2+n V) workspace; no pattern search"}
    common.write_json(out / "summary.json", result)
    return result
