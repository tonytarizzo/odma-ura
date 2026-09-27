"""Matched real-GMAC comparisons with independent native and learned receivers.

No training or evaluation step is given a true support. D3/D4 propose candidates
by searching the complete small-B alphabet; missed proposals remain PUPE errors.
Native B>=60 uses bit lists only. All dB values use physical Eb/N0, not the older
framework's real-noise convention (whose numerical dB values are 3.0103 higher).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from baselines import make_baseline, message_bits, message_indices
from framework.candidate_decoders import CandidateRestrictedEffectiveChannelPGD, CandidateSet, LearnedCandidateGeometryPGD
from framework.core import ComponentSpec, URASpec
from framework.early_stopping import EarlyStopping
from framework.encoder import ComponentConstraints, Encoder, build_encoder
from framework.learned_decoders import FactorAttentionISTANet, UnrolledBernoulliPGD, UnrolledEffectiveChannelPGD
from framework.losses import effective_channel_loss, support_count_loss
from .ura_bounds import collision_rates


DEFAULTS = {"max_epochs": 200, "patience": 10, "batches_per_epoch": 100, "batch_size": 8,
            "validation_batches": 16, "layers": 8, "hidden_dim": 32, "power_iters": 12,
            "candidate_size": 1024, "lr": 0.002, "encoder_lr": 0.001, "train_ebn0": [-4.0, 10.0],
            "loads": [7, 15, 22, 26], "eval_ebn0": [-4, -2, 0, 2, 4, 6, 8, 10], "eval_frames": 256,
            "train_sampling": "distinct", "eval_sampling": ["distinct", "iid"], "threads": 4}
PAPERS = ("odma_polar", "dynamic_cs", "ccs_amp", "ccs_block")


class ExplicitEncoder(Encoder):
    """Identity-factor specialization; retain existing decoder/constraint interfaces."""

    def explicit_matrix(self): return self.components[0].C
    def matvec(self, counts): return counts @ self.explicit_matrix().T
    def rmatvec(self, residual): return residual @ self.explicit_matrix()
    def message_columns(self, indices): return self.explicit_matrix()[:, indices]

    def spectral_norm_squared(self, num_iters=20, generator=None, use_cache=True):
        generator = generator or torch.Generator(device=self.device).manual_seed(730019)
        return super().spectral_norm_squared(num_iters, generator, use_cache)


def materialize(baseline, chunk_size=256):
    if baseline.payload_bits > 20:
        raise ValueError("Materialization is only allowed for B<=20")
    m = 1 << baseline.payload_bits
    matrix = np.empty((baseline.n, m), dtype=np.float32)
    for start in range(0, m, chunk_size):
        stop = min(m, start + chunk_size)
        matrix[:, start:stop] = baseline.encode(message_bits(np.arange(start, stop), baseline.payload_bits)).T
    return matrix


def make_encoder(config, baseline=None):
    b, n, seed = config["B"], config["n"], config["seed"]
    if not 1 <= b <= 20:
        raise ValueError("Explicit learned receivers are restricted to 1<=B<=20")
    m, learn = 1 << b, config["mode"] == "joint"
    if learn and config["family"] in PAPERS:
        raise ValueError("Do not train away a published encoder's construction")
    support = None
    if baseline is not None:
        matrix = materialize(baseline)
    else:
        rng = np.random.default_rng(seed)
        matrix = rng.standard_normal((n, m)).astype(np.float32)
        if config["family"] == "sparse":
            size = int(config.get("support", max(1, n // 8)))
            if not 1 <= size <= n:
                raise ValueError("Support must lie in [1,n]")
            support = np.zeros((n, m), dtype=bool)
            for column in range(m):
                support[rng.choice(n, size, replace=False), column] = True
            matrix *= support
        elif config["family"] != "dense":
            raise ValueError("Expected dense, sparse or a published encoder")
        matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    spec = URASpec(n, m, max(config["loads"]), payload_bits=b)
    component = ComponentSpec(1, n, m, R_init="identity", C_init="explicit", T_init="identity", learn_C=learn,
                              explicit_C=torch.from_numpy(matrix),
                              fixed_C_support=None if support is None else torch.from_numpy(support))
    base = build_encoder(spec, [component], [ComponentConstraints(C="unit_norm_columns" if learn else "none")])
    return ExplicitEncoder(list(base.components), spec)


def make_decoder(config):
    layers, name = config["layers"], config["decoder"]
    if name == "d0": return UnrolledBernoulliPGD(layers, power_iters=config["power_iters"])
    if name == "d1": return FactorAttentionISTANet(layers, hidden_dim=config["hidden_dim"], power_iters=config["power_iters"])
    if name == "d2": return UnrolledEffectiveChannelPGD(layers)
    if name == "d3": return CandidateRestrictedEffectiveChannelPGD(layers)
    if name == "d4": return LearnedCandidateGeometryPGD(layers, hidden_dim=config["hidden_dim"])
    raise ValueError(f"Unknown decoder {name!r}")


def sample_messages(rng, batch_size, k, b, sampling):
    if sampling not in {"iid", "distinct"}:
        raise ValueError("sampling must be iid or distinct")
    if b <= 20:
        ids = (rng.integers(0, 1 << b, (batch_size, k)) if sampling == "iid" else
               np.stack([rng.choice(1 << b, k, replace=False) for _ in range(batch_size)]))
        return message_bits(ids, b)
    bits = rng.integers(0, 2, (batch_size, k, b), dtype=np.uint8)
    if sampling == "distinct":
        for row in bits:
            while len(np.unique(row, axis=0)) != k:
                row[:] = rng.integers(0, 2, row.shape, dtype=np.uint8)
    return bits


def noise_variance(b, ebn0): return 1.0 / (2 * b * 10.0 ** (float(ebn0) / 10))


def learned_batch(encoder, bits, standard_noise, ebn0):
    indices = torch.from_numpy(message_indices(bits)).long()
    counts = torch.zeros(len(bits), encoder.num_codewords)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=counts.dtype))
    noise_var = noise_variance(encoder.spec.payload_bits, ebn0)
    y = encoder.matvec(counts) + np.sqrt(noise_var) * torch.as_tensor(standard_noise, dtype=encoder.dtype)
    return counts, y, noise_var


def forward(config, decoder, encoder, y, k, noise_var):
    h = torch.ones(len(y), 1, dtype=y.dtype)
    if config["decoder"] not in {"d3", "d4"}:
        return decoder(encoder, y[..., None], h, k, noise_var=noise_var), None
    size = min(config["candidate_size"], encoder.num_codewords)
    if size <= k:
        raise ValueError("D3/D4 require candidate_size>K")
    with torch.no_grad():
        energies = encoder.explicit_matrix().square().sum(0).clamp_min(1e-12)
        score = encoder.rmatvec(y) / energies.sqrt()
        indices = score.topk(size, dim=1).indices
    candidates = CandidateSet(message_indices=indices, source="full_alphabet_matched_filter",
                              metadata={"oracle": False, "searched_messages": encoder.num_codewords})
    output = decoder(encoder, y[..., None], h, k, noise_var=noise_var, candidates=candidates)
    return output, indices


def objective(config, output, counts, indices):
    if indices is not None:
        # A rejected message has zero predicted count and clipped negative evidence.
        # Include it in supervision/validation; hard shortlist membership has no gradient.
        evidence = [torch.full_like(counts, -30.0).scatter(1, indices, value)
                    for value in output.meta["layer_evidence_logits"]]
        soft = torch.zeros_like(counts).scatter(1, indices, output.meta["soft_counts"])
        output = SimpleNamespace(meta={**output.meta, "layer_evidence_logits": evidence, "soft_counts": soft})
    loss_fn = support_count_loss if config["decoder"] in {"d0", "d1"} else effective_channel_loss
    return loss_fn(output, counts)[0]


def decoded_lists(output, indices, k, b):
    hard = output.counts.detach().cpu().numpy()
    result = []
    for row, values in enumerate(hard):
        selected = np.flatnonzero(values > 0)
        selected = selected[np.argsort(-values[selected], kind="stable")[:k]]
        if indices is not None:
            selected = indices[row, selected].cpu().numpy()
        result.append(message_bits(selected, b))
    return result


def frame_metrics(truth, decoded, candidates=None):
    true_keys = [x.tobytes() for x in truth]
    found = {x.tobytes() for x in decoded}
    if len(decoded) > len(truth) or len(found) != len(decoded):
        raise ValueError("Receiver must return a unique list of at most K messages")
    multiplicity = Counter(true_keys)
    missed = [key not in found for key in true_keys]
    result = {"pupe": float(np.mean(missed)), "false_alarm": len(found - set(true_keys)) / max(1, len(found)),
              "strict_pupe": float(np.mean([miss or multiplicity[key] > 1 for key, miss in zip(true_keys, missed)])),
              "duplicate_users": float(np.mean([multiplicity[key] > 1 for key in true_keys])),
              "any_duplicate": float(len(multiplicity) != len(truth)), "list_size": len(found)}
    if candidates is not None:
        proposed = {x.tobytes() for x in candidates}
        result["candidate_recall"] = float(np.mean([key in proposed for key in true_keys]))
    return result


def validation(config, encoder, decoder):
    rng = np.random.default_rng(config["seed"] + 200_000)
    decoder.eval()
    losses, pupes, recalls = [], [], []
    with torch.no_grad():
        for _ in range(config["validation_batches"]):
            k = int(rng.choice(config["loads"]))
            ebn0 = float(rng.uniform(*config["train_ebn0"]))
            bits = sample_messages(rng, config["batch_size"], k, config["B"], config["train_sampling"])
            counts, y, noise_var = learned_batch(encoder, bits, rng.standard_normal((len(bits), config["n"])), ebn0)
            output, indices = forward(config, decoder, encoder, y, k, noise_var)
            losses.append(float(objective(config, output, counts, indices)))
            pupes.extend(frame_metrics(t, d)["pupe"] for t, d in zip(bits, decoded_lists(output, indices, k, config["B"])))
            if indices is not None:
                recalls.extend((counts.gather(1, indices).sum(1) / counts.sum(1)).tolist())
    return {"loss": float(np.mean(losses)), "pupe": float(np.mean(pupes)),
            **({"candidate_recall": float(np.mean(recalls))} if recalls else {})}


def train(config, encoder, decoder, out_dir):
    parameters = [{"params": list(decoder.parameters()), "lr": config["lr"]}]
    if config["mode"] == "joint":
        parameters.append({"params": list(encoder.parameters()), "lr": config["encoder_lr"]})
    optimizer = torch.optim.Adam(parameters)
    stopper = EarlyStopping(config["patience"])
    modules = {"encoder": encoder, "decoder": decoder}
    initial = validation(config, encoder, decoder)
    stopper.update(initial["loss"], 0, modules)
    rng = np.random.default_rng(config["seed"] + 100_000)
    history, started = [], time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        decoder.train(); total = 0.0
        for _ in range(config["batches_per_epoch"]):
            k = int(rng.choice(config["loads"]))
            ebn0 = float(rng.uniform(*config["train_ebn0"]))
            bits = sample_messages(rng, config["batch_size"], k, config["B"], config["train_sampling"])
            counts, y, noise_var = learned_batch(encoder, bits, rng.standard_normal((len(bits), config["n"])), ebn0)
            output, indices = forward(config, decoder, encoder, y, k, noise_var)
            loss = objective(config, output, counts, indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            trainable = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
            optimizer.step()
            if config["mode"] == "joint":
                encoder.apply_constraints()
                if hasattr(decoder, "clear_geometry_cache"): decoder.clear_geometry_cache()
            total += float(loss.detach())
        val = validation(config, encoder, decoder)
        record = {"epoch": epoch, "training_loss": total / config["batches_per_epoch"], "validation": val}
        history.append(record)
        print(json.dumps(record), flush=True)
        with (out_dir / "training.jsonl").open("a") as handle: handle.write(json.dumps(record) + "\n")
        if stopper.update(val["loss"], epoch, modules): break
    restored = stopper.restore(modules)
    encoder._spectral_cache.clear(); encoder._mean_energy_cache = None
    if hasattr(decoder, "clear_geometry_cache"): decoder.clear_geometry_cache()
    final = validation(config, encoder, decoder)
    torch.save({"config": config, **{key: module.state_dict() for key, module in modules.items()}}, out_dir / "checkpoint.pt")
    return {"initial": initial, "final": final, "history": history, "seconds": time.perf_counter() - started,
            "early_stopping": stopper.summary(len(history), restored)}


def evaluate(config, baseline, encoder, decoder, out_dir):
    cells = []
    if decoder is not None: decoder.eval()
    for sampling in config["eval_sampling"]:
        for k in config["loads"]:
            for ebn0 in config["eval_ebn0"]:
                # Identical messages and standard noise across codebooks, decoders and SNRs.
                rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 300_000, k, sampling == "iid"]))
                frames, native_meta = [], []
                elapsed = 0.0
                for start in range(0, config["eval_frames"], config["batch_size"]):
                    size = min(config["batch_size"], config["eval_frames"] - start)
                    bits = sample_messages(rng, size, k, config["B"], sampling)
                    noise = rng.standard_normal((size, config["n"]))
                    variance = noise_variance(config["B"], ebn0)
                    if decoder is None:
                        for truth, z in zip(bits, noise):
                            y = baseline.encode(truth).sum(0) + np.sqrt(variance) * z
                            before = time.perf_counter()
                            decoded, meta = baseline.decode(y, k, variance)
                            elapsed += time.perf_counter() - before
                            frames.append(frame_metrics(truth, decoded)); native_meta.append(meta)
                    else:
                        with torch.no_grad():
                            _, y, _ = learned_batch(encoder, bits, noise, ebn0)
                            before = time.perf_counter()
                            output, indices = forward(config, decoder, encoder, y, k, variance)
                            decoded = decoded_lists(output, indices, k, config["B"])
                            elapsed += time.perf_counter() - before
                        for i, (truth, found) in enumerate(zip(bits, decoded)):
                            candidates = None if indices is None else message_bits(indices[i].cpu().numpy(), config["B"])
                            frames.append(frame_metrics(truth, found, candidates))
                values = np.array([row["pupe"] for row in frames])
                cell = {"sampling": sampling, "K": k, "ebn0_db": ebn0, "frames": len(frames),
                        "means": {key: float(np.mean([row[key] for row in frames])) for key in frames[0]},
                        "pupe_standard_error_across_frames": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values)>1 else None,
                        "decoder_seconds_per_frame": elapsed / len(frames), "frame_metrics": frames,
                        "analytic_collisions": collision_rates(config["B"], k) if sampling == "iid" else None}
                # Full native diagnostics remain per cell, including search caps and failed CRCs.
                if native_meta: cell["native_diagnostics"] = native_meta
                cells.append(cell)
                with (out_dir / "evaluation.jsonl").open("a") as handle: handle.write(json.dumps(cell, allow_nan=False) + "\n")
                print(f"{config['family']}/{config['decoder']} {sampling} K={k} Eb/N0={ebn0}: PUPE={values.mean():.4f}", flush=True)
    return cells


def run_experiment(config, out_dir):
    config = {**DEFAULTS, **config}
    if config["mode"] not in {"fixed", "joint", "native"}:
        raise ValueError("mode must be fixed, joint or native")
    if (config["decoder"] == "native") != (config["mode"] == "native"):
        raise ValueError("Native receiver and native mode must be selected together")
    if max(config["loads"]) >= 2 ** config["B"] or min(config["loads"]) < 1:
        raise ValueError("All loads must satisfy 1<=K<2^B")
    if min(config[key] for key in ("batch_size", "eval_frames", "validation_batches", "batches_per_epoch")) < 1:
        raise ValueError("Batch and evaluation budgets must be positive")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_files = sorted(Path("baselines").glob("*.py")) + sorted(Path("benchmarks").glob("*.py"))
    source_files += sorted(Path("framework").glob("*.py")) + [Path("src/ura_bound.py"), Path("tests/ccs_amp_author.py")]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite results in {out_dir}; choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"] + 400_000)
    baseline = make_baseline(config["family"], config["B"], config["n"], config["seed"],
                             **config.get("baseline_params", {})) if config["family"] in PAPERS else None
    encoder = None if config["decoder"] == "native" else make_encoder(config, baseline)
    if encoder is not None:
        matrix = encoder.explicit_matrix().detach().numpy()
        initial_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
        energies = np.square(matrix.astype(float)).sum(0)
        energy_scope = "exhaustive"
    else:
        if baseline is None: raise ValueError("Native mode requires a published receiver")
        rng = np.random.default_rng(config["seed"] + 500_000)
        audit_bits = sample_messages(rng, 1, 256, config["B"], "iid")[0]
        energies = np.square(baseline.encode(audit_bits)).sum(1)
        energy_scope, initial_hash = "256 sampled messages; not a worst-case certificate", None
    energy = {"scope": energy_scope, "min": float(energies.min()), "mean": float(energies.mean()),
              "max": float(energies.max()), "std": float(energies.std())}
    decoder = None if encoder is None else make_decoder(config)
    training = None if decoder is None else train(config, encoder, decoder, out_dir)
    cells = evaluate(config, baseline, encoder, decoder, out_dir)
    if source_hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}:
        raise RuntimeError("Source files changed during this run; refuse a misleading completed-result fingerprint")
    summary = {"status": "complete", "config": config, "revision": revision, "source_sha256": source_hashes,
               "baseline": None if baseline is None else baseline.metadata(), "energy_audit_initial": energy,
               "initial_matrix_sha256": initial_hash, "training": training, "evaluation": cells,
               "decoder_parameters": 0 if decoder is None else sum(p.numel() for p in decoder.parameters()),
               "encoder_learned_parameters": 0 if encoder is None else sum(p.numel() for p in encoder.parameters()),
               "physical_noise_convention": "real AWGN sigma^2=1/(2*B*10^(EbN0_dB/10)); nominal E=1",
               "known_to_receiver": ["K", "noise variance", "codebook/construction", "unit real channel"],
               "candidate_policy": "non-oracle full-alphabet matched filter" if config["decoder"] in {"d3", "d4"} else None,
               "candidate_training": ("Hard top-C; no membership gradient. Full-alphabet loss assigns rejected messages "
                                      "zero counts and evidence -30. No differentiable proposer objective."
                                      if config["decoder"] in {"d3", "d4"} else None)}
    if encoder is not None:
        final_energy = encoder.explicit_matrix().detach().square().sum(0)
        summary["energy_audit_final"] = {"min": float(final_energy.min()), "max": float(final_energy.max())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(json.loads(args.config.read_text()), args.out_dir)


if __name__ == "__main__": main()
