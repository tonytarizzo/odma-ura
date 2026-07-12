"""Adapter for the authors' pinned CCS-AMP research code.

The upstream repository has no declared software licence, so it is not copied
into this repository.  Experiment drivers import a user-supplied checkout and
verify its commit before using the original factor-graph and AMP classes.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


AUTHOR_REPO = "https://github.com/vamsi128/CCS-AMP-Code.git"
AUTHOR_COMMIT = "92080d85408d5d19a123d1d61ba76ec6f15451a5"


def default_author_dir() -> Path:
    return Path(os.environ.get("CCS_AMP_AUTHOR_DIR", ".cache/CCS-AMP-Code"))


def verify_author_checkout(path: Path, *, allow_unpinned: bool = False) -> str:
    required = [path / "PyFHT_local.py", path / "ccsfg.py", path / "FactorGraphGeneration.py", path / "ccsinnercode.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"CCS-AMP author checkout is missing {missing}. Run: git clone {AUTHOR_REPO} {path} && "
            f"git -C {path} checkout {AUTHOR_COMMIT}")
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    if not allow_unpinned and commit != AUTHOR_COMMIT:
        raise RuntimeError(f"expected author commit {AUTHOR_COMMIT}, found {commit}; pass --allow-unpinned-author-code to override")
    return commit


def load_author_modules(path: Path, *, transform_seed: int, allow_unpinned: bool = False) -> SimpleNamespace:
    """Load the original modules while repairing their missing local NumPy import.

    Upstream ``ccsinnercode.py`` imports ``pyfht``, but the repository only ships
    ``PyFHT_local.py`` and that file omits ``import numpy as np``.  The shim keeps
    the upstream algorithms unchanged, supplies NumPy, and maps its ``seed=None``
    call to an explicit seed so direct/framework initialisations are reproducible.
    """
    commit = verify_author_checkout(path, allow_unpinned=allow_unpinned)
    fht_spec = importlib.util.spec_from_file_location("ccs_amp_author_pyfht", path / "PyFHT_local.py")
    if fht_spec is None or fht_spec.loader is None:
        raise ImportError(f"cannot load {path / 'PyFHT_local.py'}")
    fht = importlib.util.module_from_spec(fht_spec)
    fht_spec.loader.exec_module(fht)
    fht.np = np
    upstream_block_sub_fht = fht.block_sub_fht

    def deterministic_block_sub_fht(n, m, l, seed=0, ordering=None, new_embedding=False):
        fixed_seed = int(transform_seed) if seed is None else seed
        return upstream_block_sub_fht(n, m, l, seed=fixed_seed, ordering=ordering, new_embedding=new_embedding)

    fht.block_sub_fht = deterministic_block_sub_fht
    sys.modules["pyfht"] = fht
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    for name in ("ccsfg", "FactorGraphGeneration", "ccsinnercode"):
        sys.modules.pop(name, None)
    with contextlib.redirect_stdout(io.StringIO()):
        fg = importlib.import_module("FactorGraphGeneration")
        inner = importlib.import_module("ccsinnercode")
    return SimpleNamespace(FG=fg, inner=inner, fht=fht, commit=commit)


def graph_for_preset(modules: SimpleNamespace, preset: str):
    with contextlib.redirect_stdout(io.StringIO()):
        if preset == "paper_b128":
            return modules.FG.Triadic8(16)
        if preset == "adapted_b100":
            return modules.FG.Triadic10(10)
        if preset == "explicit_b8":
            return modules.FG.Triadic4(2)
    raise ValueError(f"unknown CCS-AMP preset {preset!r}")


def preset_parameters(preset: str) -> dict:
    if preset == "paper_b128":
        return {"payload_bits": 128, "n": 38400, "sections": 16, "section_bits": 16, "paper_comparable": True}
    if preset == "adapted_b100":
        return {"payload_bits": 100, "n": 38400, "sections": 20, "section_bits": 10, "paper_comparable": False}
    if preset == "explicit_b8":
        return {"payload_bits": 8, "n": 64, "sections": 8, "section_bits": 2, "paper_comparable": False}
    raise ValueError(f"unknown CCS-AMP preset {preset!r}")


def codeword_key(codeword: np.ndarray) -> bytes:
    return np.packbits(np.asarray(codeword, dtype=np.uint8)).tobytes()


def number_matches(true_codewords: np.ndarray, decoded_codewords: list[np.ndarray], max_count: int) -> int:
    true_multiset: dict[bytes, int] = {}
    for row in true_codewords:
        key = codeword_key(row)
        true_multiset[key] = true_multiset.get(key, 0) + 1
    matches = 0
    for row in decoded_codewords[:max_count]:
        key = codeword_key(row)
        if true_multiset.get(key, 0) > 0:
            matches += 1
            true_multiset[key] -= 1
    return matches


def run_author_trial(modules: SimpleNamespace, *, preset: str, K: int, ebn0_db: float, seed: int,
                     amp_iterations: int, bp_iterations: int, enhanced: bool, list_extra: int = 10) -> dict:
    params = preset_parameters(preset)
    B, n = params["payload_bits"], params["n"]
    np.random.seed(int(seed))
    graph = graph_for_preset(modules, preset)
    power = 2.0 * B * 10.0 ** (float(ebn0_db) / 10.0) / n
    inner = modules.inner.DenseInnerCode(n, power, 1.0, int(K), graph)
    bits = np.random.randint(0, 2, size=(int(K), B))
    true_codewords = graph.encodemessages(bits)
    sparse_sum = np.sum(true_codewords, axis=0)
    clean = inner.Encode(sparse_sum)
    noise = np.random.randn(n, 1)
    observation = clean + noise
    estimates, tau = inner.Decode(observation, int(amp_iterations), enhanced, int(bp_iterations), graph)
    with contextlib.redirect_stdout(io.StringIO()):
        decoded = graph.decoder(estimates.copy(), int(K) + int(list_extra))
    matches = number_matches(true_codewords, decoded, int(K))
    return {"pupe": (int(K) - matches) / float(K), "matches": matches, "decoded": len(decoded),
            "tau": np.asarray(tau, dtype=float).reshape(-1).tolist()}

