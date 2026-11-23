"""Measure a sufficient empty-list stability criterion on archived DEC-35 queries."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "jobs/037_polar_tractable")]
from baselines import polar
from framework.polar_count_cavity import action_hash, cavity_query
from framework.polar_full_pattern import CountBank
from rollback_frames import geometry, make_frame


class ObserveCuts:
    """Delegate NumPy operations and observe the unmodified production prune inputs."""

    def __init__(self, code):
        self.limit = code.list_size
        self.remaining = iter(code.n - 1 - code.information)
        self.gaps = []

    def __getattr__(self, name):
        return getattr(np, name)

    def argsort(self, values, **kwargs):
        order = np.argsort(values, **kwargs)
        suffix = next(self.remaining, None)
        if len(order) > self.limit:
            assert suffix is not None
            self.gaps.append((int(suffix), float(values[order[self.limit]] - values[order[self.limit - 1]])))
        return order


def decode_cuts(code, channel):
    witness = ObserveCuts(code)
    with patch.object(polar, "np", witness):
        payloads, _ = code.decode_list(channel)
    if not witness.gaps:
        raise ValueError("Study requires a code and width with actual list truncation")
    return payloads, np.asarray(witness.gaps).T


def finite():
    rng = np.random.default_rng(39061)
    raw = np.array(list(itertools.product((0, 1), repeat=8)), dtype=np.uint8)
    signs = 1.0 - 2.0 * polar.polar_transform(raw)
    code = polar.PolarCode(3, 8, 4, 2)
    checks = Counter()
    for scale in (1e-7, .002, .04, .5):
        for _ in range(36):
            base = rng.normal(size=8)
            shift = rng.normal(size=8) * scale
            weights = .5 * signs @ base
            changed = .5 * signs @ (base + shift)
            temperature = rng.uniform(.7, 1.3)
            for depth in range(1, 9):
                group_size = 2**(8-depth)
                delta = logsumexp(changed.reshape(-1, group_size), axis=1) - logsumexp(weights.reshape(-1, group_size), axis=1)
                assert np.ptp(delta) <= np.abs(shift).sum() + 1e-13
                checks["prefix_span"] += 1
                z = logsumexp(weights.reshape(-1, group_size), axis=1)
                z_new = logsumexp(changed.reshape(-1, group_size), axis=1)
                remaining = np.abs(base + shift - temperature * base).sum()
                assert np.ptp(z_new - temperature*z) <= abs(temperature-1)*np.log(group_size) + remaining + 1e-13
                checks["temperature_span"] += 1
            old, (suffixes, cuts) = decode_cuts(code, base)
            uniform = cuts - np.abs(shift).sum()
            scaled = temperature * cuts - abs(temperature-1)*suffixes*np.log(2) - remaining
            if np.minimum.reduce(np.maximum(uniform, scaled)) > 1e-11:
                new, _ = code.decode_list(base + shift)
                assert set(map(tuple, old)) == set(map(tuple, new))
                checks["certified"] += 1
                checks["certified_empty"] += int(not len(old))
    assert checks["certified_empty"] > 0
    return checks


def measure(seed):
    path = ROOT / f"results/polar_dec35_exploration_2026-11-11/count_growth/main/frame_{seed}.json"
    data = json.loads(path.read_bytes())
    arm = data["arms"]["count_growth"]
    log = arm["meta"]
    result = Counter(actual_calls=log["actual_scl_calls"], cache_hits=log["cache_hits"], queries=len(log["queries"]))
    model, _ = geometry(seed)
    scenario, labels = make_frame(model, seed, 0)
    assert labels == data["labels"]
    bank = CountBank(scenario, arm["bank"])
    moves = {m["query"]: m for m in log["count_action_path"]}
    action = Counter(dict(log["initial_refit_action"]))
    raw = scenario.y - bank.matrix @ bank.vector(action)
    previous, gaps, records = {}, {}, []
    for q in log["queries"]:
        assert q["pre_action_sha256"] == action_hash(action)
        pattern = q["pattern"]
        channel, _ = cavity_query(scenario, action, raw, pattern)
        key = hashlib.sha256(channel.tobytes()).hexdigest()
        assert q["llr_sha256"] == key
        if not q["cache_hit"] and pattern in previous:
            old, old_key, empty, old_variance = previous[pattern]
            if empty:
                if old_key not in gaps:
                    ids, gaps[old_key] = decode_cuts(model.code, old)
                    assert len(ids) == 0
                delta = float(np.abs(channel-old).sum())
                suffixes, cuts = gaps[old_key]
                scale = old_variance / q["variance"]
                sparse_change = float(np.abs(channel-scale*old).sum())
                scaled = scale*cuts - abs(scale-1)*suffixes*np.log(2) - sparse_change
                gap = float(cuts.min())
                certified = bool(np.maximum(cuts-delta, scaled).min() > 1e-10)
                if certified:
                    assert not q["ids"] and not len(model.code.decode_list(channel)[0])
                result["eligible_old_empty"] += 1
                result["uniform_certified"] += int(gap > delta + 1e-10)
                result["variance_changed"] += int(scale != 1)
                result["certified"] += int(certified)
                records.append([q["query"], pattern, gap, delta, scale, sparse_change, certified])
        previous[pattern] = channel, key, not q["ids"], q["variance"]
        if q["query"] in moves:
            move = moves[q["query"]]
            assert Counter(dict(move["before"])) == action
            action = Counter(dict(move["after"]))
            raw = scenario.y - bank.matrix @ bank.vector(action)
    return dict(seed=seed, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), counts=result, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checked = finite()
    rows = [measure(seed) for seed in range(7101, 7117)]
    total = sum((Counter(row["counts"]) for row in rows), Counter())
    frames = sum(row["counts"]["certified"] > 0 for row in rows)
    passed = total["certified"] > .05 * total["actual_calls"] and frames >= 3
    args.out.write_text(json.dumps(dict(finite=checked, totals=total, certified_frames=frames, promote=passed, frames=rows), indent=2))
    print(dict(finite=checked, totals=total, certified_frames=frames, promote=passed))
