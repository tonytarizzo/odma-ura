"""Full-support polar proposals with exact likelihood correction; finite research audit, not a mixing guarantee."""

import argparse
import heapq
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode
from benchmarks.polar_energy_geometry import balanced_energy
from tests.polar_approximation_audit import crc_constrained_list, decision_risk, encode_words, polar_beam, posterior_tails
from tests.polar_approximation_audit import sample_tails
from tests.polar_exact_computation import tail_action
from tests.polar_posterior_test import exact_posterior


def uniform_allowed(rng, alphabet, excluded):
    """Map a uniform allowed rank to a word without scanning the alphabet or rejection sampling."""
    word = int(rng.integers(alphabet - len(excluded)))
    for blocked in sorted(excluded):
        if word < blocked:
            break
        word += 1
    return word


class BlockProposal:
    """One fixed-outside block proposal; cache contexts only for this attempted move."""

    def __init__(self, base, energy, code, residual, fixed, sigma2, sampling, epsilon, block_size, temper=False):
        if not 0 < epsilon < 1 or sigma2 <= 0 or block_size not in (1, 2):
            raise ValueError("Require positive noise, 0 < epsilon < 1, and block size one or two")
        self.base, self.energy, self.code, self.residual = base, energy, code, residual
        self.fixed = set(map(int, fixed)) if sampling == "distinct" else set()
        self.sigma2, self.sampling, self.epsilon, self.block_size, self.temper = sigma2, sampling, epsilon, block_size, temper
        self.alphabet = base.num_patterns * (1 << code.payload_bits)
        self.contexts, self.signals = {}, {}
        self.encoded_columns = self.uniform_draws = self.empty_contexts = 0

    def signal(self, word):
        if word not in self.signals:
            self.signals[word] = encode_words(self.base, self.energy, [word])[:, 0]
            self.encoded_columns += 1
        return self.signals[word]

    def context(self, prefix):
        prefix = tuple(map(int, prefix))
        if prefix not in self.contexts:
            excluded = self.fixed | set(prefix) if self.sampling == "distinct" else set()
            residual = self.residual.copy()
            for word in prefix:
                residual -= self.signal(word)
            remaining = self.block_size - len(prefix) - 1
            variance = self.sigma2 + (remaining / self.base.n if self.temper else 0.)
            words = []
            for pattern, positions in enumerate(self.base.positions):
                llr = 2 * np.sqrt(self.energy[pattern]) * residual[positions] / variance
                payloads, _ = crc_constrained_list(self.code, llr)
                for payload in payloads:
                    word = pattern * (1 << self.code.payload_bits) + int(
                        payload @ (1 << np.arange(self.code.payload_bits - 1, -1, -1)))
                    if word not in excluded:
                        words.append(word)
            if words:
                columns = encode_words(self.base, self.energy, words)
                scores = (columns.T @ residual - np.square(columns).sum(0) / 2) / variance
                log_weights = scores - logsumexp(scores)
                self.encoded_columns += len(words)
                self.signals.update(zip(words, columns.T))
            else:
                log_weights = np.empty(0)
                self.empty_contexts += 1
            self.contexts[prefix] = (words, dict(zip(words, log_weights)), excluded)
        return self.contexts[prefix]

    def log_next(self, prefix, word):
        words, log_weights, excluded = self.context(prefix)
        if word in excluded:
            return -math.inf
        log_uniform = -math.log(self.alphabet - len(excluded))
        if not words:
            return log_uniform
        return float(np.logaddexp(math.log(self.epsilon) + log_uniform,
                                 math.log1p(-self.epsilon) + log_weights.get(word, -math.inf)))

    def log_probability(self, words):
        return sum(self.log_next(words[:j], int(word)) for j, word in enumerate(words))

    def draw(self, rng):
        result = []
        for _ in range(self.block_size):
            words, log_weights, excluded = self.context(result)
            if not words or rng.random() < self.epsilon:
                word = uniform_allowed(rng, self.alphabet, excluded)
                self.uniform_draws += 1
            else:
                word = int(rng.choice(words, p=np.exp(list(log_weights.values()))))
            result.append(word)
        return tuple(result)

    def log_acceptance(self, old, new):
        old_residual, new_residual = self.residual.copy(), self.residual.copy()
        for word in old:
            old_residual -= self.signal(int(word))
        for word in new:
            new_residual -= self.signal(int(word))
        likelihood = (old_residual @ old_residual - new_residual @ new_residual) / (2 * self.sigma2)
        return min(0., float(likelihood + self.log_probability(old) - self.log_probability(new)))


def sparse_action(tails, k):
    heap = [(-values[0], word, 0) for word, values in tails.items()]
    heapq.heapify(heap)
    action = {}
    for _ in range(k):
        if not heap:
            raise ValueError("Insufficient empirical output capacity")
        _, word, level = heapq.heappop(heap)
        action[word] = level + 1
        if level + 1 < len(tails[word]):
            heapq.heappush(heap, (-tails[word][level + 1], word, level + 1))
    return action


def restricted_update(slots, signals, signal_sum, candidates, columns, index, y, sigma2, sampling, rng):
    """Refresh inside a fixed candidate class; an outside current word stays put, preserving the full target."""
    slot = int(rng.integers(len(slots)))
    if int(slots[slot]) not in index:
        return False
    allowed = np.ones(len(candidates), dtype=bool)
    if sampling == "distinct":
        allowed &= ~np.isin(candidates, np.delete(slots, slot))
    options = np.flatnonzero(allowed)
    residual = y - signal_sum + signals[:, slot]
    logits = (columns[:, options].T @ residual - np.square(columns[:, options]).sum(0) / 2) / sigma2
    selected = int(rng.choice(options, p=np.exp(logits - logsumexp(logits))))
    changed = slots[slot] != candidates[selected]
    signal_sum += columns[:, selected] - signals[:, slot]
    signals[:, slot] = columns[:, selected]
    slots[slot] = candidates[selected]
    return bool(changed)


def revisable_tails(base, energy, y, k, sigma2, sampling, initial, list_size, mode, budget, seed, chains=4, epsilon=.1):
    """Reserved context budget gives a fixed step count; exact target, correlated finite-run estimates."""
    if mode not in ("single", "mixed", "pair", "mixed_tempered", "hybrid") or k < 2:
        raise ValueError("This audit supports K>=2 and the documented fixed block schedules")
    code = PolarCode(base.polar.payload_bits, base.polar.n, base.polar.crc_bits, list_size)
    totals, unique_counts, all_scored = {}, set(), set()
    samples = moves = accepted = changes = contexts = encoded = uniform = empty = charged_total = 0
    outside_copies = 0
    local_steps = local_changes = 0
    initial_set, restart_tails = set(map(int, initial)), []
    started = time.perf_counter()
    initial_columns = encode_words(base, energy, initial) if mode == "hybrid" else None
    initial_index = {int(word): i for i, word in enumerate(initial)}
    for chain in range(chains):
        rng = np.random.default_rng(seed + chain)
        slots = rng.choice(initial, k, replace=sampling == "iid")
        signals = encode_words(base, energy, slots)
        signal_sum = signals.sum(1)
        charged = step = saved = 0
        local = {}
        while True:
            block_size = 2 if mode == "pair" or (mode.startswith("mixed") and step % 8 == 7) else 1
            charge = 2 * block_size - 1
            if charged + charge > budget:
                break
            if mode == "hybrid":
                for _ in range(7):
                    local_changes += restricted_update(slots, signals, signal_sum, initial, initial_columns,
                                                      initial_index, y, sigma2, sampling, rng)
                    local_steps += 1
            block = np.sort(rng.choice(k, block_size, replace=False))
            old = tuple(map(int, slots[block]))
            outside = np.delete(slots, block)
            residual = y - signal_sum + signals[:, block].sum(1)
            proposal = BlockProposal(base, energy, code, residual, outside, sigma2, sampling, epsilon,
                                     block_size, mode == "mixed_tempered")
            new = proposal.draw(rng)
            if rng.random() < math.exp(proposal.log_acceptance(old, new)):
                replacement = np.column_stack([proposal.signal(word) for word in new])
                signal_sum += replacement.sum(1) - signals[:, block].sum(1)
                signals[:, block] = replacement
                slots[block] = new
                accepted += 1
                changes += sorted(old) != sorted(new)
            contexts += len(proposal.contexts)
            encoded += proposal.encoded_columns
            uniform += proposal.uniform_draws
            empty += proposal.empty_contexts
            all_scored.update(proposal.signals)
            charged += charge
            step += 1
            moves += 1
            if charged > budget // 4:
                words, counts = np.unique(slots, return_counts=True)
                for word, count in zip(words, counts):
                    for target in (totals, local):
                        values = target.setdefault(int(word), [])
                        values.extend([0] * max(0, int(count) - len(values)))
                        for j in range(count):
                            values[j] += 1
                    outside_copies += int(count) * (int(word) not in initial_set)
                unique_counts.add(tuple(sorted(map(int, slots))))
                samples += 1
                saved += 1
        if saved == 0:
            raise ValueError("Budget leaves no retained configurations")
        charged_total += charged
        restart_tails.append({(w, j): value / saved for w, values in local.items() for j, value in enumerate(values)})
    tails = {word: [value / samples for value in values] for word, values in totals.items()}
    events = set().union(*restart_tails)
    spread = max(max(t.get(event, 0.) for t in restart_tails) - min(t.get(event, 0.) for t in restart_tails) for event in events)
    return tails, {"moves": moves, "accepted": accepted, "count_changes": changes, "samples": samples,
        "reserved_contexts": charged_total, "actual_contexts": contexts, "polar_calls": contexts * base.num_patterns,
        "encoded_columns": encoded, "uniform_component_draws": uniform, "empty_contexts": empty,
        "restricted_steps": local_steps, "restricted_count_changes": local_changes,
        "unique_scored_words": len(all_scored), "unique_sampled_words": len(tails), "unique_counts": len(unique_counts),
        "sample_copy_fraction_outside_initial": outside_copies / (samples * k), "restart_tail_spread": spread,
        "seconds": time.perf_counter() - started}


def audit(budget, modes, payload_bits, placement_seeds, observation_seeds, list_size):
    rows, k, prefix_bits = [], 3, 2
    for bc, placement in itertools.product(payload_bits, placement_seeds):
        n, nc = (12, 8) if bc == 2 else (24, 16)
        base = ODMAPolarBaseline(bc + prefix_bits, n, placement, prefix_bits=prefix_bits, code_length=nc, crc_bits=4)
        energy, _ = balanced_energy(base.positions, n)
        alphabet = base.num_patterns * (1 << bc)
        phi = encode_words(base, energy, np.arange(alphabet))
        for sampling, sigma2, seed in itertools.product(("iid", "distinct"), (.05, .3), observation_seeds):
            rng = np.random.default_rng(9100 + seed)
            truth = np.bincount(rng.choice(alphabet, k, replace=sampling == "iid"), minlength=alphabet)
            y = phi @ truth + sigma2 ** .5 * rng.normal(size=n)
            states, p = exact_posterior(phi, y, k, sigma2, sampling)
            exact = posterior_tails(states, p, k)
            optimum = decision_risk(states, p, tail_action(exact, k), k)
            candidates, proposal_work = polar_beam(base, energy, y, k, sigma2, sampling, 4, list_size, False, True)
            row = {"payload_bits": bc, "M": alphabet, "n": n, "Nc": nc, "placement_seed": placement,
                "sampling": sampling, "sigma2": sigma2, "seed": seed, "list_size": list_size,
                "bayes_risk": optimum, "initial_candidates": candidates.tolist(), "proposal_work": proposal_work}
            if not len(candidates) or (sampling == "distinct" and len(candidates) < k):
                row["failure"] = "no_complete_initial_configuration"
                rows.append(row)
                continue
            outside = np.setdiff1d(np.arange(alphabet), candidates)
            keep = states[:, outside].sum(1) == 0
            conditional = p[keep] / p[keep].sum()
            conditional_tails = posterior_tails(states[keep][:, candidates], conditional, k)

            def regret(local_action):
                action = np.zeros(alphabet, dtype=int)
                for word, count in local_action.items():
                    action[word] = count
                return max(0., decision_risk(states, p, action, k) - optimum)

            conditional_action = tail_action(conditional_tails, k)
            restricted_action = tail_action(exact[candidates], k)
            row.update({"eta": float(max(0., 1 - p[keep].sum())), "conditional_regret": regret(dict(zip(candidates, conditional_action))),
                "candidate_action_lower_bound": regret(dict(zip(candidates, restricted_action))),
                "truth_missing_copies": int(truth[outside].sum())})
            started = time.perf_counter()
            estimated, work = sample_tails(encode_words(base, energy, candidates), y, k, sigma2, sampling, 1,
                                          4096, 12000 + seed * 10)
            work["seconds"] = time.perf_counter() - started
            row["fixed_candidate_gibbs"] = {"regret": regret(dict(zip(candidates, tail_action(estimated, k)))), **work}
            row["revisable"] = []
            for mode in modes:
                estimated, work = revisable_tails(base, energy, y, k, sigma2, sampling, candidates, list_size,
                                                  mode, budget, 12000 + seed * 10)
                dense = np.zeros((alphabet, k))
                for word, values in estimated.items():
                    dense[word, :len(values)] = values
                row["revisable"].append({"mode": mode, "regret": regret(sparse_action(estimated, k)),
                    "max_full_tail_error": float(abs(dense - exact).max()), **work})
            rows.append(row)
            print(f"Bc={bc} bank={placement} {sampling} sigma2={sigma2} seed={seed}", flush=True)
    return {"contract": {"K": k, "prefix_bits": prefix_bits, "CRC": 4, "epsilon": .1, "chains": 4,
            "reserved_contexts_per_chain": budget, "list_size": list_size, "beam_width": 4,
            "modes": modes, "status": "exact stationary target; finite exploration diagnostic; exhaustive labels are not inputs"},
            "rows": rows}


def bridge_audit():
    """Oracle-labelled conditional diagnosis of one observed failure; never used by the sampler."""
    base = ODMAPolarBaseline(6, 24, 3219, prefix_bits=2, code_length=16, crc_bits=4)
    energy, _ = balanced_energy(base.positions, 24)
    phi = encode_words(base, energy, np.arange(64))
    rng, sigma2 = np.random.default_rng(9100), .05
    truth = rng.choice(64, 3, replace=False)
    y = phi[:, truth].sum(1) + sigma2 ** .5 * rng.normal(size=24)
    states, p = exact_posterior(phi, y, 3, sigma2, "distinct")
    old, target, fixed = (27, 32), (14, 48), [28]
    residual = y - phi[:, fixed].sum(1)
    gram = phi.T @ phi / sigma2
    unary = phi.T @ residual / sigma2 - gram.diagonal() / 2
    logits = unary[:, None] + unary[None, :] - gram
    logits[fixed, :], logits[:, fixed] = -np.inf, -np.inf
    np.fill_diagonal(logits, -np.inf)
    pair = np.exp(logits - logsumexp(logits))
    rows = []
    for size in (2, 4):
        code = PolarCode(4, 16, 4, size)
        proposal = BlockProposal(base, energy, code, residual, fixed, sigma2, "distinct", .1, 2)
        root_words, _, _ = proposal.context(())
        covered = np.zeros((64, 64), dtype=bool)
        for first in root_words:
            second, _, _ = proposal.context((first,))
            covered[first, second] = True
        target_orders = (target, target[::-1])
        probability = sum(math.exp(proposal.log_probability(words)) for words in target_orders)
        transition = sum(math.exp(proposal.log_probability(words) + proposal.log_acceptance(old, words)) for words in target_orders)
        single_steps = []
        for outside, replacement in (([27, 28], 48), ([28, 48], 14)):
            r = y - phi[:, outside].sum(1)
            one = BlockProposal(base, energy, code, r, outside, sigma2, "distinct", .1, 1)
            words, _, _ = one.context(())
            single_steps.append({"fixed": outside, "replacement": replacement, "in_list": replacement in words,
                                 "proposal_probability": math.exp(one.log_probability((replacement,)))})
        rows.append({"list_size": size, "root_words": root_words, "list_only_conditional_mass": float(pair[covered].sum()),
                     "target_pair_proposal_probability": probability, "target_pair_transition_probability": transition,
                     "single_steps": single_steps})
    posterior_mass = lambda words: float(p[np.all(states == np.bincount(words, minlength=64), axis=1)].sum())
    return {"contract": "reference-only diagnosis; bank3219 distinct sigma2=.05 seed0; fixed word28 is oracle conditioning",
        "truth": truth.tolist(), "initial_conditional_mode": [27, 28, 32], "full_posterior_mode": [14, 28, 48],
        "initial_mode_probability": posterior_mass([27, 28, 32]), "full_mode_probability": posterior_mass([14, 28, 48]),
        "target_pair_conditional_probability": float(pair[14, 48] + pair[48, 14]), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--modes", nargs="+", default=["single", "mixed", "pair"])
    parser.add_argument("--payload-bits", type=int, nargs="+", default=[4])
    parser.add_argument("--placements", type=int, nargs="+", default=[3218, 3219])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--list-size", type=int, default=2)
    parser.add_argument("--diagnose-bridge", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bridge_audit() if args.diagnose_bridge else audit(
        args.budget, args.modes, args.payload_bits, args.placements, args.seeds, args.list_size)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
