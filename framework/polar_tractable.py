"""Procedural Report-11 receivers. Fixed budgets avoid a full payload alphabet, not inference error."""

from collections import Counter
from dataclasses import dataclass
import heapq
import itertools
import math

import numpy as np
from scipy.special import logsumexp

from baselines.polar import PolarCode
from framework.polar_pair_proposals import joint_pair_list, xor_pair_list


def bits(value, size):
    return np.array([(int(value) >> i) & 1 for i in range(size - 1, -1, -1)], dtype=np.uint8)


def integer(values):
    result = 0
    for bit in values:
        result = (result << 1) | int(bit)
    return result


def independent(key, pivots):
    while key:
        bit = key.bit_length() - 1
        if bit not in pivots:
            pivots[bit] = key
            return True
        key ^= pivots[bit]
    return False


@dataclass
class PolarModel:
    code: PolarCode
    positions: np.ndarray
    amplitudes: np.ndarray
    n: int

    def __post_init__(self):
        positions = np.asarray(self.positions)
        self.amplitudes = np.asarray(self.amplitudes, dtype=float).copy()
        if positions.ndim != 2 or positions.shape[1] != self.code.n or np.any(positions != positions.astype(int)):
            raise ValueError("Expected integer positions[P,polar_length]")
        self.positions = positions.astype(int, copy=True)
        self.p, self.b = len(positions), self.code.payload_bits
        if self.p < 1 or self.p & (self.p - 1) or self.n < self.code.n:
            raise ValueError("Require a power-of-two pattern bank and n>=polar length")
        if np.any(self.positions < 0) or np.any(self.positions >= self.n):
            raise ValueError("Pattern position outside the observation")
        if any(len(np.unique(row)) != self.code.n for row in self.positions):
            raise ValueError("A pattern must use distinct resources")
        if self.amplitudes.shape != positions.shape or not np.isfinite(self.amplitudes).all():
            raise ValueError("Expected finite amplitudes[P,polar_length]")
        if np.any(self.amplitudes <= 0) or not np.allclose(np.square(self.amplitudes).sum(1), 1, rtol=0, atol=1e-10):
            raise ValueError("Require positive amplitudes and unit-energy waveforms")
        self.B = self.b + (self.p.bit_length() - 1)
        generator = self.code.encode(np.eye(self.b, dtype=np.uint8)).T
        self.keys = [integer(row) for row in generator]
        self.groups = {key: np.flatnonzero(np.array(self.keys, dtype=object) == key) for key in set(self.keys)}
        self.mean = np.zeros(self.n)
        constant = np.array(self.keys, dtype=object) == 0
        np.add.at(self.mean, self.positions[:, constant].ravel(), self.amplitudes[:, constant].ravel() / self.p)

    def word(self, pattern, payload):
        return (int(pattern) << self.b) | integer(payload)

    def split(self, word):
        word = int(word)
        if not 0 <= word < 1 << self.B:
            raise ValueError("Message outside the configured alphabet")
        return word >> self.b, word & ((1 << self.b) - 1)

    def symbols(self, word):
        p, v = self.split(word)
        return p, self.amplitudes[p] * (1. - 2. * self.code.encode(bits(v, self.b)))

    def waveform(self, word):
        p, values = self.symbols(word)
        result = np.zeros(self.n)
        result[self.positions[p]] = values
        return result

    def uniform(self, rng):
        return int.from_bytes(rng.bytes((self.B + 7) // 8), "little") & ((1 << self.B) - 1)


@dataclass
class PolarScenario:
    model: PolarModel
    y: np.ndarray
    k: int
    noise_var: float
    sampling: str = "iid"

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        if self.y.shape != (self.model.n,) or not np.isfinite(self.y).all():
            raise ValueError("Require one finite physical observation")
        if int(self.k) != self.k or self.k < 1 or not np.isfinite(self.noise_var) or self.noise_var <= 0:
            raise ValueError("Require positive integer K and finite positive physical variance")
        if self.sampling != "iid":
            raise ValueError("Tractable receiver variants currently implement the IID source")


class Work:
    def __init__(self, model):
        self.model, self.stats, self.atoms = model, Counter(), {}

    def atom(self, word):
        word = int(word)
        if word not in self.atoms:
            self.atoms[word] = self.model.waveform(word)
            self.stats["waveforms_encoded"] += 1
        return self.atoms[word]

    def patterns(self, residual, limit):
        self.stats["patterns_scored"] += self.model.p
        score = np.abs(residual[self.model.positions]).sum(1)
        return np.argsort(-score, kind="stable")[:min(limit, self.model.p)]


def context(scenario, model):
    model = scenario.model if isinstance(scenario, PolarScenario) else model
    if not isinstance(model, PolarModel):
        raise ValueError("Tractable decoders require explicit procedural polar geometry via model")
    if scenario.sampling != "iid" or len(scenario.y) != model.n:
        raise ValueError("Require matching geometry and IID messages")
    if hasattr(scenario, "phi") and scenario.phi.shape[1] != 1 << model.B:
        raise ValueError("Procedural model and explicit fixture alphabet differ")
    return model, Work(model)


def positive_int(**params):
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in params.values()):
        raise ValueError(f"Require positive integer budgets: {params}")


class ParityReference:
    """Independent parity envelope and exact finite-set repair in log arithmetic."""

    def __init__(self, model, pattern, residual, variance, work, flip_order=0, max_refine=1, repair=False):
        self.model, self.pattern, self.work = model, int(pattern), work
        work.stats["basis_builds"] += 1
        h = model.amplitudes[pattern] * residual[model.positions[pattern]] / variance
        coeff = {g: float(h[index].sum()) for g, index in model.groups.items()}
        self.h0 = coeff.pop(0, 0.)
        self.coeff = coeff
        chosen, pivots = [], {}
        for key in sorted(coeff, key=lambda g: (-abs(coeff[g]), g)):
            if independent(key, pivots):
                chosen.append(key)
        self.rank = len(chosen)
        rows = chosen.copy()
        for i in range(model.b):
            if independent(1 << i, pivots):
                rows.append(1 << i)
        self.rows, self.chosen = rows, chosen
        self.inverse = self._inverse(rows)
        retained = np.array([coeff[g] for g in chosen])
        self.logzero = np.r_[-np.logaddexp(0., -2 * retained), np.full(model.b - self.rank, -math.log(2))]
        self.logone = np.r_[-np.logaddexp(0., 2 * retained), np.full(model.b - self.rank, -math.log(2))]
        self.mode = sum((int(coeff[g] < 0) << i) for i, g in enumerate(chosen))
        remainder = sum(abs(v) for g, v in coeff.items() if g not in chosen)
        self.log_u = self.h0 + (model.b - self.rank) * math.log(2) + np.logaddexp(retained, -retained).sum() + remainder
        self.explored = {}
        if repair:
            for payload in self.candidates(flip_order, max_refine):
                self.explored[payload] = self.score(payload)
        self.trie = {}
        for payload in self.explored:
            node = self.trie
            coordinate = self.coordinates(payload)
            for i in range(model.b):
                node = node.setdefault((coordinate >> i) & 1, {})
        self.log_tau = self._remaining(self.trie, 0) if self.explored else 0.
        self.explored_ids = list(self.explored)
        self.log_za = float(logsumexp(list(self.explored.values()))) if self.explored else -np.inf
        self.log_bound = float(np.logaddexp(self.log_za, self.log_u + self.log_tau))
        if not np.isfinite(self.log_bound):
            raise FloatingPointError("Nonfinite parity envelope")

    @staticmethod
    def _inverse(rows):
        rows, masks = rows.copy(), [1 << i for i in range(len(rows))]
        for bit in range(len(rows)):
            pivot = next(i for i in range(bit, len(rows)) if rows[i] & (1 << bit))
            rows[bit], rows[pivot] = rows[pivot], rows[bit]
            masks[bit], masks[pivot] = masks[pivot], masks[bit]
            for i in range(len(rows)):
                if i != bit and rows[i] & (1 << bit):
                    rows[i] ^= rows[bit]
                    masks[i] ^= masks[bit]
        return masks

    def coordinates(self, payload):
        return sum((((g & payload).bit_count() & 1) << i) for i, g in enumerate(self.rows))

    def payload(self, coordinate):
        return sum((((mask & coordinate).bit_count() & 1) << i) for i, mask in enumerate(self.inverse))

    def candidates(self, order, budget):
        produced = 0
        for count in range(min(order, self.rank) + 1):
            for changed in itertools.combinations(range(self.rank), count):
                coordinate = self.mode
                for i in changed:
                    coordinate ^= 1 << i
                # If rank deficient, this is one representative; the complement retains every other word.
                yield self.payload(coordinate)
                produced += 1
                if produced >= budget:
                    return

    def score(self, payload):
        return self.h0 + sum(v * (1 - 2 * ((g & payload).bit_count() & 1)) for g, v in self.coeff.items())

    def base_logq(self, payload):
        c = self.coordinates(payload)
        return sum(self.logone[i] if (c >> i) & 1 else self.logzero[i] for i in range(self.model.b))

    def _remaining(self, node, depth):
        if depth == self.model.b:
            value = -np.inf
        else:
            children = [self._remaining(node[c], depth + 1) if c in node else 0. for c in (0, 1)]
            value = float(np.logaddexp(self.logzero[depth] + children[0], self.logone[depth] + children[1]))
        node["mass"] = value
        return value

    def draw(self, rng):
        if self.explored and math.log(max(rng.random(), np.finfo(float).tiny)) < self.log_za - self.log_bound:
            probabilities = np.exp(np.array(list(self.explored.values())) - self.log_za)
            return self.explored_ids[int(rng.choice(len(probabilities), p=probabilities))]
        node, coordinate = self.trie if self.explored else None, 0
        for i in range(self.model.b):
            logp = np.array([self.logzero[i], self.logone[i]])
            if node is not None:
                logp += [node[c]["mass"] if c in node else 0. for c in (0, 1)]
            normalizer = float(logsumexp(logp))
            if not np.isfinite(normalizer):
                raise FloatingPointError("Attempted to sample an empty complement")
            value = int(rng.random() < math.exp(logp[1] - normalizer))
            coordinate |= value << i
            node = node.get(value) if node is not None else None
        return self.payload(coordinate)

    def envelope(self, payload):
        return self.explored.get(payload, self.log_u + self.base_logq(payload))


class ParityProposal:
    def __init__(self, work, residual, variance, repair, flip_order, max_refine):
        work.stats["proposal_builds"] += 1
        self.model = work.model
        self.references = [ParityReference(self.model, p, residual, variance, work, flip_order, max_refine, repair)
                           for p in range(self.model.p)]
        self.bounds = np.array([ref.log_bound for ref in self.references])
        self.log_z = float(logsumexp(self.bounds))

    def draw(self, rng):
        p = int(rng.choice(self.model.p, p=np.exp(self.bounds - self.log_z)))
        return (p << self.model.b) | self.references[p].draw(rng)

    def logq(self, word):
        p, v = self.model.split(word)
        return self.references[p].envelope(v) - self.log_z


def candidates(work, residual, variance, patterns, list_size, mode="scl", flip_order=1, max_refine=32):
    model, found = work.model, set()
    code = PolarCode(model.b, model.code.n, model.code.crc_bits, list_size)
    for p in work.patterns(residual, patterns):
        if mode in ("scl", "union"):
            work.stats["polar_calls"] += 1
            payloads, _ = code.decode_list(2 * model.amplitudes[p] * residual[model.positions[p]] / variance)
            work.stats["empty_polar_lists"] += not len(payloads)
            found.update(model.word(p, v) for v in payloads)
        if mode in ("basis", "union"):
            ref = ParityReference(model, p, residual, variance, work)
            found.update((int(p) << model.b) | v for v in ref.candidates(flip_order, max_refine))
    return sorted(found)


def beam_states(s, work, width, patterns, list_size, mode, flip_order, max_refine, rng):
    states = [()]
    scores = np.zeros(1)
    for j in range(s.k):
        unique = {}
        for words in states:
            residual = s.y.copy()
            for w in words:
                residual -= work.atom(w)
            remaining = s.k - j - 1
            variance = s.noise_var + remaining / work.model.n
            ids = candidates(work, residual - remaining * work.model.mean, variance, patterns, list_size,
                             mode, flip_order, max_refine)
            if not ids:
                ids = [work.model.uniform(rng)]
                work.stats["empty_list_prior_draws"] += 1
            for w in ids:
                child = tuple(sorted((*words, w)))
                if child in unique:
                    continue
                r = residual - work.atom(w) - remaining * work.model.mean
                orbit = math.lgamma(j + 2) - sum(math.lgamma(c + 1) for c in Counter(child).values())
                unique[child] = orbit - r @ r / (2 * variance)
            work.stats["beam_contexts"] += 1
        ranked = sorted(unique, key=lambda w: (-unique[w], w))[:width]
        states, scores = ranked, np.array([unique[w] for w in ranked])
    return states, scores


def sparse_readout(states, weights, k):
    tails = {}
    for words, weight in zip(states, weights):
        for word, count in Counter(words).items():
            row = tails.setdefault(int(word), [])
            row.extend([0.] * (count - len(row)))
            for j in range(count):
                row[j] += float(weight)
    heap = [(-row[0], word, 0) for word, row in tails.items()]
    heapq.heapify(heap)
    counts = Counter()
    for _ in range(k):
        if not heap:
            raise ValueError("Empirical posterior cannot allocate the declared count budget")
        _, word, rank = heapq.heappop(heap)
        counts[word] += 1
        if rank + 1 < len(tails[word]):
            heapq.heappush(heap, (-tails[word][rank + 1], word, rank + 1))
    return dict(counts), [[word, tails[word]] for word in sorted(tails)]


def metadata(work, tails, target):
    return {"tail_probabilities": tails, "candidate_ids": sorted(work.atoms), "target": target,
            "full_alphabet_scan": False, "readout": "sparse_count_tails", "trainable_parameters": 0,
            "work": dict(work.stats), "cached_waveform_bytes": sum(x.nbytes for x in work.atoms.values()),
            "excess_risk_bound": None, "certificate": "none: finite search/sampling budget"}


def tractable_beam(scenario, *, model=None, width=16, patterns=8, list_size=16, flip_order=1, max_refine=32,
                   candidate_method="scl", seed=0):
    positive_int(width=width, patterns=patterns, list_size=list_size, max_refine=max_refine)
    if candidate_method not in ("scl", "basis", "union") or not isinstance(flip_order, int) or flip_order < 0:
        raise ValueError("Invalid candidate method or flip order")
    _, work = context(scenario, model)
    states, score = beam_states(scenario, work, width, patterns, list_size, candidate_method, flip_order, max_refine,
                                np.random.default_rng(seed))
    weight = np.exp(score - logsumexp(score))
    counts, tails = sparse_readout(states, weight, scenario.k)
    meta = metadata(work, tails, "posterior_conditioned_on_discovered_multisets")
    meta.update({"retained_states": len(states), "candidate_method": candidate_method, "seed": int(seed)})
    return counts, meta


class ListProposal:
    def __init__(self, work, residual, variance, patterns, list_size, defensive, pair_method=None):
        self.work, self.defensive, self.size = work, defensive, 2 if pair_method else 1
        model = work.model
        if pair_method is None:
            self.words = [(w,) for w in candidates(work, residual, variance, patterns, list_size)]
        else:
            found = set()
            for p in work.patterns(residual, patterns):
                args = (model.code, residual[model.positions[p]], model.amplitudes[p], variance)
                if pair_method == "joint":
                    pairs, _ = joint_pair_list(*args, list_size)
                else:
                    pairs, _ = xor_pair_list(*args, list_size, list_size)
                work.stats["pair_polar_calls"] += 1
                found.update(tuple(model.word(p, v) for v in pair) for pair in pairs)
            self.words = sorted(found)
        self.log_prior = -self.size * model.B * math.log(2)
        self.logp = {}
        if self.words:
            score = []
            for words in self.words:
                r = residual.copy()
                for w in words:
                    r -= work.atom(w)
                score.append(-r @ r / (2 * variance))
            values = np.array(score) - logsumexp(score)
            self.logp = dict(zip(self.words, values))
        work.stats["list_proposal_builds"] += 1

    def draw(self, rng):
        if not self.words or rng.random() < self.defensive:
            return tuple(self.work.model.uniform(rng) for _ in range(self.size))
        p = np.exp([self.logp[w] for w in self.words])
        return self.words[int(rng.choice(len(p), p=p))]

    def logq(self, words):
        if not self.words or self.defensive == 1:
            return self.log_prior
        return float(np.logaddexp(math.log(self.defensive) + self.log_prior,
                                 math.log1p(-self.defensive) + self.logp.get(tuple(words), -np.inf)))


def tractable_chain(scenario, *, model=None, samples=64, burn_in=64, thinning=1, patterns=8, list_size=16,
                    init_width=4, defensive=.1, flip_order=1, max_refine=32, cache_centers=4,
                    proposal_method="parity", init_method="scl", seed=0):
    positive_int(samples=samples, thinning=thinning, patterns=patterns, list_size=list_size,
                 init_width=init_width, max_refine=max_refine)
    if any(not isinstance(x, int) or x < 0 for x in (burn_in, flip_order, cache_centers)):
        raise ValueError("Require nonnegative integer burn-in, flip order and cache centers")
    if not 0 < defensive <= 1 or proposal_method not in ("parity", "refined", "cached", "scl", "joint", "xor"):
        raise ValueError("Invalid proposal method or defensive probability")
    if init_method not in ("scl", "basis", "union"):
        raise ValueError("Invalid chain initialization method")
    model, work = context(scenario, model)
    rng = np.random.default_rng(seed)
    initial, _ = beam_states(scenario, work, init_width, patterns, list_size, init_method, flip_order, max_refine, rng)
    words = list(initial[0])
    initial_counts = sorted(Counter(words).items())
    residual = scenario.y - sum((work.atom(w) for w in words), np.zeros(model.n))
    cache = []
    if proposal_method == "cached" and cache_centers:
        centers = [scenario.y.copy(), *(residual + work.atom(w) for w in words)]
        for center in centers[:cache_centers]:
            cache.append((center, ParityProposal(work, center, scenario.noise_var, True, flip_order, max_refine)))
    states, max_oscillation = [], 0.
    moves = burn_in + samples * thinning
    for step in range(moves):
        # Mixed pair schemes retain single-slot moves; selecting indices is independent of current word values.
        pair = proposal_method in ("joint", "xor") and scenario.k >= 2 and step % 2 == 1
        slots = rng.choice(scenario.k, size=2 if pair else 1, replace=False)
        previous = tuple(words[int(i)] for i in slots)
        outside = residual.copy()
        for w in previous:
            outside += work.atom(w)
        if proposal_method in ("scl", "joint", "xor"):
            q = ListProposal(work, outside, scenario.noise_var, patterns, list_size, defensive,
                             proposal_method if pair else None)
            proposed = q.draw(rng)
            log_ratio = q.logq(previous) - q.logq(proposed)
        else:
            if cache:
                index = min(range(len(cache)), key=lambda i: np.linalg.norm(outside - cache[i][0]))
                center, q = cache[index]
                work.stats["cache_hits"] += 1
                max_oscillation = max(max_oscillation, 2 * float(np.linalg.norm(outside - center)) / scenario.noise_var)
            else:
                q = ParityProposal(work, outside, scenario.noise_var, proposal_method != "parity", flip_order, max_refine)
            proposed = (q.draw(rng),)
            log_ratio = q.logq(previous[0]) - q.logq(proposed[0])
        new_residual = outside.copy()
        for w in proposed:
            new_residual -= work.atom(w)
        log_ratio += (residual @ residual - new_residual @ new_residual) / (2 * scenario.noise_var)
        if not np.isfinite(log_ratio):
            raise FloatingPointError("Nonfinite physical MH ratio")
        if math.log(max(rng.random(), np.finfo(float).tiny)) < min(0., log_ratio):
            work.stats["changed_moves"] += previous != proposed
            for i, w in zip(slots, proposed):
                words[int(i)] = w
            residual = new_residual
            work.stats["accepted_moves"] += 1
        work.stats["pair_moves" if pair else "single_moves"] += 1
        if step >= burn_in and (step - burn_in) % thinning == thinning - 1:
            states.append(tuple(words))
    counts, tails = sparse_readout(states, np.full(samples, 1 / samples), scenario.k)
    meta = metadata(work, tails, "finite_chain_empirical_count_tails; exact_IID_posterior_invariant_kernels")
    meta.update({"samples": samples, "burn_in": burn_in, "thinning": thinning, "seed": int(seed),
                 "proposal_method": proposal_method, "initialization": f"bounded_{init_method}_multiset_beam",
                 "initial_counts": initial_counts,
                 "cache_centers_built": len(cache), "maximum_cache_score_oscillation_bound": max_oscillation,
                 "mixing_guarantee": None})
    return counts, meta


class ExchangeableProposal:
    """Common one-word parity law mixed with an orbit-corrected multiset beam."""

    def __init__(self, scenario, work, states, score, beam_weight, flip_order, max_refine):
        self.work, self.k, self.beam_weight = work, scenario.k, beam_weight
        variance = scenario.noise_var + max(0, self.k - 1) / work.model.n
        self.word_law = ParityProposal(work, scenario.y, variance, True, flip_order, max_refine)
        self.states = states
        self.log_probabilities = score - logsumexp(score)
        self.orbits = np.array([math.lgamma(self.k + 1) - sum(math.lgamma(c + 1) for c in Counter(w).values())
                                for w in states])
        self.beam_density = {w: float(lp - orbit) for w, lp, orbit in zip(states, self.log_probabilities, self.orbits)}
        t = scenario.y / variance
        log_prior = -self.k * work.model.B * math.log(2)
        self.log_evidence_lower = float(logsumexp(score)) + log_prior
        # Completing the square bounds the *whole* configuration, including every user interaction.
        self.log_envelope = self.k * (self.word_law.log_z - work.model.B * math.log(2))
        self.log_envelope += -float(t @ scenario.y) + scenario.noise_var * float(t @ t) / 2
        if self.k == 1:
            # Here t=y/sigma^2, so the discarded square is exactly ||phi||^2=1.
            self.log_envelope -= 1 / (2 * scenario.noise_var)
        self.log_weight_bound = self.log_envelope - math.log1p(-beam_weight)

    def draw(self, rng):
        if rng.random() < self.beam_weight:
            words = list(self.states[int(rng.choice(len(self.states), p=np.exp(self.log_probabilities)))])
            rng.shuffle(words)
            return tuple(words)
        return tuple(self.word_law.draw(rng) for _ in range(self.k))

    def logq(self, words):
        product = math.log1p(-self.beam_weight) + sum(self.word_law.logq(w) for w in words)
        if self.beam_weight == 0:
            return product
        beam = math.log(self.beam_weight) + self.beam_density.get(tuple(sorted(words)), -np.inf)
        return float(np.logaddexp(product, beam))


def expected_is_regret(log_kappa, samples, bits_per_message, k):
    """Expected excess count-PUPE bound; the expectation is over independent importance draws."""
    logarithm = math.log(2 * k) + bits_per_message * math.log(2)
    log_x = max(0., log_kappa) + math.log(logarithm / samples)
    if log_x >= -math.log(8):
        return 1.
    x = math.exp(log_x)
    return min(1., 2 * (math.sqrt(2 * x) + x / 3))


def tractable_importance(scenario, *, model=None, samples=256, width=32, patterns=8, list_size=16,
                         flip_order=1, max_refine=32, beam_weight=.5, seed=0):
    positive_int(samples=samples, width=width, patterns=patterns, list_size=list_size, max_refine=max_refine)
    if not isinstance(flip_order, int) or flip_order < 0 or not 0 <= beam_weight < 1:
        raise ValueError("Require nonnegative flip order and beam_weight in [0,1)")
    model, work = context(scenario, model)
    rng = np.random.default_rng(seed)
    states, score = beam_states(scenario, work, width, patterns, list_size, "basis", flip_order, max_refine, rng)
    proposal = ExchangeableProposal(scenario, work, states, score, beam_weight, flip_order, max_refine)
    draws, log_weights = [], []
    for _ in range(samples):
        words = proposal.draw(rng)
        residual = scenario.y - sum((work.atom(w) for w in words), np.zeros(model.n))
        logf = -float(residual @ residual) / (2 * scenario.noise_var) - scenario.k * model.B * math.log(2)
        draws.append(words)
        log_weights.append(logf - proposal.logq(words))
    log_weights = np.array(log_weights)
    if not np.isfinite(log_weights).all() or not np.isfinite(proposal.log_weight_bound):
        raise FloatingPointError("Nonfinite independent-importance weights or envelope")
    if np.max(log_weights) > proposal.log_weight_bound + 1e-8:
        raise AssertionError("Global proposal envelope violated")
    normalized = np.exp(log_weights - logsumexp(log_weights))
    counts, tails = sparse_readout(draws, normalized, scenario.k)
    log_kappa = max(0., proposal.log_weight_bound - proposal.log_evidence_lower)
    bound = expected_is_regret(log_kappa, samples, model.B, scenario.k)
    meta = metadata(work, tails, "independent_importance_corrected_full_IID_posterior")
    meta.update({"samples": samples, "seed": int(seed), "beam_weight": beam_weight,
                 "effective_sample_size": float(1 / np.square(normalized).sum()),
                 "log_mean_weight": float(logsumexp(log_weights) - math.log(samples)),
                 "log_evidence_lower": proposal.log_evidence_lower, "log_global_weight_bound": proposal.log_weight_bound,
                 "log_density_ratio_bound": log_kappa, "expected_excess_risk_bound": bound,
                 "certificate": "expected regret over independent draws, conditional on observation/proposal; floating point",
                 "proposal": "common_tilted_parity_product_plus_multiset_beam", "retained_states": len(states)})
    return counts, meta
