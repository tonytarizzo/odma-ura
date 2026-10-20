"""Tiny exhaustive certificates for Report 7; not a scalable decoder or a PUPE experiment."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import gammaln, logsumexp

from baselines import message_bits
from baselines.odma_polar import ODMAPolarBaseline
from baselines.polar import PolarCode, polar_transform
from benchmarks.polar_energy_geometry import balanced_energy
from benchmarks.ura_comparison import frame_metrics


def count_states(m, k, sampling):
    combinations = itertools.combinations if sampling == "distinct" else itertools.combinations_with_replacement
    return np.array([np.bincount(words, minlength=m) for words in combinations(range(m), k)])


def exact_posterior(phi, y, k, sigma2, sampling):
    """Enumerate unordered states; intended only for the tiny tests below."""
    states = count_states(phi.shape[1], k, sampling)
    log_prior = np.zeros(len(states)) if sampling == "distinct" else -gammaln(states + 1).sum(1)
    log_weight = log_prior - ((states @ phi.T - y) ** 2).sum(1) / (2 * sigma2)
    return states, np.exp(log_weight - logsumexp(log_weight))


def binary_rank(matrix):
    matrix = matrix.copy().astype(np.uint8)
    rank = 0
    for column in range(matrix.shape[1]):
        indices = np.flatnonzero(matrix[rank:, column])
        if not len(indices):
            continue
        pivot = rank + indices[0]
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(len(matrix)):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        rank += 1
    return rank


class PolarPosteriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = ODMAPolarBaseline(4, 12, 3217, prefix_bits=2, code_length=8, crc_bits=4, list_size=256)
        cls.energy, cls.balance_info = balanced_energy(cls.base.positions, cls.base.n)
        cls.bits = message_bits(np.arange(16), 4)
        cls.signs = 1.0 - 2.0 * cls.base.polar.encode(message_bits(np.arange(4), 2))
        cls.phi = np.zeros((12, 16))
        for p in range(4):
            cls.phi[np.ix_(cls.base.positions[p], np.arange(4 * p, 4 * p + 4))] = (
                cls.energy[p, :, None] ** .5 * cls.signs.T)
        cls.sigma2 = .3
        cls.y = cls.phi[:, [2, 6, 13]].sum(1) + np.random.default_rng(731).normal(0, cls.sigma2 ** .5, 12)

    def test_encoder_and_local_polar_factors(self):
        np.testing.assert_allclose((self.phi ** 2).sum(0), 1, atol=1e-14)
        mass = np.bincount(self.base.positions.ravel(), weights=self.energy.ravel(), minlength=12)
        np.testing.assert_allclose(mass, 4 / 12, atol=1e-10)
        code = self.base.polar
        payloads = message_bits(np.arange(4), 2)
        generator = code.encode(np.eye(2, dtype=np.uint8)).T
        np.testing.assert_array_equal((payloads @ generator.T) % 2, code.encode(payloads))
        # Independently enumerate all mother-code inputs and enforce frozen/CRC constraints.
        inputs = message_bits(np.arange(256), 8)
        frozen_ok = (inputs[:, code.frozen] == 0).all(1)
        valid_outputs = []
        for u in inputs[frozen_ok]:
            data = u[code.information]
            candidate = data[:code.payload_bits]
            if np.array_equal(polar_transform(u), code.encode(candidate)):
                valid_outputs.append(polar_transform(u))
        np.testing.assert_array_equal(np.unique(valid_outputs, axis=0), np.unique(code.encode(payloads), axis=0))

    def test_slot_count_posteriors_and_pair_energy_agree(self):
        for sampling in ("iid", "distinct"):
            states, posterior = exact_posterior(self.phi, self.y, 3, self.sigma2, sampling)
            slots = np.array(list(itertools.product(range(16), repeat=3)))
            if sampling == "distinct":
                slots = slots[np.array([len(set(row)) == 3 for row in slots])]
            waveforms = self.phi.T[slots]
            likelihood = -((self.y - waveforms.sum(1)) ** 2).sum(1) / (2 * self.sigma2)
            pair_score = (waveforms @ self.y).sum(1)
            pair_score -= sum((waveforms[:, i] * waveforms[:, j]).sum(1) for i in range(3) for j in range(i + 1, 3))
            np.testing.assert_allclose(likelihood, pair_score / self.sigma2
                                       - (self.y @ self.y + 3) / (2 * self.sigma2), atol=2e-14)
            probabilities = np.exp(likelihood - logsumexp(likelihood))
            by_state = {tuple(a): i for i, a in enumerate(states)}
            aggregated = np.zeros(len(states))
            for row, prob in zip(slots, probabilities):
                aggregated[by_state[tuple(np.bincount(row, minlength=16))]] += prob
            np.testing.assert_allclose(aggregated, posterior, atol=1e-14)

    def test_multiset_loss_and_optimal_readouts(self):
        truth = self.bits[[1, 1, 3]]
        self.assertAlmostEqual(frame_metrics(truth, self.bits[[1, 3]])["pupe"], 1 / 3)
        self.assertEqual(frame_metrics(truth, self.bits[[1, 3]])["list_pupe"], 0)
        self.assertEqual(frame_metrics(truth, truth)["pupe"], 0)
        self.assertAlmostEqual(frame_metrics(truth, self.bits[[1, 3]])["strict_pupe"], 2 / 3)
        for sampling in ("distinct", "iid"):
            states, posterior = exact_posterior(self.phi, self.y, 3, self.sigma2, sampling)
            means = posterior @ states
            chosen = np.argsort(-means)[:3]
            risks = np.array([1 - means[list(s)].sum() / 3 for s in itertools.combinations(range(16), 3)])
            self.assertAlmostEqual(1 - means[chosen].sum() / 3, risks.min(), places=14)
            # Primary PUPE: allocate K count units by posterior tail probabilities.
            tails = np.array([posterior @ (states >= j) for j in range(1, 4)]).T
            allocation = np.zeros(16, dtype=int)
            for _ in range(3):
                scores = np.array([tails[w, allocation[w]] if allocation[w] < 3 else -np.inf for w in range(16)])
                allocation[np.argmax(scores)] += 1
            risk = 1 - posterior @ np.minimum(states, allocation).sum(1) / 3
            all_risks = [1 - posterior @ np.minimum(states, a).sum(1) / 3 for a in count_states(16, 3, "iid")]
            self.assertAlmostEqual(risk, min(all_risks), places=14)

    def test_map_and_minimum_pupe_can_differ(self):
        # A posterior over six distinct pairs; no claim that this is a measured operating point.
        states = count_states(4, 2, "distinct")
        probability = np.array([.30, .22, .01, .24, .01, .22])
        map_action = np.flatnonzero(states[np.argmax(probability)])
        bayes_action = np.argsort(-(probability @ states))[:2]
        self.assertEqual(set(map_action), {0, 1})
        self.assertEqual(set(bayes_action), {1, 2})
        self.assertAlmostEqual(1 - (probability @ states)[map_action].sum() / 2, .46)
        self.assertAlmostEqual(1 - (probability @ states)[bayes_action].sum() / 2, .385)

    def test_conditional_polar_evidence_and_pattern_normalizers(self):
        residual = self.y - self.phi[:, [2, 13]].sum(1)
        logits = self.phi.T @ residual / self.sigma2
        gaussian = -((residual[:, None] - self.phi) ** 2).sum(0) / (2 * self.sigma2)
        np.testing.assert_allclose(logits - logsumexp(logits), gaussian - logsumexp(gaussian), atol=1e-14)
        local_scores = []
        for p in range(4):
            llr = 2 * self.energy[p] ** .5 * residual[self.base.positions[p]] / self.sigma2
            local_scores.append(self.signs @ llr / 2)
        np.testing.assert_allclose(np.concatenate(local_scores), logits, atol=1e-14)
        pattern_log_z = logsumexp(np.array(local_scores), axis=1)
        joint = np.exp(logits - logsumexp(logits)).reshape(4, 4)
        np.testing.assert_allclose(joint.sum(1), np.exp(pattern_log_z - logsumexp(pattern_log_z)), atol=1e-14)
        allowed = np.ones(16, dtype=bool)
        allowed[[2, 13]] = False
        self.assertAlmostEqual(np.exp(logsumexp(logits[allowed])),
                               np.exp(logsumexp(logits)) - np.exp(logits[[2, 13]]).sum(), places=10)

    def test_candidate_conditioning_mass_and_variational_identity(self):
        states, posterior = exact_posterior(self.phi, self.y, 3, self.sigma2, "iid")
        candidate = np.argsort(-(posterior @ states))[:8]
        keep = states[:, np.setdiff1d(np.arange(16), candidate)].sum(1) == 0
        mass = posterior[keep].sum()
        q = np.zeros(len(states))
        q[keep] = posterior[keep] / mass
        self.assertAlmostEqual(np.abs(q - posterior).sum() / 2, 1 - mass, places=14)
        self.assertAlmostEqual((q[keep] * np.log(q[keep] / posterior[keep])).sum(), -np.log(mass), places=14)
        missed_user_mass = (posterior @ states)[np.setdiff1d(np.arange(16), candidate)].sum() / 3
        self.assertLessEqual(missed_user_mass, 1 - mass + 1e-14)

    def test_full_support_polar_proposal_has_detailed_balance(self):
        target = np.exp(self.phi.T @ self.y / self.sigma2 - logsumexp(self.phi.T @ self.y / self.sigma2))
        candidate = np.argsort(-target)[:4]
        proposal = np.full(16, .1 / 16)
        proposal[candidate] += .9 * target[candidate] / target[candidate].sum()
        transition = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                if i != j:
                    transition[i, j] = proposal[j] * min(1, target[j] * proposal[i] / (target[i] * proposal[j]))
            transition[i, i] = 1 - transition[i].sum()
        flow = target[:, None] * transition
        np.testing.assert_allclose(flow, flow.T, atol=1e-15)
        np.testing.assert_allclose(target @ transition, target, atol=1e-15)

    def test_mean_field_free_energy_and_coordinate_update(self):
        rng = np.random.default_rng(19)
        q = rng.dirichlet(np.ones(16), size=3)
        slots = np.array(list(itertools.product(range(16), repeat=3)))
        energy = ((self.phi.T[slots].sum(1) - self.y) ** 2).sum(1) / (2 * self.sigma2)

        def free_energy(q):
            mu = q @ self.phi.T
            return ((self.y - mu.sum(0)) @ (self.y - mu.sum(0)) + 3 - (mu ** 2).sum()) / (2 * self.sigma2) \
                + (q * np.log(q)).sum()

        joint = np.prod(np.stack([q[k, slots[:, k]] for k in range(3)]), axis=0)
        self.assertAlmostEqual(free_energy(q), joint @ energy + (q * np.log(q)).sum(), places=12)
        expected_log = -(energy.reshape(16, 16, 16) * q[1, None, :, None] * q[2, None, None, :]).sum((1, 2))
        scores = self.phi.T @ (self.y - (q[1:] @ self.phi.T).sum(0)) / self.sigma2
        np.testing.assert_allclose(scores - logsumexp(scores), expected_log - logsumexp(expected_log), atol=1e-14)
        old = free_energy(q)
        q[0] = np.exp(scores - logsumexp(scores))
        self.assertLessEqual(free_energy(q), old + 1e-13)

    def test_colored_and_diagonal_gaussian_metrics(self):
        rng = np.random.default_rng(5)
        basis = rng.normal(size=(12, 12))
        covariance = .3 * np.eye(12) + basis @ basis.T / 12
        precision = np.linalg.inv(covariance)
        for p in range(4):
            a = np.zeros((12, 8))
            a[self.base.positions[p], np.arange(8)] = self.energy[p] ** .5
            score = self.signs @ a.T @ precision @ self.y
            quadratic = np.einsum("vi,ij,vj->v", self.signs, a.T @ precision @ a, self.signs)
            direct = -.5 * np.einsum("vi,ij,vj->v", self.y - self.signs @ a.T, precision,
                                    self.y - self.signs @ a.T)
            np.testing.assert_allclose(score - quadratic / 2, direct + self.y @ precision @ self.y / 2, atol=1e-13)
        # The retained nonlocal quadratic term is not constant over this actual code.
        self.assertGreater(np.ptp(quadratic), .01)
        variance = np.linspace(.2, 1.0, 12)
        scores = []
        offsets = []
        for p in range(4):
            idx = self.base.positions[p]
            llr = 2 * self.energy[p] ** .5 * self.y[idx] / variance[idx]
            offset = -.5 * (self.energy[p] / variance[idx]).sum()
            offsets.append(offset)
            scores.extend(self.signs @ llr / 2 + offset)
        direct = -.5 * (((self.y[:, None] - self.phi) ** 2) / variance[:, None]).sum(0)
        np.testing.assert_allclose(np.array(scores) - logsumexp(scores), direct - logsumexp(direct), atol=1e-13)
        self.assertGreater(np.ptp(offsets), .01)

    def test_count_map_replacement_and_projection_descent(self):
        remaining = np.bincount([2, 2], minlength=16)
        residual = self.y - self.phi @ remaining
        score = self.phi.T @ residual / self.sigma2 - np.log(remaining + 1)
        states = remaining[None, :] + np.eye(16, dtype=int)
        energy = ((states @ self.phi.T - self.y) ** 2).sum(1) / (2 * self.sigma2) + gammaln(states + 1).sum(1)
        np.testing.assert_allclose(score - score[0], -energy + energy[0], atol=1e-13)
        old = self.phi[:, [0, 2, 9]].T
        residual = self.y - old.sum(0)
        step = .2
        target = old + step * residual
        selected = np.argmax(target @ self.phi, axis=1)
        new = self.phi[:, selected].T
        before = .5 * residual @ residual
        after = .5 * np.square(self.y - new.sum(0)).sum()
        bound = before - (1 / (2 * step) - 3 / 2) * np.square(new - old).sum()
        self.assertLessEqual(after, bound + 1e-13)

    def test_current_code_two_user_identifiability(self):
        code = PolarCode(6, 32, crc_bits=8, list_size=8)
        generator = code.encode(np.eye(6, dtype=np.uint8)).T
        self.assertEqual(binary_rank(generator), 6)
        for difference in message_bits(np.arange(1, 64), 6):
            self.assertEqual(binary_rank(generator[(generator @ difference) % 2 == 0]), 5)
        signs = 1 - 2 * code.encode(message_bits(np.arange(64), 6)).astype(np.int16)
        sums = np.array([signs[i] + signs[j] for i, j in itertools.combinations_with_replacement(range(64), 2)])
        self.assertEqual(len(np.unique(sums, axis=0)), math.comb(65, 2))


if __name__ == "__main__":
    unittest.main()
