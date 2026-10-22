"""Independent finite posterior, proposal and invariant-kernel checks for Job 036."""

import itertools
import math
import unittest

import numpy as np
from scipy.special import logsumexp

from framework.polar_decoders import REGISTRY, Scenario, run
from framework.polar_decoder_core import count_tails
from framework.polar_decoder_combinations import (block_proposal, combined_particles, count_log_orbit, mixture_density,
                                                  mixture_draw, multiset_states, proposal_setup, terminal_moves)


class CombinationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(832)
        phi = rng.normal(size=(3, 4)); phi /= np.linalg.norm(phi, axis=0)
        self.s = Scenario(phi, np.array([.5, -.2, .8]), 3, .6)
        self.words = np.array(list(itertools.product(range(4), repeat=3)))
        self.log_l = -np.square(self.s.y - phi[:, self.words].sum(2).T).sum(1) / 1.2  # 2 sigma^2
        self.p = np.exp(self.log_l - logsumexp(self.log_l))

    def test_multiset_beam_exhaustion_recovers_exact_count_posterior(self):
        for variance in [.02, .6, 20.]:
            s = Scenario(self.s.phi, self.s.y, 3, variance)
            counts, got = run('DEC-14', s, width=math.comb(6, 3))
            ref, exact = run('REF', s)
            np.testing.assert_allclose(got['tails'], exact['tails'], atol=3e-14)
            np.testing.assert_array_equal(counts, ref)
            self.assertAlmostEqual(got['log_retained_evidence'], exact['log_evidence'], places=12)
        words, _, _ = multiset_states(self.s, width=4)
        self.assertEqual(len(np.unique(words, axis=0)), len(words))

    def test_multiset_orbit_mass_handles_duplicates(self):
        words = np.array([[0,0,0], [0,0,1], [0,1,2]])
        np.testing.assert_allclose(np.exp(count_log_orbit(words)), [1,3,6], atol=2e-15)
        # Alias columns leave the multinomial prior, rather than a uniform distribution over count states.
        s = Scenario(np.ones((1,3)), np.array([2.]), 3, 1.)
        _, got = run('DEC-14', s, width=10)
        np.testing.assert_allclose(got['tails'], np.tile([19/27,7/27,1/27], (3,1)), atol=1e-14)

    def test_every_mixture_normalizes_and_recovers_exact_posterior_algebraically(self):
        for pw,bw in [(0.,0.),(.3,0.),(0.,.3),(.2,.2)]:
            proposal = proposal_setup(self.s, product_weight=pw, beam_weight=bw, width=7)
            q = np.exp(mixture_density(self.s, self.words, proposal))
            self.assertAlmostEqual(q.sum(), 1., places=13)
            self.assertTrue(np.all(q >= .1/64 - 1e-15))
            weighted = q * np.exp(self.log_l - math.log(64) - np.log(q))
            tails = count_tails(self.words, weighted/weighted.sum(), 4)
            np.testing.assert_allclose(tails, run('REF', self.s)[1]['tails'], atol=1e-13)

    def test_mixture_sampler_matches_evaluated_density_with_beam_permutations(self):
        proposal = proposal_setup(self.s, defensive=.1, product_weight=.1, beam_weight=.7, width=4)
        rng = np.random.default_rng(91)
        words, logq = mixture_draw(self.s, 50_000, proposal, rng)
        np.testing.assert_allclose(logq, mixture_density(self.s, words, proposal), atol=1e-13)
        index = words @ np.array([16,4,1])
        histogram = np.bincount(index, minlength=64)/len(words)
        q = np.exp(mixture_density(self.s, self.words, proposal))
        np.testing.assert_allclose(histogram, q, atol=.007)

    def test_guided_block_proposal_and_metropolis_detailed_balance(self):
        pairs = np.array(list(itertools.product(range(4), repeat=2)))
        residual = self.s.y - self.s.phi[:, 2]
        outside = np.repeat(residual[None], len(pairs), axis=0)
        _, logq = block_proposal(self.s, outside, 2, words=pairs)
        q = np.exp(logq)
        self.assertAlmostEqual(q.sum(), 1., places=13)
        logp = -np.square(residual - self.s.phi[:, pairs].sum(2).T).sum(1)/(2*self.s.noise_var)
        p = np.exp(logp-logsumexp(logp))
        ratio = logp[None,:]-logp[:,None] + logq[:,None]-logq[None,:]
        transition = q[None,:] * np.exp(np.minimum(0., ratio))
        transition[np.diag_indices(16)] += 1-transition.sum(1)
        np.testing.assert_allclose(p[:,None]*transition, p[None,:]*transition.T, atol=2e-14)
        np.testing.assert_allclose(p @ transition, p, atol=2e-14)
        drawn,_ = block_proposal(self.s, np.repeat(residual[None], 30_000, axis=0), 2, rng=np.random.default_rng(71))
        histogram = np.bincount(drawn @ [4,1], minlength=16)/len(drawn)
        np.testing.assert_allclose(histogram, q, atol=.01)

    def test_actual_terminal_kernels_preserve_posterior_in_distribution(self):
        for size in [1,2,3]:
            rng = np.random.default_rng(915 + size)
            words = self.words[rng.choice(64, size=30_000, p=self.p)]
            moved, meta = terminal_moves(self.s, words, rng, moves=3, block_size=size, batch_size=1000)
            tails = count_tails(moved, np.full(len(moved),1/len(moved)), 4)
            np.testing.assert_allclose(tails, run('REF', self.s)[1]['tails'], atol=.015)
            self.assertEqual(meta['terminal_attempts'],90_000)

    def test_combined_importance_and_transport_match_exact_at_large_sample_size(self):
        exact = run('REF', self.s)[1]
        for name in ['DEC-15','DEC-16','DEC-17','DEC-18','DEC-19']:
            _, got = run(name, self.s, samples=12_000, batch_size=512, seed=319)
            np.testing.assert_allclose(got['tails'], exact['tails'], atol=.035)
            self.assertLess(abs(got['log_mean_weight']-exact['log_evidence']), .05)

    def test_smc_reduces_to_historical_gaussian_smc_without_new_components(self):
        _, old = run('DEC-07', self.s, batches=3, particles=32, seed=28, guide_log_floor=-1000.)
        _, new = combined_particles(self.s, batches=3, particles=32, seed=28, product_weight=0.)
        np.testing.assert_allclose(new['tails'], old['tails'], atol=1e-13)
        self.assertAlmostEqual(new['log_mean_weight'], old['log_mean_weight'],places=12)

    def test_smc_evidence_and_count_marks_agree_with_exact_across_batches(self):
        exact = run('REF', self.s)[1]
        for name in ['DEC-20','DEC-21','DEC-22','DEC-23','DEC-24']:
            _, got = run(name, self.s, batches=128, particles=32, seed=114)
            np.testing.assert_allclose(got['tails'],exact['tails'],atol=.055)
            self.assertLess(abs(got['log_mean_weight']-exact['log_evidence']),.12)

    def test_registry_contract_reproducibility_and_source_rejection(self):
        for i in range(14,25):
            name=f'DEC-{i:02}'
            params={} if i==14 else ({'samples':64} if i<20 else {'particles':16,'batches':2})
            a,ma=run(name,self.s,**params);b,mb=run(name,self.s,**params)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ma['tails'],mb['tails'])
            self.assertEqual(a.sum(),3)
            self.assertTrue(np.isfinite(ma['tails']).all())
            self.assertLessEqual(np.max(np.diff(ma['tails'],axis=1)),1e-12)
            with self.assertRaises(ValueError):
                run(name,Scenario(self.s.phi,self.s.y,3,.6,'distinct'))
        for params in [{'defensive':0.},{'product_weight':.9,'beam_weight':.5},{'beam_weight':-.1}]:
            with self.assertRaises(ValueError):proposal_setup(self.s,**params)
        for params in [{'moves':-1},{'block_size':4}]:
            with self.assertRaises(ValueError):terminal_moves(self.s,self.words,np.random.default_rng(0),**params)


if __name__ == '__main__':
    unittest.main()
