"""Independent finite enumeration of SO-SCL, sparse cavity moments and complete hard-readout checks."""

import itertools
import json

import numpy as np

from paper_soft import SoftSCL, cavity_llrs, hard_action, run, soft_stage
from baselines.polar import PolarCode, polar_transform
from framework.polar_tractable import PolarModel, PolarScenario, bits


def finite_prefix_tree(code, llr, *, terminal=False):
    us = np.array(list(itertools.product((0, 1), repeat=code.n)), dtype=np.uint8)
    cs = polar_transform(us)
    probs = np.exp(-np.logaddexp(0., -(1.-2.*cs)*llr).sum(axis=1))
    np.testing.assert_allclose(probs.sum(), 1.)
    parents = [np.arange(len(us))]
    lost = 0.
    for i in range(code.n):
        children = [par[us[par,i] == b] for par in parents for b in ((0,) if code.frozen[i] else (0,1))]
        children.sort(key=lambda ids: -sum(probs[ids]))
        if terminal and i==code.n-1:
            tail = children[code.list_size:]
            final_ids = np.concatenate(tail) if tail else np.array([],dtype=int)
        lost += sum(sum(probs[ids]) for ids in children[code.list_size:]) * 2.**(-int(code.frozen[i+1:].sum()))
        parents = children[:code.list_size]
    indices = np.concatenate(parents)
    total = probs[indices].sum() + lost
    mean = (probs[indices] @ (1.-2.*cs[indices]) + lost*np.tanh(llr/2.))/total
    if terminal:
        last = probs[final_ids].sum()/total
        corrected = mean + probs[final_ids] @ (1.-2.*cs[final_ids])/total - last*np.tanh(llr/2.)
        return mean, corrected, np.log(total), lost/total, last
    return mean, np.log(total), lost/total


def main():
    rng = np.random.default_rng(100304)
    tested = 0
    errors = []
    for width in (1, 2, 3, 8, 64):
        code = SoftSCL(3, 8, 0, width)
        for _ in range(7):
            llr = rng.normal(0, 2, 8)
            actual, log_z, lost = code.decode_means(llr)
            reference, exact_log_z, reference_lost = finite_prefix_tree(code, llr)
            errors.append(np.max(abs(actual-reference)))
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
            np.testing.assert_allclose((log_z,lost), (exact_log_z,reference_lost), rtol=0, atol=1e-12)
            if width >= 8:
                assert lost == 0
            tested += 1

    positions = np.array([[0,1,3],[1,2,3],[0,2,3]])
    amps = rng.uniform(.1,1,positions.shape)
    means = rng.uniform(-1,1,positions.shape)
    y = rng.normal(size=4)
    llrs = cavity_llrs(y, positions, amps, means, .23)
    for p,i in np.ndindex(positions.shape):
        others = [(a,j) for a,j in np.ndindex(positions.shape) if a != p and positions[a,j] == positions[p,i]]
        moment = variance = 0.
        for signs in itertools.product((-1,1),repeat=len(others)):
            probability = np.prod([(1.+s*means[a,j])/2. for s,(a,j) in zip(signs, others, strict=True)])
            value = sum(s*amps[a,j] for s,(a,j) in zip(signs, others, strict=True))
            moment += probability*value
            variance += probability*value**2
        direct = 2.*amps[p,i]*(y[positions[p,i]]-moment)/(.23+variance-moment**2)
        np.testing.assert_allclose(llrs[p,i], direct, rtol=0, atol=1e-12)

    code = PolarCode(4, 16, 4, 128)
    m = PolarModel(code,np.arange(32).reshape(2,16),np.ones((2,16))/4.,32)
    words = [5,20]
    s = PolarScenario(m, m.waveform(words[0])+m.waveform(words[1]), 2, .001)
    for w in range(32):
        p,v = m.split(w)
        means = 1.-2.*code.encode(bits(v,4))[None, :]
        ids, count = hard_action(m, np.array([p]), means, {})
        assert ids == {w:1} and count == 1
    legal_data = np.array(list(itertools.product((0,1), repeat=4)),np.uint8)
    signs = 1.-2.*code.encode(legal_data)
    field = rng.normal(0.,1.,code.n)
    energy = rng.uniform(.02,.3,code.n)
    relative = np.exp(signs@field-energy.sum())
    full_bf = relative.mean()
    full_mean = relative@signs/relative.sum()
    ratio = np.cosh(field)-full_mean*np.sinh(field)
    leave_mean = (full_mean*np.cosh(field)-np.sinh(field))/ratio
    leave_bf = full_bf*np.exp(energy)*ratio
    for i in range(code.n):
        omit = np.arange(code.n)!=i
        direct_ratio = np.exp(signs[:,omit]@field[omit]-energy[omit].sum())
        np.testing.assert_allclose(leave_bf[i],direct_ratio.mean(),rtol=2e-14,atol=1e-12)
        np.testing.assert_allclose(leave_mean[i],direct_ratio@signs[:,i]/direct_ratio.sum(),rtol=0,atol=1e-12)
    a,b, meta = soft_stage(s,{})
    assert a == b == dict.fromkeys(words,1) and len(meta["rounds"]) == 1
    assert run(s)[0] == run(s, full_user=True)[0] == b
    print(json.dumps(dict(independent_prefix_enumerations=tested, max_chip_error=max(errors),
                          exhaustive_other_chip_moments=len(llrs.ravel()), leave_chip_mean_and_odds_checks=code.n, legal_waveform_reads=32, two_user_smoke=True)))


if __name__ == "__main__":
    main()
