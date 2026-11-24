"""Assess required-energy uncertainty at the resolution of the actual tested grid."""
import argparse

import numpy as np

import load_curve as study
from analysis import BOOTSTRAPS, crossing, interval, finite


def endpoints(c):
    # A crossing between measurements can lie anywhere in the observed bracket.
    return tuple(c[3]) if c[3] is not None else (c[1], c[2])


def grid_bands(k, energies, counts):
    n = len(counts)
    x = np.asarray(energies, dtype=float) / 100
    users = n * k
    rng = np.random.default_rng([431903, k])
    weights = rng.multinomial(n, np.full(n, 1/n), size=BOOTSTRAPS)
    resampled = (weights @ counts.reshape(n, -1)).reshape(BOOTSTRAPS, len(x), len(study.ARMS))
    paired = []
    arms = {}
    for a, name in enumerate(study.ARMS):
        low, high = endpoints(crossing(x, counts[:, :, a].sum(0), users))
        bounds = np.array([endpoints(crossing(x, v[:, a], users)) for v in resampled])
        arms[name] = dict(observed_threshold_bounds_db=[finite(low), finite(high)],
                          grid_bootstrap_95_db=interval(bounds[:, 0], bounds[:, 1]))
        paired.append(bounds)
    left, right = paired
    return dict(arms=arms, dec38_minus_dec39_grid_bootstrap_95_db=interval(left[:, 0]-right[:, 1], left[:, 1]-right[:, 0]))


def report(preview):
    if preview:
        import preview12
        out = preview12.OUT
    else:
        out = study.OUT
    summary = study.read(out/'summary.json')
    revision = study.receiver_version()
    if summary['receiver_release'] != revision:
        raise ValueError('Incompatible receiver version')
    if not preview and summary['release'] != study.declaration(revision):
        raise ValueError('Incompatible confirmation study version')
    by_load = {}
    for key, result in summary['loads'].items():
        k = int(key)
        energies = [round(v * 100) for v in result['energies_db']]
        if preview:
            counts = preview12.load(k, energies, revision)[0]
        else:
            counts = study.collect(out, revision, k, 'confirm', energies)[0]['count_misses']
        if len(counts) != result['frames'] or any(counts[:, :, a].sum(0).tolist() != result['arms'][arm]['misses']
                                                 for a, arm in enumerate(study.ARMS)):
            raise ValueError(f'Frame receipts disagree with the plotted curve at K={k}')
        data = grid_bands(k, energies, counts)
        if preview:
            original = list(study.PREVIEW[k])
            coarse = counts[:, [energies.index(e) for e in original], :]
            data['original_grid_estimate_db'] = {name: v['snr_db'] for name, v in study.analyze(k, original, coarse)['arms'].items()}
        by_load[key] = data
        for name in study.ARMS:
            row = data['arms'][name]
            print(k, name, 'threshold bracket', row['observed_threshold_bounds_db'], 'grid-aware 95%', row['grid_bootstrap_95_db'])
    evidence = dict(panels=by_load, receiver_release=revision, preview=preview, replicates=BOOTSTRAPS,
                    interpretation='Frame-bootstrap percentile bounds retain the measured energy bin for each crossing. '
                                   'Censored and recrossing draws retain their open bounds. The intervals are pointwise by load '
                                   'and conditional on the measured grid. The local preview also reused the same frames to choose its grid.')
    study.equal_file(out/'grid_precision.json', evidence)
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview12', action='store_true')
    report(p.parse_args().preview12)
