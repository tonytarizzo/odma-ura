"""Estimate paired, fixed-grid 5% crossings for the balanced load experiment."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '039_polar_paper'))
from paper_curve_analysis import TARGET, crossing, finite, interval, report_crossing

ARMS = ('DEC-38', 'DEC-39')
BOOTSTRAPS = 2000


def analyze(k, energies_cdb, counts):
    """Input axes are independent holdout frames, increasing energy (centidB), then DEC-38/39."""
    grid, x = np.asarray(energies_cdb), np.asarray(counts)
    if type(k) is not int or k < 1:
        raise ValueError('Expected a positive integer load')
    if grid.ndim != 1 or len(grid) < 2 or grid.dtype.kind not in 'iu' or np.any(grid[:-1] >= grid[1:]):
        raise ValueError('Expected at least two strictly increasing integer energies in centidB')
    if x.ndim != 3 or x.shape[1:] != (len(grid), len(ARMS)) or len(x) < 2 or x.dtype.kind not in 'iu':
        raise ValueError('Expected integer missed copies indexed by at least two frames, energy and receiver')
    if np.any(x < 0) or np.any(x > k):
        raise ValueError('Missed copies per frame must be between zero and the known load')
    x = x.astype(np.int64, copy=False)
    n = len(x)
    users = n * k
    db = grid.astype(float) / 100
    totals = x.sum(axis=0)
    rng = np.random.default_rng([431903, k])
    weights = rng.multinomial(n, np.full(n, 1/n), size=BOOTSTRAPS)
    draws = (weights @ x.reshape(n, -1)).reshape(BOOTSTRAPS, len(db), len(ARMS))

    arms, estimates, boot = {}, [], []
    for a, name in enumerate(ARMS):
        point = crossing(db, totals[:, a], users)
        resampled = [crossing(db, v[:, a], users) for v in draws]
        estimates.append(point)
        boot.append(resampled)
        arms[name] = dict(**report_crossing(point), misses=totals[:, a].tolist(), pupe=(totals[:, a] / users).tolist(),
                          pointwise_95_pupe=np.quantile(draws[:, :, a] / users, [.025, .975], axis=0).T.tolist(),
                          paired_bootstrap_95_db=interval([v[1] for v in resampled], [v[2] for v in resampled]),
                          bootstrap_status=dict(Counter(v[0] for v in resampled)))

    a, b = estimates
    low, high = a[1] - b[2], a[2] - b[1]
    paired = list(zip(*boot, strict=True))
    advantage = x[:, :, 0] - x[:, :, 1]
    contrast = dict(
        snr_db_saved_by_39=finite(low) if low == high else None, censor_bounds_db=[finite(low), finite(high)],
        paired_bootstrap_95_db=interval([a[1] - b[2] for a, b in paired], [a[2] - b[1] for a, b in paired]),
        both_crossings_observed_resamples=sum(a[1] == a[2] and b[1] == b[2] for a, b in paired),
        pupe_saved_by_39=(advantage.mean(axis=0) / k).tolist(),
        pointwise_95_pupe_saved_by_39=np.quantile((draws[:, :, 0] - draws[:, :, 1]) / users, [.025, .975], axis=0).T.tolist(),
        improved_frames_by_39=(advantage > 0).sum(axis=0).tolist(), worsened_frames_by_39=(advantage < 0).sum(axis=0).tolist())
    return dict(k=k, frames=n, target=TARGET, energies_db=db.tolist(), arms=arms, contrast_38_minus_39=contrast,
                bootstrap_replicates=BOOTSTRAPS,
                uncertainty='Pointwise 95% frame bootstrap, conditional on the pilot-selected grid. One draw of frames spans all '
                            'energies and both receivers at this load. Interpolation assumes a linear mean PUPE between adjacent '
                            'energies. Censored and ambiguous resamples contribute their full bounds. Null endpoints are unbounded.')
