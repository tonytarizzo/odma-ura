"""Recompute complete receivers on the original balanced B100/K100 scalar channel.

The common panel uses sixteen of the 0.5 dB high-panel geometries and includes the
five-pass scalar transfer of Li et al.'s data stage. The high and waterfall panels
use disjoint geometries. Every panel checks transmitted and estimated multiplicities.
The published Li receiver also uses pilots, fading and sixteen receive antennas.
"""
from argparse import ArgumentParser
from collections import Counter
import time

from frames import PANELS, frame, rollback_frames
from paper_native import SEEDS, paper_curve
from paper_soft import run as soft_data
from framework.polar_catalogue import run as catalogue


REFERENCE = {
    'common': {'native': (132, 21), 'DEC-35': (90, 4), 'scalar_soft_five': (1500, 0), 'DEC-38': (9, 9), 'DEC-39': (8, 7)},
    'waterfall': {'native': (2146, 56), 'DEC-35': (1721, 35), 'DEC-38': (199, 48), 'DEC-39': (455, 57)},
    'high': {'native': (838, 150), 'DEC-35': (423, 37), 'DEC-38': (140, 52), 'DEC-39': (81, 53)},
}
SEED_PANELS = {
    'common': [*SEEDS[:8], *SEEDS[16:24]],
    'waterfall': [*range(989201, 989225), *range(989301, 989325)],
    'high': PANELS['A'] + PANELS['B'],
}


def replay(seed, panel='common'):
    scenario, labels, geometry = frame(seed, .25 if panel == 'waterfall' else .5)
    records = {}
    for name in REFERENCE[panel]:
        start = time.perf_counter()
        if name == 'native':
            accepted, _ = rollback_frames.native(scenario)
            output = dict.fromkeys(accepted, 1)
        elif name == 'scalar_soft_five':
            output, _ = soft_data(scenario, full_user=True)
        else:
            output, meta = catalogue(name, scenario)
            complete = meta['continuation']['complete'] if name == 'DEC-35' else meta['complete']
            if not complete:
                raise RuntimeError(f'{name} returned an incomplete receiver trace')
        seconds = geometry + time.perf_counter() - start
        score = paper_curve.score(labels, output)
        records[name] = dict(misses=score['count_misses'], false=score['false_copies'], seconds=seconds)
    return records


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=REFERENCE, default='common')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    seeds = SEED_PANELS[args.panel]
    if args.seed is not None:
        if args.seed not in seeds:
            parser.error('the seed must belong to the specified panel')
        seeds = [args.seed]
    totals = {name: Counter() for name in REFERENCE[args.panel]}
    for seed in seeds:
        row = replay(seed, args.panel)
        for name, record in row.items():
            totals[name].update(record)
        print(seed, {k: z['misses'] for k, z in row.items()}, flush=True)
    print('frames', len(seeds), 'transmitted', 100 * len(seeds))
    for name, record in totals.items():
        actual = (record['misses'], record['false'])
        print(name, 'misses', actual[0], 'false', actual[1], 'seconds', round(record['seconds'], 2))
        if args.seed is None and actual != REFERENCE[args.panel][name]:
            raise RuntimeError(f'{name} full-panel outcome {actual} differed from {REFERENCE[args.panel][name]}')
