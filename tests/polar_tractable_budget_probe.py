"""Time one declared Job-037 frame per receiver; this is a cost probe, not a PUPE estimate."""

import argparse
import json
from pathlib import Path
import runpy
import time

import numpy as np
import torch

from benchmarks.polar_decoders import build_encoder, fixture, noise_variance, source_hashes
from framework.polar_decoders import REGISTRY, run
from framework.polar_tractable import PolarModel, PolarScenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--B', type=int, default=14)
    parser.add_argument('--K', type=int, default=26)
    parser.add_argument('--ebn0', type=float, default=4.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = runpy.run_path('jobs/037_polar_tractable/build_manifest.py')['manifest']()
    configs = [r['runs'][0] for r in rows if r['runs'][0]['B'] == args.B and r['runs'][0]['seed'] == 3201]
    c = {**configs[0], 'eval_frames': 1}
    base, encoder, phi, _ = build_encoder(c)
    ids, noise, signals = fixture(c, phi, args.K, 'iid')
    variance = noise_variance(args.B, args.ebn0)
    model = PolarModel(base.polar, base.positions, encoder.codebook.amplitudes().detach().numpy(), base.n)
    s = PolarScenario(model, signals[0]+np.sqrt(variance)*noise[0], args.K, variance)
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for c in configs:
        name, = c['decoders']
        params = {**dict(REGISTRY[name].defaults), **c['decoder_params'][name]}
        if REGISTRY[name].stochastic:
            params['seed'] = 12345
        before = time.perf_counter()
        counts, meta = run(name, s, **params)
        elapsed = time.perf_counter()-before
        recovered = sum(min(counts.get(int(w), 0), int(np.sum(ids[0] == w))) for w in np.unique(ids[0]))
        row = {'decoder': name, 'seconds': elapsed, 'pupe': 1-recovered/args.K, 'params': params,
               'meta': meta, 'naive_8192_frame_hours': elapsed*8192/3600}
        records.append(row)
        args.out.write_text(json.dumps({'B': args.B, 'K': args.K, 'ebn0': args.ebn0,
                                      'source_sha256': source_hashes(), 'records': records}, indent=2)+'\n')
        print(name, round(elapsed, 3), round(row['pupe'], 4), flush=True)


if __name__ == '__main__':
    main()
