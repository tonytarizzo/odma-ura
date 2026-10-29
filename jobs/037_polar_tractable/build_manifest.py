"""Job 037: procedural receivers on the paired Job-035/036 balanced-encoder fixtures."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import normalized_config


def parameters(number):
    if 27 <= number <= 32:
        return {'init_method': 'basis', 'init_width': 32, 'patterns': 8, 'list_size': 16,
                'max_refine': 32, 'samples': 64, 'burn_in': 64}
    result = {'width': 32, 'patterns': 8, 'list_size': 16, 'max_refine': 32}
    if number == 33:
        result['samples'] = 1024
    return result


def manifest(profile='research'):
    rows = []
    bits, seeds = ([12, 14], [3201, 3202, 3203]) if profile == 'research' else ([4], [3217])
    for b in bits:
        for seed in seeds:
            for number in range(25, 35):
                decoder = f'DEC-{number:02}'
                config = normalized_config({'profile': profile, 'B': b, 'prefix_bits': b-6 if b >= 12 else 2,
                                            'seed': seed, 'mapping': 'balanced', 'threads': 2, 'decoders': [decoder],
                                            'decoder_params': {decoder: parameters(number)}, 'eval_sampling': ['iid']})
                config['name'] = f'B{b}_n{config["n"]}_balanced_iid_{decoder}_s{seed}'
                rows.append({'name': config['name'], 'runs': [config]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['research', 'smoke'], default='research')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('manifest.jsonl'))
    args = parser.parse_args()
    rows = manifest(args.profile)
    args.out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    frames = sum(r['eval_frames']*len(r['loads'])*len(r['eval_ebn0']) for row in rows for r in row['runs'])
    print(f'{len(rows)} rows; {frames:,} decoder-frame evaluations; historical controls are reused.')


if __name__ == '__main__':
    main()
