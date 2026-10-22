"""Job 036: only new combinations; reuse paired Job-035 controls without retraining."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import normalized_config


def manifest(profile='research'):
    rows = []
    bits, seeds = ([12,14], [3201,3202,3203]) if profile == 'research' else ([4], [3217])
    for b in bits:
        for seed in seeds:
            for number in range(14,25):
                name = f'DEC-{number:02}'
                config = normalized_config({'profile': profile, 'B': b, 'prefix_bits': b-6 if b>=12 else 2,
                                            'seed': seed, 'mapping': 'balanced', 'threads': 2, 'decoders': [name],
                                            'eval_sampling': ['iid']})
                shards = [[7,15],[22,26]] if profile=='research' and b==14 and 15<=number<=19 else [config['loads']]
                for loads in shards:
                    child = {**config, 'loads': loads}
                    suffix = '_K'+'-'.join(map(str,loads)) if len(shards)>1 else ''
                    child['name'] = f'B{b}_n{config["n"]}_balanced_iid_{name}_s{seed}{suffix}'
                    rows.append({'name': child['name'], 'runs': [child]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['research','smoke'], default='research')
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('manifest.jsonl'))
    args = parser.parse_args()
    rows = manifest(args.profile)
    args.out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    frames = sum(r['eval_frames']*len(r['loads'])*len(r['eval_ebn0']) for row in rows for r in row['runs'])
    print(f'{len(rows)} jobs; {frames:,} decoder-frame evaluations. D2/ODMA and old controls are reused, not queued.')


if __name__ == '__main__':
    main()
