"""Strict matched comparison with audited Job-035/036 controls; preserve each run's source provenance."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import merge_runs, normalized_config, plots, source_hashes, write_json

JOB = Path(__file__).resolve().parent
CONTROLS = ['D2', 'ODMA', 'DEC-05', 'DEC-07', 'DEC-14', 'DEC-16', 'DEC-17', 'DEC-24']
GROUPS = {
    'discovery': ['D2', 'ODMA', 'DEC-14', 'DEC-25', 'DEC-26', 'DEC-34'],
    'independent_sampling': ['D2', 'ODMA', 'DEC-05', 'DEC-16', 'DEC-17', 'DEC-26', 'DEC-33'],
    'single_corrections': ['D2', 'ODMA', 'DEC-26', 'DEC-27', 'DEC-28', 'DEC-29', 'DEC-32'],
    'pair_corrections': ['D2', 'ODMA', 'DEC-07', 'DEC-24', 'DEC-26', 'DEC-30', 'DEC-31'],
}


def rows(path):
    return [r for line in path.read_text().splitlines() for r in json.loads(line)['runs']]


def validate_sources(contract):
    current = source_hashes()
    if current != contract['current_source_sha256']:
        raise ValueError('Current runtime differs from the audited Job037 snapshot; review changes before updating its contract')
    return [current, *[v['source_sha256'] for v in contract['historical_snapshots']]]


def checked_run(path, row, sources, parameters):
    if not (path/'summary.json').is_file():
        raise ValueError(f'Missing completed run: {path}. Select a complete --B/--seeds/--controls cohort explicitly.')
    saved = json.loads((path/'summary.json').read_text())
    if saved['status'] != 'complete' or saved['config'] != normalized_config(row):
        raise ValueError(f'Run status/config differs from its manifest: {path}')
    if saved['source_sha256'] not in sources:
        raise ValueError(f'Run source is outside the audited compatibility set: {path}')
    name, = row['decoders']
    if saved['decoder_params'] != {name: parameters[name]}:
        raise ValueError(f'Decoder parameters differ from the audited variant: {path}')
    return path


def grouped_plots(result, out):
    for label, names in GROUPS.items():
        selected = [name for name in names if name in result['config']['decoders']]
        if len(selected) < 2:
            continue
        subset = {**result, 'config': {**result['config'], 'decoders': selected},
                  'evaluation': [r for r in result['evaluation'] if r['decoder'] in selected], 'training': {}}
        folder = out/label
        folder.mkdir(parents=True, exist_ok=True)
        plots(subset, folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=JOB/'results_tractable')
    parser.add_argument('--baseline-results', type=Path, default=ROOT/'jobs/035_polar_decoders/results_all_decoders')
    parser.add_argument('--combination-results', type=Path, default=ROOT/'jobs/036_polar_combinations/results_combinations')
    parser.add_argument('--retry-results', type=Path, help='Explicit Job035 retry root; matching names take precedence')
    parser.add_argument('--controls', nargs='*', choices=CONTROLS, default=CONTROLS)
    parser.add_argument('--decoders', nargs='*', choices=[f'DEC-{i}' for i in range(25, 35)],
                        default=[f'DEC-{i}' for i in range(25, 35)])
    parser.add_argument('--B', nargs='+', type=int, choices=[12, 14], default=[12, 14])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[3201, 3202, 3203], default=[3201, 3202, 3203])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.decoders and not args.controls:
        parser.error('Select at least one decoder or control')
    contract = json.loads((JOB/'reference_contract.json').read_text())
    sources = validate_sources(contract)
    new = rows(JOB/'manifest.jsonl')
    old = rows(ROOT/'jobs/035_polar_decoders/manifest.jsonl')
    combinations = rows(ROOT/'jobs/036_polar_combinations/manifest.jsonl')
    for b in args.B:
        paths = []
        for seed in args.seeds:
            for decoder in args.decoders+args.controls:
                is_new = decoder in args.decoders
                bank = new if is_new else old if decoder in ('D2', 'ODMA', 'DEC-05', 'DEC-07') else combinations
                matches = [r for r in bank if (r['B'], r['seed'], r['mapping'], r['decoders']) ==
                           (b, seed, 'balanced', [decoder])]
                loads = [k for r in matches for k in r['loads']]
                if sorted(loads) != [7, 15, 22, 26]:
                    raise ValueError(f'Incomplete or duplicate manifest coverage: {(b, seed, decoder)}')
                for row in matches:
                    root = args.results if is_new else args.baseline_results if bank is old else args.combination_results
                    if bank is old and args.retry_results and (args.retry_results/row['name']).exists():
                        root = args.retry_results
                    paths.append(checked_run(root/row['name'], row, sources[:1] if is_new else sources,
                                             contract['decoder_params']))
        out = args.out/f'B{b}_balanced'
        merged = merge_runs(paths, out, compatible_sources=sources)
        grouped_plots(merged, out/'groups')
        write_json(out/'reuse_audit.json', {'contract': contract, 'controls': args.controls,
                                           'checks': 'complete manifests, pinned sources/parameters, paired matrices/observations, equal seed sets',
                                           'scope': 'exploratory seeds; not held-out validation'})
        print(out)


if __name__ == '__main__':
    main()
