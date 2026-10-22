"""Merge Job-036 combinations with unchanged, paired Job-035 controls and explicit source provenance."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.polar_decoders import merge_runs, normalized_config, source_hashes, write_json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_groups import plot_groups

CONTROLS = ['D2','ODMA','DEC-02','DEC-04','DEC-05','DEC-06','DEC-07']


def validate_reference_sources(contract):
    current = source_hashes()
    old = contract['source_sha256']
    changed_modules = set(contract['shared_definition_sha256'])
    if set(current)-set(old) != set(contract['new_modules']) or set(old)-set(current):
        raise ValueError('Source inventory differs from the audited Job035-to-036 change')
    for name, expected in old.items():
        if name not in changed_modules and current[name] != expected:
            raise ValueError(f'Shared dependency changed since Job035: {name}')
    for name, definitions in contract['shared_definition_sha256'].items():
        actual = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                  for node in ast.parse((ROOT/name).read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for function, expected in definitions.items():
            if actual.get(function) != expected:
                raise ValueError(f'Shared model/evaluation definition changed: {name}:{function}')
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.jsonl'))
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('results_combinations'))
    parser.add_argument('--baseline-results', type=Path, default=ROOT/'jobs/035_polar_decoders/results_all_decoders')
    parser.add_argument('--controls', nargs='*', choices=CONTROLS, default=CONTROLS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name('reference_contract.json').read_text())
    current = validate_reference_sources(contract)
    rows = [r for line in args.manifest.read_text().splitlines() for r in json.loads(line)['runs']]
    old_rows = [r for line in (ROOT/'jobs/035_polar_decoders/manifest.jsonl').read_text().splitlines()
                for r in json.loads(line)['runs']]
    panels = {}
    for row in rows:
        path = args.results/row['name']
        if not (path/'summary.json').is_file():raise ValueError(f'Missing new completed run: {path}')
        result = json.loads((path/'summary.json').read_text())
        if result['config'] != normalized_config(row) or result['source_sha256'] != current:
            raise ValueError(f'New run differs from its manifest/current source: {path}')
        panels.setdefault((row['B'],row['mapping']),[]).append(path)
    for (b,mapping), paths in panels.items():
        seeds = {r['seed'] for r in rows if (r['B'],r['mapping'])==(b,mapping)}
        for seed in sorted(seeds):
            for decoder in args.controls:
                matches = [r for r in old_rows if (r['B'],r['mapping'],r['seed'],r['decoders'])==(b,mapping,seed,[decoder])]
                if len(matches)!=1:raise ValueError(f'No unique Job035 reference for {(b,mapping,seed,decoder)}')
                row=matches[0];path=args.baseline_results/row['name']
                if not (path/'summary.json').is_file():
                    raise ValueError(f'Missing completed control: {path}. Wait for it or select available --controls explicitly.')
                result=json.loads((path/'summary.json').read_text())
                if result['config']!=normalized_config(row) or result['source_sha256']!=contract['source_sha256']:
                    raise ValueError(f'Control config/source differs from pinned Job035: {path}')
                if result['decoder_params']!={decoder:contract['control_params'][decoder]}:
                    raise ValueError(f'Control parameters differ from pinned Job035: {path}')
                paths.append(path)
        out=(args.out or args.results/'analysis')/f'B{b}_{mapping}'
        merged=merge_runs(paths,out,compatible_sources=[current,contract['source_sha256']])
        plot_groups(merged,out/'groups')
        write_json(out/'reuse_audit.json',{'reference_contract':contract,'new_source_sha256':current,
                                         'reused_controls':args.controls,
                                         'checks':'pinned old sources, unchanged shared definitions/dependencies, full configs, codebook hashes, paired observation hashes, equal seed sets',
                                         'scope':'same seeds as exploration; this is not held-out validation'})
        print(out)


if __name__=='__main__':main()
