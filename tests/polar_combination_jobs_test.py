"""Job-036 scheduling, pairing and explicit historical-control reuse contracts."""

import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.polar_decoders import merge_runs, normalized_config, run_experiment, source_hashes

JOB = Path('jobs/036_polar_combinations')


class CombinationJobTests(unittest.TestCase):
    def test_manifest_is_paired_with_035_and_has_no_duplicate_work(self):
        build = runpy.run_path(str(JOB/'build_manifest.py'))['manifest']
        rows = build()
        self.assertEqual(rows,[json.loads(line) for line in (JOB/'manifest.jsonl').read_text().splitlines()])
        self.assertEqual(len(rows),81)
        self.assertEqual(len({b['name'] for b in rows}),81)
        old = [r for line in Path('jobs/035_polar_decoders/manifest.jsonl').read_text().splitlines()
               for r in json.loads(line)['runs'] if r['decoders']==['D2'] and r['mapping']=='balanced']
        excluded = {'decoders','decoder_params','name','loads'}
        seen=set();frames=0
        for bundle in rows:
            for c in bundle['runs']:
                self.assertTrue(14<=int(c['decoders'][0][4:])<=24)
                reference=next(r for r in old if (r['B'],r['seed'])==(c['B'],c['seed']))
                self.assertEqual({k:v for k,v in c.items() if k not in excluded},
                                 {k:v for k,v in reference.items() if k not in excluded})
                for k in c['loads']:
                    key=(c['B'],c['seed'],c['decoders'][0],k)
                    self.assertNotIn(key,seen);seen.add(key)
                frames+=c['eval_frames']*len(c['loads'])*len(c['eval_ebn0'])
        self.assertEqual(len(seen),2*3*11*4)
        self.assertEqual(frames,540_672)
        script=(JOB/'036_polar_combinations.sh').read_text()
        self.assertIn('#PBS -J 1-81%4',script)
        self.assertIn('ncpus=2:mem=16gb',script)

    def test_reuse_requires_unchanged_shared_dependencies_and_definitions(self):
        module=runpy.run_path(str(JOB/'merge.py'));validate=module['validate_reference_sources']
        contract=json.loads((JOB/'reference_contract.json').read_text())
        self.assertEqual(validate(contract),source_hashes())
        bad=copy.deepcopy(contract)
        bad['source_sha256']['src/metrics.py']='bad'
        with self.assertRaisesRegex(ValueError,'Shared dependency'):validate(bad)
        bad=copy.deepcopy(contract)
        bad['shared_definition_sha256']['benchmarks/polar_decoders.py']['fixture']='bad'
        with self.assertRaisesRegex(ValueError,'Shared model'):validate(bad)
        bad=copy.deepcopy(contract);bad['new_modules']=[]
        with self.assertRaisesRegex(ValueError,'inventory'):validate(bad)

    def test_zero_moves_allowed_but_invalid_mixtures_rejected(self):
        c=normalized_config({'decoders':['DEC-17'],'decoder_params':{'DEC-17':{'moves':0}}})
        self.assertEqual(c['decoder_params']['DEC-17']['moves'],0)
        with self.assertRaises(ValueError):
            normalized_config({'decoders':['DEC-17'],'decoder_params':{'DEC-17':{'moves':-1}}})

    def test_split_new_runs_match_full_run_and_complete_children_are_skipped(self):
        execute=runpy.run_path(str(JOB/'run_row.py'))['run_bundle']
        config=normalized_config({'decoders':['DEC-17','DEC-22'],'loads':[1,2],'eval_frames':2,'eval_ebn0':[0.],
                                  'bounds':False,'threads':2,'decoder_params':{'DEC-17':{'samples':16},
                                  'DEC-22':{'particles':4,'batches':2,'moves':2}}})
        with tempfile.TemporaryDirectory() as folder, patch('benchmarks.polar_decoders.plots'), contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);run_experiment(config,root/'full');paths=[]
            for k in [1,2]:
                c={**config,'loads':[k],'name':f'K{k}'};row={'name':f'K{k}','runs':[c]}
                execute(row,root);paths.append(root/f'K{k}')
                before=(paths[-1]/'frames.jsonl').read_bytes();execute(row,root)
                self.assertEqual(before,(paths[-1]/'frames.jsonl').read_bytes())
            merge_runs(paths,root/'merged')
            full=[json.loads(line) for line in (root/'full/frames.jsonl').read_text().splitlines()]
            joined=[json.loads(line) for line in (root/'merged/frames.jsonl').read_text().splitlines()]
            for a,b in zip(full,joined):
                self.assertEqual(a['counts'],b['counts']);self.assertEqual(a['input_sha256'],b['input_sha256'])

    def test_cross_source_merge_is_opt_in_and_keeps_original_provenance(self):
        config={'decoders':['DEC-14'],'loads':[1],'eval_frames':1,'eval_ebn0':[0.],'bounds':False,'threads':2}
        with tempfile.TemporaryDirectory() as folder, patch('benchmarks.polar_decoders.plots'), contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder)
            a=run_experiment(config,root/'a')
            b=run_experiment({**config,'decoders':['REF']},root/'b')
            b['source_sha256']={'test_historical_snapshot':'test_digest'};b['revision']='test_old_revision'
            (root/'b/summary.json').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'Source hashes differ'):merge_runs([root/'a',root/'b'],root/'rejected')
            with self.assertRaisesRegex(ValueError,'compatibility set'):
                merge_runs([root/'a',root/'b'],root/'unapproved',compatible_sources=[a['source_sha256']])
            merged=merge_runs([root/'a',root/'b'],root/'merged',compatible_sources=[a['source_sha256'],b['source_sha256']])
            self.assertIsNone(merged['source_sha256']);self.assertIsNone(merged['revision'])
            self.assertEqual(merged['source_provenance_by_run'][1]['source_sha256'],b['source_sha256'])
            self.assertEqual(merged['source_provenance_by_run'][1]['revision'],'test_old_revision')
            self.assertEqual(json.loads((root/'b/summary.json').read_text())['source_sha256'],b['source_sha256'])


if __name__=='__main__':unittest.main()
