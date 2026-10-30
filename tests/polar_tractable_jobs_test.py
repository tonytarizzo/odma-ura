"""Job037 resource, pairing and historical-source guards."""

import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmarks.polar_decoders import normalized_config, source_hashes

JOB = Path('jobs/037_polar_tractable')


class TractableJobsTests(unittest.TestCase):
    def test_manifest_exactly_covers_paired_research_grid(self):
        rows = runpy.run_path(str(JOB/'build_manifest.py'))['manifest']()
        self.assertEqual(rows, [json.loads(line) for line in (JOB/'manifest.jsonl').read_text().splitlines()])
        self.assertEqual(len(rows), 60)
        old = [r for line in Path('jobs/035_polar_decoders/manifest.jsonl').read_text().splitlines()
               for r in json.loads(line)['runs'] if r['decoders'] == ['D2'] and r['mapping'] == 'balanced']
        excluded = {'decoders', 'decoder_params', 'name'}
        seen = set(); frames = 0
        for bundle in rows:
            c, = bundle['runs']; name, = c['decoders']
            reference = next(r for r in old if (r['B'], r['seed']) == (c['B'], c['seed']))
            self.assertEqual({k: v for k, v in c.items() if k not in excluded},
                             {k: v for k, v in reference.items() if k not in excluded})
            self.assertNotIn((c['B'], c['seed'], name), seen)
            seen.add((c['B'], c['seed'], name))
            frames += c['eval_frames']*len(c['loads'])*len(c['eval_ebn0'])
            if 27 <= int(name[4:]) <= 32:
                self.assertEqual(c['decoder_params'][name]['init_method'], 'basis')
                self.assertEqual(c['decoder_params'][name]['init_width'], 32)
        self.assertEqual(frames, 491520)
        script = (JOB/'037_polar_tractable.sh').read_text()
        self.assertIn('#PBS -J 1-60\n', script)
        self.assertIn('ncpus=2:mem=16gb', script)

    def test_source_audit_rejects_unreviewed_runtime_changes(self):
        module = runpy.run_path(str(JOB/'merge.py'))
        contract = json.loads((JOB/'reference_contract.json').read_text())
        approved = module['validate_sources'](contract)
        self.assertEqual(approved[0], source_hashes())
        bad = copy.deepcopy(contract); bad['current_source_sha256']['src/metrics.py'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'audited Job037 snapshot'):
            module['validate_sources'](bad)
        old = json.loads(Path('jobs/036_polar_combinations/reference_contract.json').read_text())
        self.assertEqual(approved[1], old['source_sha256'])
        self.assertEqual(contract['decoder_params']['DEC-07']['guide_log_floor'], -1000.)

    def test_reuse_requires_complete_config_source_and_parameters(self):
        check = runpy.run_path(str(JOB/'merge.py'))['checked_run']
        contract = json.loads((JOB/'reference_contract.json').read_text())
        source = contract['historical_snapshots'][0]['source_sha256']
        row = normalized_config({'decoders': ['DEC-05']})
        saved = {'status': 'complete', 'config': row, 'source_sha256': source,
                 'decoder_params': {'DEC-05': contract['decoder_params']['DEC-05']}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaisesRegex(ValueError, 'Missing completed'):
                check(path, row, [source], contract['decoder_params'])
            for key, replacement, message in [('status', 'running', 'status/config'),
                                               ('source_sha256', {}, 'outside'),
                                               ('decoder_params', {'DEC-05': {}}, 'parameters')]:
                (path/'summary.json').write_text(json.dumps({**saved, key: replacement}))
                with self.assertRaisesRegex(ValueError, message):
                    check(path, row, [source], contract['decoder_params'])
            (path/'summary.json').write_text(json.dumps(saved))
            self.assertEqual(check(path, row, [source], contract['decoder_params']), path)

    def test_runner_skips_only_exact_complete_matches(self):
        execute = runpy.run_path(str(JOB/'run_row.py'))['run_bundle']
        row = runpy.run_path(str(JOB/'build_manifest.py'))['manifest']()[0]
        c, = row['runs']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/c['name']; path.mkdir()
            saved = {'status': 'complete', 'config': c, 'source_sha256': source_hashes()}
            (path/'summary.json').write_text(json.dumps(saved))
            run = Mock()
            with patch.dict(execute.__globals__, run_experiment=run):
                execute(row, folder)
                run.assert_not_called()
                saved['source_sha256'] = {}
                (path/'summary.json').write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, 'differs'):
                    execute(row, folder)


if __name__ == '__main__':
    unittest.main()
