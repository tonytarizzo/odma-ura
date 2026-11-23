import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import paper_curve as curve
import paper_terminal as terminal
from framework.polar_count_cavity import action_hash
from framework.polar_terminal_query import run


class TerminalTest(unittest.TestCase):
    def test_normal_entry_agrees_on_a_paid_and_an_inactive_query(self):
        block = curve.ODMAPolarBaseline(15,512,40219,prefix_bits=5,code_length=32,crc_bits=16,list_size=128)
        model = curve.PolarModel(block.polar,block.positions,np.full(block.positions.shape,np.sqrt(1/32)),block.n)
        for db,trigger in ((-2000,True),(2500,False)):
            s,labels,source = curve.make_frame(model,1,41001,db)
            action,meta = run(s,selector='magnitude')
            with patch.dict(curve.CURVE_SEEDS,{1:[41001]}),patch.dict(curve.CURVE_DB,{1:[db]}),\
                    patch.object(curve,'make_geometry',return_value=(model,.2)),patch.object(curve,'audit_cell',return_value=False):
                row = curve.run_curve_frame(1,41001,db)
            t = row['terminal_magnitude']
            self.assertEqual(t['action'],sorted(action.items()))
            self.assertEqual(t['metrics'],curve.score(labels,action))
            self.assertEqual(t['y_sha256'],source['y_sha256'])
            self.assertEqual(t['triggered'],trigger)
            self.assertEqual(t['terminal_scl4096'],int(trigger))
            if trigger:
                tr = meta['terminal_query']
                for key in ('pattern','query_ids','list_metrics','new_ids','cancellation'):
                    self.assertEqual(t['trace'][key],tr[key])
                self.assertTrue(tr['query_ids'])
                from framework.polar_full_pattern import CountBank
                fitted = dict(row['stages']['cancellation_action'])
                bank = CountBank(s,row['stages']['bank'])
                residual = s.y-bank.matrix @ bank.vector(fitted)
                p = tr['pattern']
                self.assertEqual(p,np.argmax(np.abs(residual[model.positions]).sum(1)))
                llr = 2*model.amplitudes[p]*residual[model.positions[p]]/s.noise_var
                self.assertEqual(t['llr_sha256'],curve.digest(llr))
            else:
                self.assertIsNone(t['trace'])
                self.assertEqual(t['action'],row['endpoints']['growth']['action'])

    def test_reduction_and_persisted_audit(self):
        seeds,grid = {50:[41,42],100:[43,44]},{50:[0,25,50],100:[0,25,50]}
        rows = []
        for k,panel in seeds.items():
            for seed in panel:
                for db in grid[k]:
                    labels = list(range(k))
                    missed = k//5 if db==0 else (k//12 if db==25 else 1)
                    base_action = sorted(dict.fromkeys(labels[missed:],1).items())
                    fitted = base_action
                    stages = dict(cancellation_action=fitted,bank=sorted({w for w,c in fitted+base_action}))
                    base = dict(action=base_action,metrics=curve.score(labels,dict(base_action)),receiver_seconds=3.,
                                native_scl128=60,rollback_scl512=32,continuation_scl128=70)
                    out = dict(base_action)
                    active = db==50
                    trace,ctx = None,None
                    if active:
                        out[0]=1
                        ctx = dict(context='raw',restored=[],restored_count=0,outside_count=1,pending=1,variance=.003)
                        trace = dict(pattern=0,query_ids=[0],list_metrics=[.5],new_ids=[0],signed_prefix_end=None,
                                     model_top_probability=None,fit={'converged':True},cancellation=sorted(out.items()),
                                     completion={'additions':[]})
                    yhash = curve.digest(np.array([seed,db]))
                    t = dict(**{key:base[key] for key in ('native_scl128','rollback_scl512','continuation_scl128')},
                             action=sorted(out.items()),triggered=active,metrics=curve.score(labels,out),trace=trace,
                             y_sha256=yhash,llr_sha256='a'*64 if active else None,query_context=ctx,
                             prefix_cancellation_sha256=action_hash(dict(fitted)),terminal_scl4096=int(active),
                             receiver_seconds=3.25,incremental_seconds=.25,seconds_with_geometry=3.75)
                    rows.append(dict(k=k,seed=seed,snr_cdb=db,labels=labels,source=dict(y_sha256=yhash,variance=.003),
                                     stages=stages,geometry_seconds=.5,endpoints={'growth':base},terminal_magnitude=t))
        with patch.dict(curve.CURVE_DB,grid,clear=True),patch.dict(curve.CURVE_SEEDS,seeds,clear=True),patch.object(terminal,'BOOTSTRAPS',100):
            result = terminal.summarize_terminal(json.loads(json.dumps(rows)))
            self.assertEqual(result['100']['by_snr']['50']['paired']['count_pupe']['saved'],2)
            self.assertEqual(result['100']['by_snr']['50']['terminal_scl4096'],2)
            self.assertGreater(result['100']['crossings']['list_pupe']['snr_db_saved'],0.)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                terminal.write_terminal_tables(result,root)
                for name,count in [('terminal_energy.csv',12),('terminal_curve.csv',18),('terminal_frames.csv',12)]:
                    with (root/name).open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))),count)
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                terminal.summarize_terminal(rows[1:])
            bad = copy.deepcopy(rows[-1]); bad['terminal_magnitude']['trace']['new_ids'] = []
            with self.assertRaisesRegex(ValueError,'list or input'):
                terminal.verify(bad)
            bad = copy.deepcopy(rows[0]); bad['terminal_magnitude']['seconds_with_geometry'] -= .1
            with self.assertRaisesRegex(ValueError,'work'):
                terminal.verify(bad)

    def test_censored_paired_energy_keeps_unbounded_replicates(self):
        total = np.array([[10,7],[6,4],[3,1]])
        boot = np.stack([total]*8+[np.full(total.shape,8)]*2)
        result = terminal.target([0,.5,1],total,boot,100)
        self.assertEqual(result['paired_bootstrap_95_db'],[None,None])
        self.assertEqual(result['both_crossings_observed_resamples'],8)
        self.assertEqual(result['thresholds']['growth']['bootstrap_status']['right_censored'],2)
        json.dumps(result,allow_nan=False)


if __name__=='__main__':
    unittest.main()
