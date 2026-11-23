"""Finite checks for paper-scale case generation, complete staged endpoints, and curve inference."""

from collections import Counter
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_curve as c
import paper_curve_analysis as a


class PaperCurveTest(unittest.TestCase):
    def test_published_configuration_and_paired_waveform(self):
        for k,bp,seed in ((50,12,3201),(100,13,923101)):
            m,_ = c.make_geometry(k,seed)
            block = c.baseline(k,seed)
            self.assertEqual((block.prefix_bits,m.p,m.b,m.code.crc_bits,m.code.list_size),(bp,1<<bp,100-bp,16,128))
            self.assertEqual(block.power_levels,(1.,))
            s,labels,h = c.make_frame(m,k,seed,50)
            s2,labels2,h2 = c.make_frame(m,k,seed,25)
            self.assertEqual(labels,labels2)
            self.assertEqual(h['standard_noise_sha256'],h2['standard_noise_sha256'])
            self.assertNotEqual(h['y_sha256'],h2['y_sha256'])
            self.assertEqual(c.verify_physical(s,block,labels,seed),h['y_sha256'])
            self.assertEqual(c.verify_physical(s2,block,labels2,seed),h2['y_sha256'])
            self.assertAlmostEqual(s.noise_var,1/(200*10**.05))
            self.assertTrue(np.allclose(np.square(m.amplitudes).sum(1),1.,rtol=0,atol=2e-14))
        self.assertEqual(sum(len(c.CURVE_SEEDS[k])*len(v) for k,v in c.CURVE_DB.items()),6656)
        self.assertEqual(sum(c.audit_cell(k,snrseed,snr) for k in c.CURVE_DB for snrseed in c.CURVE_SEEDS[k]
                             for snr in c.CURVE_DB[k]),12)

    def test_small_staged_ordinary_receiver(self):
        block = c.ODMAPolarBaseline(11,128,982063,prefix_bits=5,code_length=32,crc_bits=8,list_size=128,max_iterations=30)
        model = c.PolarModel(block.polar,block.positions,np.full(block.positions.shape,np.sqrt(1/32)),128)
        s,labels,_ = c.make_frame(model,2,982064,2800)
        actions,costs,stages = c.with_endpoints(s)
        self.assertEqual(actions['growth'],Counter(labels))
        m = c.verify_ordinary(s,block,actions,stages)
        self.assertTrue(m['passed'] and m['baseline_order_equal'] and m['baseline_calls_equal'])
        self.assertEqual(costs['rollback']['rollback_scl512'],32)
        self.assertLess(costs['native']['receiver_seconds'],costs['growth']['receiver_seconds'])

    def test_count_and_paper_list_distinction(self):
        self.assertEqual(c.score([4,4,1,7],{4:1,7:1,3:1}),dict(list_misses=1,count_misses=2,false_copies=1,
            foreign_ids=1,output_copies=3,list_size=3))
        with self.assertRaises(ValueError):
            c.score([7],{7:2})

    def test_grid_crossing_domains(self):
        db = [-.5,0,.5]
        self.assertEqual(a.crossing(db,[8,6,4],100),('interpolated',.25,.25,[0.,.5]))
        self.assertEqual(a.crossing(db,[8,5,3],100),('at_grid',0.,0.,[0.,0.]))
        self.assertEqual(a.crossing(db,[5,5,3],100),('at_grid',-.5,-.5,[-.5,-.5]))
        self.assertEqual(a.crossing(db,[6,4,7],100)[0],'ambiguous')
        self.assertEqual(a.crossing(db,[7,6,8],100)[0],'right_censored')
        self.assertEqual(a.crossing(db,[3,2,4],100)[0],'left_censored')
        self.assertEqual(a.interval([-np.inf]*10+[0.]*90,[0.]*90+[np.inf]*10),[None,None])
        self.assertEqual(a.interval(np.arange(100),np.arange(100)),[2.,97.])

    def test_censoring_preserved_in_paired_gain(self):
        cells = np.array([[8,7,6],[6,5,4],[4,3,2]])
        draws = np.stack([cells]*9+[np.full(cells.shape,7)])
        r = a.crossing_analysis(cells,draws,np.array([-.5,0.,.5]),100,1)
        result = r['required_energy_contrasts']['native_minus_growth']
        self.assertEqual(result['snr_db_saved_by_second'],.5)
        self.assertEqual(result['both_crossings_observed_resamples'],9)
        self.assertEqual(result['paired_bootstrap_95_db'],[None,None])
        json.dumps(r,allow_nan=False)

    def test_refuse_incomplete_or_undeclared_studies(self):
        with self.assertRaises(ValueError):
            c.summarize_curve([])
        with self.assertRaises(ValueError):
            c.run_curve_frame(50,923101,50)

    def test_complete_reduction_and_pairing_guards(self):
        def fixture(k,seed,v):
            labels = list(range(seed*k,(seed+1)*k))
            data = np.frombuffer(b''.join(w.to_bytes(13,'big') for w in labels),dtype=np.uint8).reshape(k,13)
            src = {key+'_sha256':c.digest(np.array([seed],dtype=np.int64)) for key in
                   ('positions','amplitudes','information','standard_noise')}
            src.update(bits_sha256=c.digest(np.unpackbits(data,axis=1)[:,4:]),y_sha256=c.digest(np.array([seed,v],dtype=np.int64)),
                       repeated_patterns=k-1,repeated_messages=0,variance=1/(200*10**(v/1000)))
            misses = {0:8,25:5,50:2}[v]
            endpoints = {}
            for i,name in enumerate(c.ENDPOINTS):
                words = labels[:k-max(0,misses-i)]
                endpoints[name] = dict(action=[[w,1] for w in words],metrics=c.score(labels,dict.fromkeys(words,1)),
                    receiver_seconds=1.+i,seconds_with_geometry=2.+i,native_scl128=k+5,rollback_scl512=32 if i else 0,
                    continuation_scl128=1 if i==2 else 0)
            accepted = [w for w,n in endpoints['native']['action']]
            stages = dict(native_order_round=[[w,0] for w in accepted],native_round_additions=[len(accepted)],native_calls=k+5,
                native_valid_ids=accepted,rollback_batches=[dict(patterns=list(range(8))) for _ in range(4)],
                base_bank=labels,bank=labels,cancellation_action=endpoints['growth']['action'],terminal_additions=[],
                logical_continuation=2,cache_hits=1)
            control = dict(passed=True,ordinary_native_calls=k+5,accepted_ids=accepted,baseline_accepted_ids=accepted,
                baseline_scl128_calls=k+5,baseline_order_equal=True,baseline_calls_equal=True,independent_y_sha256=src['y_sha256'])
            return dict(k=k,seed=seed,snr_cdb=v,study='paper_curve',labels=labels,source=src,endpoints=endpoints,stages=stages,
                        geometry_seconds=1.,source_seconds=.1,ordinary_control=control if c.audit_cell(k,seed,v) else None)
        seeds = {50:[21,22,23,24],100:[31,32,33,34]}
        grid = {50:[0,25,50],100:[0,25,50]}
        with patch.dict(c.CURVE_SEEDS,seeds,clear=True),patch.dict(c.CURVE_DB,grid,clear=True),patch.dict(c.CENTRAL_DB,{50:25,100:25}),\
                patch.object(a,'BOOTSTRAPS',20):
            rows = [fixture(k,s,db) for k in seeds for s in seeds[k] for db in grid[k]]
            report = c.summarize_curve(iter(rows))
            import tempfile
            from pathlib import Path
            from paper_outputs import write_tables_and_figure, write_crossings
            import paper_quadrature as quadrature
            for row in rows:
                base = row['endpoints']['native']
                ids = [word for word,n in base['action']]
                row['complex_screen'] = dict(ordered=ids,rounds=[len(ids)],calls=base['native_scl128'],
                    seconds_with_geometry=base['seconds_with_geometry'],metrics=base['metrics'],
                    standard_imaginary_sha256=row['source']['standard_noise_sha256'])
            with patch.object(quadrature,'BOOTSTRAPS',20), tempfile.TemporaryDirectory() as scratch:
                out = Path(scratch)
                write_tables_and_figure(report,out)
                write_crossings(report,quadrature.summarize(iter(rows)),out)
                self.assertEqual(len((out/'required_energy.csv').read_text().splitlines()),19)
                self.assertEqual(len((out/'energy_gain.csv').read_text().splitlines()),19)
            self.assertEqual(report['loads']['100']['by_snr']['25']['endpoints']['native']['count_pupe'],.05)
            self.assertEqual(report['loads']['100']['crossings']['list_pupe']['thresholds']['native']['snr_db'],.25)
            json.dumps(report,allow_nan=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                c.summarize_curve(rows+[rows[-1]])
            with self.assertRaisesRegex(ValueError,'missing 1'):
                c.summarize_curve(rows[:-1])
            rows[1]['source']['standard_noise_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError,'across SNR'):
                c.summarize_curve(rows)


if __name__=='__main__':
    unittest.main()
