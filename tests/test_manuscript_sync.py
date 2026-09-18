"""Execute public mathematical paths on fixtures and readbacks on released data.

Fixtures test code semantics, not trained-checkpoint scientific outcomes.
No training assets, network access, checkpoint loading, or Formal runs.
"""
from pathlib import Path
import csv
import importlib.util
import json
import sys
import types
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from future_response_dynamics import route_construction as route, operator as op, causal_validation as causal, path_integration as path, classifier
from future_response_dynamics.aggregation import aggregate_certified_rows
from generate_manuscript_tables import build
torch.set_num_threads(1)

def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module

def csv_rows(relative):
    with (ROOT / relative).open(encoding='utf-8-sig') as stream:
        data = list(csv.DictReader(stream))
    for r in data:
        if 'checkpoint' in r: r['checkpoint'] = int(r['checkpoint'])
    return data

class ManuscriptSyncTests(unittest.TestCase):
    def test_reaggregate_released_units(self):
        expected = json.loads((ROOT / 'provenance/RESULT.json').read_text())
        floor = json.loads((ROOT / 'provenance/classification_summary.json').read_text())['contrastability_floor']
        summary, counts = aggregate_certified_rows(csv_rows('results/ALIGNMENT_UNIT_OR_SUMMARY.csv'), csv_rows('results/CAUSAL_UNIT_LEVEL.csv'), csv_rows('results/FULL_DOSE_UNIT_LEVEL.csv'), [291402, 291403, 291404], floor)
        self.assertEqual(counts, expected['checkpoint_support_counts'])
        self.assertEqual(summary['overall'], expected['overall_classification'])
        for family in ('alignment_cells', 'causal_cells', 'dose_cells'):
            self.assertEqual(summary[family], expected['checkpoint_first_family_summaries'][family])

    def test_current_tables_rederive_from_released_results(self):
        tables, numerical = build(ROOT)
        for name, data in tables.items():
            with (ROOT / 'results/manuscript_tables' / (name + '.csv')).open(encoding='utf-8', newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [{k: str(v) for k, v in row.items()} for row in data])
        self.assertEqual(sum(r['Cell'] == 'Pass' for r in tables['table4_finite_dose']), 5)
        self.assertEqual(next(r['Cell'] for r in tables['table4_finite_dose'] if r['Checkpoint'] == 291404 and r['Stratum'] == 'S1'), 'Fail')
        self.assertLess(numerical['median_E_int32']['max'], .01)
        self.assertLess(numerical['median_I32_I16_relative']['max'], .005)

    def test_gru_operational_patch_does_not_read_native_cf(self):
        class NoOracle(dict):
            def __getitem__(self, key):
                if key in {'counterfactual_observations', 'native_cf_carrier', 'z_cf'}: raise AssertionError('Oracle read')
                return super().__getitem__(key)
        record = NoOracle(edit_object_id=0, edit_axis=1, counterfactual_values=[.3], factual_values=[.1], factual_state_t=np.arange(8))
        interface = {'U4': np.eye(192, 4), 'feature_mean': np.zeros(13), 'feature_scale': np.ones(13), 'affine_weights': np.arange(52).reshape(13,4)/100}
        expected = route.primitive_features([record]) @ interface['affine_weights'] @ interface['U4'].T
        np.testing.assert_allclose(route.patch_delta([record], interface), expected.astype(np.float32))

    def test_one_shot_patch_and_no_future_assimilation(self):
        torch.manual_seed(10)
        model = route.DeterministicBeliefModel().eval()
        records = [dict(actions=np.zeros((20,4)), factual_observations=np.zeros((20,2,6)), counterfactual_observations=np.ones((20,2,6)), edit_object_id=0, edit_axis=0, counterfactual_values=[.2], factual_values=[0], factual_state_t=np.zeros(8))]
        interface = {'U4': np.eye(192,4), 'feature_mean': np.zeros(13), 'feature_scale': np.ones(13), 'affine_weights': np.ones((13,4))*.1}
        routes, delta = route.construct_routes(model, records, interface, torch.device('cpu'))
        torch.testing.assert_close(routes['P'], routes['F'] + torch.tensor(delta))
        records[0]['counterfactual_observations'] = np.full((20,2,6), 100)
        altered, _ = route.construct_routes(model, records, interface, torch.device('cpu'))
        torch.testing.assert_close(altered['P'], routes['P'], rtol=0, atol=0)
        calls = []; original = model.step
        def watched(z, action, observation=None):
            self.assertIsNone(observation); calls.append(z.detach().clone()); return original(z, action, observation)
        model.step = watched
        z = routes['P'][0]; future = route.future_map(model, z, torch.zeros(8,4), torch.ones(8))
        self.assertEqual(len(calls), 8); self.assertEqual(future.shape, (8,8)); torch.testing.assert_close(calls[0], z)
        with self.assertRaisesRegex(RuntimeError, 'HORIZON'):
            route.future_map(model, z, torch.zeros(7,4), torch.ones(8))

    def test_lstm_architecture_complete_carrier_and_hidden_decoder(self):
        module = load('public_lstm_model', 'model/monolithic_lstm/lstm_model.py')
        torch.manual_seed(11); model = module.MonolithicLSTM().double().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 494664)
        self.assertEqual((model.recurrent.input_size, model.recurrent.hidden_size), (228,192))
        z = torch.randn(1,384,dtype=torch.float64)*.1; z2=z.clone();z2[:,192:]+=.2
        torch.testing.assert_close(model.decode(z),model.decode(z2),rtol=0,atol=0)
        a=torch.zeros(1,4,dtype=torch.float64)
        self.assertFalse(torch.equal(model.transition(z,a),model.transition(z2,a)))
        v=torch.zeros_like(z);v[:,192:]=1
        _, derivative=torch.func.jvp(lambda state:model.transition(state,a),(z,),(v,))
        self.assertGreater(float(derivative[:,:192].detach().norm()),0)
        seen=[]
        handle=model.observation_encoder.register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
        model.transition(z,a);handle.remove()
        expected=torch.zeros(1,12,dtype=torch.float64);expected[0,11]=1
        torch.testing.assert_close(seen[0],expected)
        with self.assertRaisesRegex(ValueError,'COMPLETE_H_AND_C'):
            model.decode(z[:,:192])

    def test_raw_factual_chain_and_single_restart_both_architectures(self):
        # Load standalone reference kernels without importing any Formal runner.
        config=types.ModuleType('runtime_config');config.C={};sys.modules['runtime_config']=config
        gru=load('public_gru_kernel','experiments/tangent_transport/gru/source/sham_kernel.py')
        package=types.ModuleType('src');package.__path__=[str(ROOT/'experiments/cross_architecture/lstm/src')];sys.modules['src']=package
        lstm=load('public_lstm_kernel','experiments/cross_architecture/lstm/src/kernel.py')
        for dim,rank,kernel in [(192,4,gru),(384,6,lstm)]:
            class Nonlinear:
                def transition(self,z,a): return z + .1*z*z + .001
                def decode(self,z): return z[:,:8]
            model=Nonlinear();h=torch.linspace(.01,.2,dim,dtype=torch.float64)[None,:]
            u=torch.eye(dim,rank,dtype=torch.float64)*2;patch=u.sum(-1)[None,:]*.01;actions=torch.zeros(1,12,4,dtype=torch.float64)
            hs,vs,ps=kernel.factual_transport(model,h,patch,u,actions)
            eh,ev,ep=h,u[None,:,:],patch
            for step in range(13):
                torch.testing.assert_close(hs[step],eh);torch.testing.assert_close(vs[step],ev);torch.testing.assert_close(ps[step],ep)
                factor=1+.2*eh;ev=factor[:,:,None]*ev;ep=factor*ep;eh=model.transition(eh,actions[:,0])
            self.assertGreater(float(vs[-1,:,0,0]),2)  # Raw columns were not re-normalized.
            inserted=hs[5]+ps[5];states,_=kernel.release(model,inserted,actions[:,5:]);expected=inserted
            for state in states:
                torch.testing.assert_close(state,expected);expected=model.transition(expected,actions[:,0])

    def test_full_future_derivative_native_cf_rank(self):
        class Linear:
            def transition(self,z,a): return .9*z
            def decode(self,z): return z[:8]
        model=Linear();z=torch.ones(192,dtype=torch.float64);actions=torch.zeros(8,4,dtype=torch.float64)
        matrix=op.response_operator(lambda state:route.future_map(model,state,actions,torch.ones(8,dtype=torch.float64)),z)
        expected=np.concatenate([(.9**k)*np.eye(8,192) for k in range(1,9)])
        np.testing.assert_allclose(matrix,expected,rtol=1e-12,atol=1e-12)
        cf=np.zeros((64,192));cf[0,0]=3;cf[1,1]=1
        self.assertEqual(op.r90_from_native_cf(cf),1)
        shifted=np.roll(cf,1,axis=1);result=op.compare_routes(shifted,cf,cf)
        self.assertEqual(result['r_CF'],1);self.assertGreater(result['Delta_align'],0)

    def test_52_directions_and_finite_perturbation_execution(self):
        rng=np.random.default_rng(71);r=rng.normal(size=(64,192));phi=rng.normal(size=(192,192));decoder=rng.normal(size=(8,192));u=np.eye(192,4)
        bank=causal.direction_bank(r,phi,decoder,u,291402,'fixture','P',4)
        second=causal.direction_bank(r,phi,decoder,u,291402,'fixture','P',4)
        self.assertEqual(len(bank),52)
        for one,two in zip(bank,second):
            np.testing.assert_array_equal(one[2],two[2]);self.assertAlmostEqual(np.linalg.norm(one[2]),1)
        # A linear future map gives independently known finite-difference effects.
        values=causal.evaluate_bank(lambda z:r@z,np.zeros(192),r,bank,.01)
        for row in values['rows']:self.assertAlmostEqual(row['predicted_gain'],row['actual_gain'],places=10)
        # Numerical null directions have roundoff-scale tied effects; compare ordering
        # on nonzero directions, while still checking all 52 effect magnitudes above.
        nonzero=[row for row in values['rows'] if row['predicted_gain']>1e-10]
        self.assertGreater(causal.spearman(np.array([r['predicted_gain'] for r in nonzero]),np.array([r['actual_gain'] for r in nonzero])),.999999)

    def test_transport_core_versus_complete_specificity_and_rank_scope(self):
        gru=[]
        for name in ('original','extension'):
            data=json.loads((ROOT/f'results/accepted_summaries/{name}.json').read_text())
            gru.extend((cp['checkpoint'],st,cell) for cp in data['checkpoints'] for st,cell in cp['strata'].items())
        self.assertEqual(len(gru),18)
        self.assertTrue(all(c['gates']['capture'] and c['gates']['functional'] for _,_,c in gru))
        self.assertEqual({(cp,st) for cp,st,c in gru if not c['supports']},{(291405,'S1'),(291413,'S2')})
        cfg=json.loads((ROOT/'configs/transport/P2_LSTM_PRIVILEGED_RANK6_TRANSPORT_CONFIRMATION_RC_V1/TRANSPORT_CORE_CONFIG.json').read_text())
        self.assertEqual((cfg['fixed_rank'],cfg['carrier_dimension']),(6,384))
        self.assertFalse(cfg['semantic_mapper']);self.assertFalse(cfg['semantic_shams_primary'])
        data=json.loads((ROOT/'results/accepted_summaries/lstm.json').read_text())
        self.assertEqual([c['checkpoint'] for c in data['checkpoints']],[392001,392002,392003])
        self.assertTrue(all(cell['supports'] for cp in data['checkpoints'] for cell in cp['strata'].values()))

    def test_capture_orthonormalizes_geometry_without_mutating_raw_columns(self):
        package=types.ModuleType('src');package.__path__=[str(ROOT/'experiments/cross_architecture/lstm/src')];sys.modules['src']=package
        config=load('src.config','experiments/cross_architecture/lstm/src/config.py');config.configure({'rank_rtol':1e-10,'eps':1e-24})
        capture=load('public_capture','experiments/cross_architecture/lstm/src/capture.py')
        basis=torch.zeros(1,384,6,dtype=torch.float64)
        basis[0,:6,:]=torch.diag(torch.arange(1,7,dtype=torch.float64))
        before=basis.clone();delta=torch.zeros(1,384,dtype=torch.float64);delta[0,0]=2
        np.testing.assert_allclose(capture.capture(basis,delta),[1])
        torch.testing.assert_close(basis,before,rtol=0,atol=0)

    def test_hidden_path_simpson_and_midpoint_only_gate(self):
        z=np.array([.2,.4]);delta=np.array([.3,-.1]);seen=[]
        def derivative(q):seen.append(q.copy());return np.diag(3*q*q)
        result=path.path_accounting(lambda q:q**3,derivative,z,delta)
        np.testing.assert_allclose(seen,z+np.arange(33)[:,None]/32*delta)
        self.assertLess(result['E_int32'],1e-12)
        with self.assertRaises(ValueError):path.composite_simpson(np.zeros((33,2)),10)
        cells=[]
        for seed in (1,2,3):
            for st in ('S1','S2'):cells.append(dict(checkpoint=seed,stratum=st,numerical_valid=True,median_E_local=.2,median_C_R_midpoint=.2))
        cells[-2]['median_C_R_midpoint']=.1
        self.assertEqual(classifier.classify_full_dose(cells),classifier.DOSE_PASS)
        cells[2]['median_C_R_midpoint']=.1
        self.assertEqual(classifier.classify_full_dose(cells),classifier.DOSE_FAIL)
        self.assertNotEqual(classifier.classify_overall(classifier.ALIGN_PASS,classifier.CAUSAL_PASS,classifier.DOSE_FAIL,other_node=999), 'FRESH_FUTURE_FUNCTIONAL_DYNAMICS_REPLICATION_PASS')

if __name__=='__main__': unittest.main(verbosity=2)
