"""Test pilot mechanics using explicit temporary fixtures, never real labels."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from counterfactual import real_error_pilot as p
from counterfactual.critic_api import sha256
from counterfactual.generator_adapter import mesh_paths
from counterfactual.calibrate import fit_calibration
from models.interaction_critic import ROOT


def draft():
    return dict(protocol_version=p.POLICY,status='draft',review_cap=20,data_domain='real_generator',
        use_for_calibration=False,unfiltered_by_critic=True,shortfall_reason='Synthetic unit test only',
        generator={'revision':'TEST FIXTURE','checkpoint_sha256':'a'*64,'training_manifest_sha256':None,'pretraining_exposure':None},
        slots=[dict(sample_id='fixture'+str(i),condition_id='fixture',source_group='fixture',action='carry',
                    object='vacuum',condition_text='TEST FIXTURE, not actual generated output',seed=100+i,candidate_index=i) for i in range(2)])


def annotation(row,identity,label=0):
    return dict(pilot_sha256=identity,sample_id=row['sample_id'],input_sha256=row['input_provenance']['input_sha256'],
        reviewer='test-only',round='primary',label_error=label,action='carry',phase='transport',
        start_frame=0,end_frame=1,error_type='none' if label==0 else 'uncertain',
        reason_codes=['phase_consistent'] if label==0 else ['insufficient_evidence'])


class PilotTests(unittest.TestCase):
    def test_locked_plan_binds_contact_and_mesh_parser_dependencies(self):
        # Modify copies only: no original source or model is changed or loaded
        # from the temporary dependency tree. This exercises actual file hashes.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            copied_files = [Path('counterfactual')/name for name in p.CODE_FILES]
            copied_files += [Path(name) for name in ('models/interaction_critic.py',
                                                    'omomo_contact.py', 'plot_omomo_alignment.py')]
            for relative in copied_files:
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, target)
            draft_path = root/'draft.json'; p.dump(draft_path, draft())
            with patch.object(p, '__file__', str(root/'counterfactual/real_error_pilot.py')):
                plan = p.lock_plan(draft_path, root/'locked.json')
                p.validate_locked(plan)
                for name in ('omomo_contact.py', 'plot_omomo_alignment.py'):
                    self.assertIn(name, plan['code_sha256'])
                    path = root/name; original = path.read_bytes()
                    path.write_bytes(original + b'\n# temporary regression fixture\n')
                    try:
                        with self.subTest(dependency=name), self.assertRaisesRegex(ValueError, 'scoring code changed'):
                            p.validate_locked(plan)
                    finally:
                        path.write_bytes(original)
                    p.validate_locked(plan)

    def test_budget_seeds_domain_and_generator_identity(self):
        p.validate_plan(draft())
        for key,value in [('review_cap',21),('data_domain','synthetic_v1'),('use_for_calibration',True),('unfiltered_by_critic',False)]:
            d=draft();d[key]=value
            with self.assertRaises(ValueError):p.validate_plan(d)
        for mutation in ('duplicate','one_seed','oversize','missing_identity'):
            d=draft()
            if mutation=='duplicate':d['slots'][1]['sample_id']=d['slots'][0]['sample_id']
            if mutation=='one_seed':d['slots'][1]['seed']=d['slots'][0]['seed']
            if mutation=='oversize':d['slots']*=11
            if mutation=='missing_identity':d['generator']['revision']=None
            with self.assertRaises(ValueError):p.validate_plan(d)

    def test_failures_retained_missing_refilled_scored_inputs_rejected(self):
        plan=draft();items=[{'sample_id':s['sample_id'],'status':'failed','failure_reason':'fixture failure'} for s in plan['slots']]
        rows,failed=p.delivery_rows(plan,items,Path('.'));self.assertEqual(len(failed),2);self.assertEqual(rows,[])
        with self.assertRaises(ValueError):p.delivery_rows(plan,items[:1],Path('.'))
        for changed in ({'p_error':.1},{'label_error':1},{'data_domain':'synthetic_v1'},{'seed':999}):
            good={**plan['slots'][0],'status':'generated','path':'fixture.npz','metadata':{'input_kind':'raw'},**changed}
            with self.assertRaises(ValueError):p.delivery_rows(plan,[good,items[1]],Path('.'))

    def test_label_integrity_and_uncertain_budget(self):
        rows=[{'sample_id':'x','frames':10,'input_provenance':{'input_sha256':'h'}}]
        r=annotation(rows[0],'pilot','uncertain');p.validate_annotations([r],rows,'pilot')
        for changes in ({'label_error':True},{'label_error':0.0},{'start_frame':True},{'input_sha256':'bad'},
                        {'pilot_sha256':'other'},{'round':'secondary'},{'error_type':'none'},
                        {'label_error':1,'error_type':'unsupported_motion','legal_alternative':'unknown'}):
            with self.assertRaises(ValueError):p.validate_annotations([{**r,**changes}],rows,'pilot')
        with self.assertRaises(ValueError):p.validate_annotations([r,r],rows,'pilot')
        with self.assertRaises(ValueError):p.validate_annotations([],rows,'pilot')

    def test_motion_focus_stays_in_original_frames(self):
        sample={'joints':np.zeros((61,24,3))};sample['joints'][30:,:,0]=1
        focus=p.focus_range(sample,10)
        self.assertEqual((focus['start'],focus['end']),(20,40))

    def test_semantic_reasons_cannot_contradict_conclusion(self):
        row={'sample_id':'x','frames':10,'input_provenance':{'input_sha256':'h'}}
        r=annotation(row,'pilot',0)
        with self.assertRaises(ValueError):p.validate_annotations([{**r,'reason_codes':['required_contact_broken']}],[row],'pilot')
        r.update(label_error=1,error_type='incompatible_contact',reason_codes=['action_incompatible_region'],legal_alternative='excluded')
        with self.assertRaises(ValueError):p.validate_annotations([r],[row],'pilot')
        r['note']='Test-only evidence text';p.validate_annotations([r],[row],'pilot')

    def test_raw_prepare_blind_review_evaluate_and_tamper(self):
        entries=[json.loads(s) for s in (ROOT/'counterfactual/v1/index.jsonl').read_text().splitlines()]
        anchor=next(e for e in entries if e['type']=='clean' and e['object']=='vacuum')
        variant=next(e for e in entries if e['source_sequence']==anchor['source_sequence'] and e['type']=='premature_release')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);d=root/'draft.json';p.dump(d,draft());planpath=root/'plan.json'
            plan=p.lock_plan(d,planpath)
            with self.assertRaises(ValueError):p.lock_plan(d,planpath)
            wrong=copy.deepcopy(plan);wrong['threshold_error']=.123
            with self.assertRaises(ValueError):p.validate_locked(wrong)
            delivery=[]
            meta={'input_kind':'raw','object':'vacuum','units':'metres','coordinate_system':'right_handed_z_up',
                  'joint_schema':'omomo24','rotation_convention':'local_to_world_column','mesh_frame':'omomo_original_local',
                  'fps':30,'accept_inferred_reference_fps':True,'mesh_sha256':{x.name:sha256(x) for x in mesh_paths('vacuum')}}
            for i,e in enumerate((anchor,variant)):
                delivery.append({**plan['slots'][i],'path':str(ROOT/'counterfactual/v1/samples'/(e['sample_id']+'.npz')),
                                 'metadata':meta,'status':'generated'})
            deliverypath=root/'delivery.jsonl';deliverypath.write_text('\n'.join(json.dumps(r) for r in delivery))
            pilot=root/'pilot';receipt=p.prepare(planpath,deliverypath,pilot)
            self.assertEqual(receipt['prepared_count'],2)
            ui=(pilot/'queue.js').read_text()
            for forbidden in ('p_error','p_valid','rule_score','label_error','threshold_error'):
                self.assertNotIn(forbidden,ui)
            rows=p.read_lines(pilot/'manifest.jsonl')
            records=[annotation(rows[0],receipt['pilot_sha256'],0),annotation(rows[1],receipt['pilot_sha256'],'uncertain')]
            ann=root/'annotations.jsonl';ann.write_text('\n'.join(json.dumps(r) for r in records))
            result=p.evaluate(pilot,ann,root/'result')
            self.assertEqual(result['experiment_status'],'executed_insufficient_label_classes')
            self.assertEqual(result['uncertain_count'],1)
            self.assertIsNone(result['metrics']['diagnostic']['critic']['roc_auc'])
            self.assertIsNone(result['metrics']['diagnostic']['critic']['balanced_accuracy'])
            self.assertFalse(result['independent_review_verified'])
            predictions=p.read_lines(root/'result/predictions.jsonl')
            with self.assertRaises(ValueError):fit_calibration(predictions)
            with self.assertRaisesRegex(ValueError,'explicitly forbids calibration'):
                fit_calibration([{**r,'split':'dev'} for r in predictions])
            with self.assertRaises(ValueError):p.evaluate(pilot,ann,root/'result')
            (pilot/'manifest.jsonl').write_text('modified')
            with self.assertRaisesRegex(ValueError,'Prepared pilot changed'):p.evaluate(pilot,ann,root/'bad')

    def test_all_generation_failures_can_be_reported_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p.dump(root/'draft.json',draft());plan=p.lock_plan(root/'draft.json',root/'plan.json')
            items=[{'sample_id':s['sample_id'],'status':'failed','failure_reason':'Test fixture'} for s in plan['slots']]
            (root/'delivery.jsonl').write_text('\n'.join(json.dumps(r) for r in items))
            p.prepare(root/'plan.json',root/'delivery.jsonl',root/'pilot')
            (root/'empty.jsonl').write_text('')
            result=p.evaluate(root/'pilot',root/'empty.jsonl',root/'result')
            self.assertEqual(result['generation_failures'],2);self.assertEqual(result['primary_review_count'],0)
            self.assertEqual(result['experiment_status'],'executed_insufficient_label_classes')


if __name__=='__main__':unittest.main()
