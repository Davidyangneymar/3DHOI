"""Synthetic review records in temporary directories; never add human annotations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from counterfactual.audit_reviews import audit
from counterfactual.critic_api import sha256
from counterfactual.reviewed_subset import select_reviewed_subset, training_eligibility, validate_parent_queue
from counterfactual.review_integrity import validate_reviewed_index


def fixtures():
    queue, votes = [], []
    for split in ('train','val','test'):
        for label in (0,1):
            sid=f'{split}_{label}'
            e=dict(sample_id=sid,source_sequence=sid,split=split,label=label,sha256='fixture',frames=10,
                   type='clean' if label==0 else 'premature_release',severity='none' if label==0 else 'low')
            queue.append(e)
            for reviewer,round_name in [('primary-person','primary'),('second-person','secondary')]:
                votes.append(dict(sample_id=sid,sample_sha256='fixture',reviewer=reviewer,round=round_name,
                     decision='valid' if label==0 else 'error',action='carry',phase='transport',
                     start_frame=1,end_frame=8,reason='Synthetic test fixture, not human evidence'))
    queue += [{**queue[-1],'sample_id':'pending','source_sequence':'pending'},
              {**queue[-1],'sample_id':'spatial','source_sequence':'spatial','type':'spatial_contact_shift'}]
    for r in votes[-2:]:votes.append({**r,'sample_id':'spatial'})
    return queue,votes


class SubsetTests(unittest.TestCase):
    def test_partial_freeze_does_not_approve_population_or_spatial(self):
        q,v=fixtures(); full=audit(q,v); s=select_reviewed_subset(q,full)
        self.assertFalse(full['ready_for_curated_freeze']);self.assertFalse(s['full_queue_ready'])
        self.assertTrue(s['ready_for_subset_freeze']);self.assertEqual(len(s['sample_ids']),6)
        self.assertEqual(s['diagnostic_only_ids'],['spatial'])
        self.assertTrue(s['training_eligibility']['training_ready'])

    def test_holds_uncertain_invalid_and_secondary_obligations_survive(self):
        q,v=fixtures();q[0]['second_review_required']=True
        v=[r for r in v if not (r['sample_id']==q[0]['sample_id'] and r['round']=='secondary')]
        for r in v:
            if r['sample_id']=='val_0':r['decision']='uncertain'
            if r['sample_id']=='test_0':r['decision']='invalid'
        full=audit(q,v,review_status={'excluded_from_confirmed_negative_ids':['train_1']})
        s=select_reviewed_subset(q,full)
        reasons={r['sample_id']:r['reason'] for r in s['exclusions']}
        self.assertEqual(reasons['train_0'],'pending_independent_secondary')
        self.assertEqual(reasons['train_1'],'semantic_hold')
        self.assertEqual(reasons['val_0'],'uncertain');self.assertEqual(reasons['test_0'],'invalid')

    def test_high_is_never_used_to_fill_validation_class(self):
        q,v=fixtures();next(e for e in q if e['sample_id']=='val_1')['severity']='high'
        s=select_reviewed_subset(q,audit(q,v));g=s['training_eligibility']
        self.assertFalse(g['training_ready']);self.assertIn('val_needs_both_classes_after_withholding_high',g['blockers'])
        self.assertEqual(g['coverage']['val']['usable_class_counts'],{'0':1})

    def test_parent_identity_and_review_binding_reject_mutations(self):
        q,v=fixtures();validate_parent_queue(q,copy.deepcopy(q));s=select_reviewed_subset(q,audit(q,v))
        entries=[{**e,'label':e['label_error'],'feature_version':'omomo_geom13_v1'} for e in s['eligible_samples']]
        validate_reviewed_index(entries,s)
        for field,value in [('split','other'),('label',True),('sha256','other')]:
            changed=copy.deepcopy(q);changed[0][field]=value
            with self.assertRaises(ValueError):validate_parent_queue(changed,q)
        for changed in (q[:-1],q+[q[0]]):
            with self.assertRaises(ValueError):validate_parent_queue(changed,q)
        for field,value in [('label_error',1),('split','test'),('sha256','changed')]:
            altered=copy.deepcopy(entries);altered[0][field]=value
            with self.assertRaises(ValueError):validate_reviewed_index(altered,s)

    def test_archive_roundtrip_and_training_gate(self):
        from counterfactual import freeze_reviewed,train_reviewed
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);parent=root/'counterfactual/v1';(parent/'samples').mkdir(parents=True)
            q,v=fixtures();next(e for e in q if e['sample_id']=='val_1')['severity']='high'
            for e in q:
                np.savez(parent/'samples'/(e['sample_id']+'.npz'),features=np.zeros((10,13),dtype=np.float32))
                e['sha256']=sha256(parent/'samples'/(e['sample_id']+'.npz'))
            by_id={e['sample_id']:e for e in q}
            for r in v:r['sample_sha256']=by_id[r['sample_id']]['sha256']
            (parent/'index.jsonl').write_text('\n'.join(json.dumps(e) for e in q))
            (parent/'manifest.json').write_text(json.dumps({'index_sha256':sha256(parent/'index.jsonl')}))
            queue=root/'queue.json';queue.write_text(json.dumps(q));reviews=root/'reviews.jsonl'
            reviews.write_text('\n'.join(json.dumps(r) for r in v),encoding='utf-8-sig')
            status=root/'status.json';status.write_text(json.dumps({'source_reviews_sha256':sha256(reviews),
                'excluded_from_confirmed_negative_ids':[],'held_sample_count':0}))
            output=root/'subset'
            argv=['freeze',str(queue),str(reviews),'--review-status',str(status),'--scope','reviewed-subset','--output',str(output)]
            with patch.object(freeze_reviewed,'OUT',parent),patch.object(sys,'argv',argv):freeze_reviewed.main()
            self.assertEqual((output/'review_records.jsonl').read_bytes(),reviews.read_bytes())
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertFalse(manifest['training_ready']);self.assertFalse(manifest['full_queue_ready'])
            training=root/'training'
            with patch('models.interaction_critic.ROOT',root),patch.object(sys,'argv',['train',str(output),'--output',str(training)]):
                with self.assertRaises(SystemExit):train_reviewed.main()
            self.assertFalse(training.exists())
            # Rehashing modified labels cannot detach the index from human records.
            entries=[json.loads(s) for s in (output/'index.jsonl').read_text().splitlines()]
            entries[0]['label_error']=1
            (output/'index.jsonl').write_text('\n'.join(json.dumps(e) for e in entries))
            manifest['index_sha256']=sha256(output/'index.jsonl');(output/'manifest.json').write_text(json.dumps(manifest))
            with patch('models.interaction_critic.ROOT',root),patch.object(sys,'argv',['train',str(output),'--output',str(training)]):
                with self.assertRaisesRegex(ValueError,'differs from reviewed'):train_reviewed.main()
            self.assertFalse(training.exists())


if __name__=='__main__':unittest.main()
