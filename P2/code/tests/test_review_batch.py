from collections import Counter
import copy
import unittest

from counterfactual.audit_reviews import required_secondaries
from counterfactual.prepare_review_batch import select_batch


class ReviewBatchTests(unittest.TestCase):
    def setUp(self):
        strata=[('clean','none')]+[(t,s) for t in ('premature_release','object_motion_without_contact','spatial_contact_shift') for s in ('low','medium','high')]
        self.entries=[dict(sample_id=f'{t}-{s}-{i}',sha256=f'hash-{t}-{s}-{i}',type=t,severity=s,
                           source_sequence=f'source-{i}',object=f'object-{i%3}',frames=20)
                      for t,s in strata for i in range(8)]
        e=self.entries[0]
        self.records=[dict(sample_id=e['sample_id'],sample_sha256=e['sha256'],reviewer='fixture',round='primary',
                           decision='uncertain',reason='Synthetic fixture only')]

    def test_quota_excludes_all_previously_reviewed_decisions(self):
        selected=select_batch(self.entries,self.records,'batch')
        self.assertEqual(len(selected),24)
        self.assertEqual(len({e['sample_id'] for e in selected}),24)
        self.assertNotIn(self.records[0]['sample_id'],{e['sample_id'] for e in selected})
        counts=Counter((e['type'],e['severity']) for e in selected)
        self.assertEqual(counts.pop(('clean','none')),6)
        self.assertEqual(len(counts),9)
        self.assertEqual(set(counts.values()),{2})

    def test_reproducible_without_mutating_queue_or_votes(self):
        before=copy.deepcopy((self.entries,self.records))
        a=select_batch(self.entries,self.records,'batch')
        self.assertEqual(a,select_batch(list(reversed(self.entries)),self.records,'batch'))
        self.assertEqual((self.entries,self.records),before)

    def test_secondary_assignment_uses_full_queue(self):
        required=required_secondaries(self.entries)
        for e in select_batch(self.entries,self.records,'batch'):
            self.assertEqual(e['second_review_required'],e['sample_id'] in required)

    def test_insufficient_stratum_fails_without_silent_rebalancing(self):
        entries=[e for e in self.entries if e['type']!='clean']
        with self.assertRaises(ValueError):select_batch(entries,[],'batch')
