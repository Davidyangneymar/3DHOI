import unittest
from counterfactual.audit_reviews import required_secondaries
from counterfactual.prepare_review_followup import plan_packets
from counterfactual.protocol_checks import canonical_hash


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.entries=[dict(sample_id=f'fixture-{i}',sha256=f'hash-{i}',type='clean',frames=20) for i in range(10)]
        self.entries[0]['type']='spatial_contact_shift'
        self.records=[dict(sample_id=e['sample_id'],sample_sha256=e['sha256'],reviewer='fixture-owner',round='primary',
                           decision='invalid' if i<2 else 'uncertain',start_frame=1,end_frame=9,
                           action='carry',phase='transport',reason='Test fixture only',note='Original fixture note')
                      for i,e in enumerate(self.entries)]

    def test_primary_preserves_reference_without_generating_new_votes(self):
        _,selected,context=plan_packets(self.entries,self.records,'fixture')[0]
        self.assertEqual(len(selected),2)
        self.assertEqual(context['required_reviewer'],'fixture-owner')
        self.assertEqual(context['required_round'],'primary')
        self.assertEqual(context['previous_records']['fixture-0']['record_sha256'],canonical_hash(self.records[0]))
        self.assertNotIn('decision',context['previous_records']['fixture-0'])
        self.assertEqual(self.records[0]['decision'],'invalid')

    def test_secondary_uses_full_queue_assignment_and_excludes_prior_votes(self):
        packets=plan_packets(self.entries,self.records,'fixture')
        _,selected,context=packets[1]
        self.assertEqual({e['sample_id'] for e in selected},required_secondaries(self.entries))
        self.assertNotIn('previous_records',context)
        self.assertEqual(context['required_round'],'secondary')
        self.assertEqual(context['excluded_reviewers'],['fixture-owner'])
        self.assertNotEqual(context['storage_key'],packets[0][2]['storage_key'])

    def test_changed_hash_or_mixed_reviewers_rejected(self):
        self.records[0]['sample_sha256']='wrong'
        with self.assertRaises(ValueError):plan_packets(self.entries,self.records,'fixture')
        self.records[0]['sample_sha256']='hash-0';self.records[0]['reviewer']='another'
        with self.assertRaises(ValueError):plan_packets(self.entries,self.records,'fixture')
