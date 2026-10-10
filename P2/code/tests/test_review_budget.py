import copy
from collections import Counter
import unittest
from counterfactual.prepare_review_budget import focus,plan


class BudgetTests(unittest.TestCase):
    def test_focus_includes_boundary_and_clamps_context(self):
        e={'type':'premature_release','frames':120,'parameters':{'cut_frame':60,'ramp_frames':12}}
        self.assertEqual((focus(e)['start'],focus(e)['end']),(48,96))
        e['parameters']['cut_frame']=2
        self.assertEqual(focus(e)['start'],0)
        e['parameters']['cut_frame']=119
        self.assertEqual(focus(e)['end'],119)

    def test_clean_and_missing_metadata_use_whole_clip(self):
        for e in [{'type':'clean','frames':90,'parameters':{'cut_frame':40}}, {'type':'premature_release','frames':90}]:
            self.assertEqual(focus(e),{'start':0,'end':89,'method':'full_sequence','description':'无可用扰动边界，保留完整回放。'})

    def fixture(self):
        strata=[('clean','none')]+[(k,s) for k in ('premature_release','object_motion_without_contact','spatial_contact_shift') for s in ('low','medium','high')]
        es=[dict(sample_id=f'{k}-{s}-{i}',sha256=f'h-{k}-{s}-{i}',type=k,severity=s,frames=100,
                 source_sequence=f'source-{i}',object=f'object-{i%3}',parameters={'cut_frame':50,'ramp_frames':8})
            for k,s in strata for i in range(30)]
        rs=[dict(sample_id=e['sample_id'],sample_sha256=e['sha256'],reviewer='test',round='primary',decision='uncertain',reason='Fixture only') for e in es[:14]]
        reserved=es[14:20]
        for k,s in strata[1:]:reserved.extend([e for e in es if (e['type'],e['severity'])==(k,s)][:2])
        return es,rs,reserved

    def test_budget_keeps_issued24_and_adds_only10_without_relabeling(self):
        es,rs,reserved=self.fixture();before=copy.deepcopy((es,rs,reserved))
        selected=plan(es,rs,reserved)
        self.assertEqual(len(selected),34)
        self.assertEqual(len({e['sample_id'] for e in selected}),34)
        self.assertEqual([e['sample_id'] for e in selected[:24]],[e['sample_id'] for e in reserved])
        self.assertFalse({r['sample_id'] for r in rs}&{e['sample_id'] for e in selected})
        counts=Counter((e['type'],e['severity']) for e in selected)
        self.assertEqual(counts.pop(('clean','none')),7);self.assertEqual(set(counts.values()),{3})
        self.assertEqual((es,rs,reserved),before)
        self.assertEqual(selected,plan(list(reversed(es)),rs,reserved))

    def test_plan_refuses_changed_scope_or_previously_reviewed_reservations(self):
        es,rs,reserved=self.fixture()
        with self.assertRaises(ValueError):plan(es,rs,reserved[:-1])
        reserved[0]=es[0]
        with self.assertRaises(ValueError):plan(es,rs,reserved)
