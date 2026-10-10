import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from counterfactual.diagnose_spatial import sha
from counterfactual.spatial_candidate_preflight import SCHEMA, SEGMENTS, preflight, run


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        ref = self.root / 'evidence.txt'
        ref.write_text('Synthetic test fixture only; not scientific evidence.', encoding='utf-8')
        self.ref = {'path': 'evidence.txt', 'sha256': sha(ref), 'locator': 'line 1'}
        self.entry = {'sample_id': 'fixture__spatial_contact_shift__low', 'sha256': 'a' * 64,
                      'frames': 30, 'parameters': {'cut_frame': 10, 'ramp_frames': 4,
                                                  'target_vertex': 17, 'hand': 'left'}}
        self.geometry = {'arm_segment_max_length_change_m': dict.fromkeys(SEGMENTS, .01),
                         'shifted_hand_near_fraction': 1., 'other_hand_near_fraction': 1.,
                         'max_other_hand_displacement_m': 0., 'object_pose_arrays_unchanged': True}
        self.evidence = {'schema': SCHEMA, 'sample_id': self.entry['sample_id'],
                         'sample_sha256': self.entry['sha256'],
                         'phase': {'action': 'carry', 'name': 'transport', 'start_frame': 5,
                                   'end_frame': 29, 'reference': self.ref},
                         'target_region': {'name': 'fixture region', 'action': 'carry', 'phase': 'transport',
                                           'target_vertex': 17, 'incompatibility_rationale': 'Fixture rationale',
                                           'alternative_explanations_considered': 'Fixture alternatives',
                                           'reference': self.ref, 'mesh_reference': self.ref},
                         'arm_tolerances': {'max_length_change_m': dict.fromkeys(SEGMENTS, .02),
                                            'reference': self.ref},
                         'pose_plausibility_reference': self.ref}

    def check(self, evidence=None, held=False):
        return preflight(self.entry, self.geometry, evidence, self.root, held)

    def test_missing_evidence_does_not_infer_label_from_contact(self):
        result = self.check()
        self.assertEqual(result['state'], 'missing_evidence')
        self.assertEqual(len(result['missing_evidence']), 4)
        self.assertIsNone(result['semantic_label'])
        self.assertFalse(result['eligible_for_training'])

    def test_complete_declarations_still_do_not_prove_semantics_or_grant_training(self):
        result = self.check(self.evidence)
        self.assertEqual(result['state'], 'preflight_pass_semantics_unverified')
        self.assertFalse(result['semantic_truth_verified'])
        self.assertFalse(result['eligible_for_training'])
        self.assertIsNone(result['semantic_label'])

    def test_stale_sample_binding_fails_closed(self):
        for key in ('sample_sha256', 'sample_id', 'schema'):
            evidence = copy.deepcopy(self.evidence)
            evidence[key] = 'stale'
            with self.assertRaises(ValueError):
                self.check(evidence)

    def test_reference_tampering_is_not_accepted(self):
        (self.root / 'evidence.txt').write_text('changed', encoding='utf-8')
        result = self.check(self.evidence)
        self.assertEqual(result['state'], 'missing_evidence')
        self.assertIn('arm_tolerance_basis', result['missing_evidence'])
        self.assertIn('action_conditioned_target_region', result['missing_evidence'])

    def test_short_phase_cannot_cover_persistent_v1_shift(self):
        for start, end in ((5, 20), (11, 29)):
            self.evidence['phase'].update(start_frame=start, end_frame=end)
            self.assertIn('perturbation_extends_outside_evidenced_phase', self.check(self.evidence)['failed_checks'])

    def test_region_must_match_action_phase_and_target_vertex(self):
        self.evidence['target_region'].update(action='place', phase='supported', target_vertex=18)
        result = self.check(self.evidence)
        self.assertIn('region_action_phase_mismatch', result['failed_checks'])
        self.assertIn('target_vertex_mismatch', result['failed_checks'])

    def test_unshifted_contact_cannot_mask_shifted_contact_failure(self):
        self.geometry['shifted_hand_near_fraction'] = .1
        self.assertIn('shifted_hand_contact_fraction_below_0_7', self.check(self.evidence)['failed_checks'])

    def test_checks_every_segment_and_rejects_nonfinite_or_negative_tolerance(self):
        for segment in SEGMENTS:
            with self.subTest(segment=segment):
                self.geometry['arm_segment_max_length_change_m'][segment] = .03
                self.assertIn('arm_length_change:' + segment, self.check(self.evidence)['failed_checks'])
                self.geometry['arm_segment_max_length_change_m'][segment] = .01
                for bad in (None, float('nan'), float('inf'), -.01, True):
                    self.evidence['arm_tolerances']['max_length_change_m'][segment] = bad
                    self.assertIn('arm_tolerance:' + segment, self.check(self.evidence)['missing_evidence'])
                self.evidence['arm_tolerances']['max_length_change_m'][segment] = .02

    def test_existing_hold_cannot_be_overridden_by_preflight(self):
        self.assertIn('existing_semantic_hold', self.check(self.evidence, held=True)['failed_checks'])

    def test_detects_object_or_other_hand_changes(self):
        self.geometry.update(object_pose_arrays_unchanged=False, max_other_hand_displacement_m=.02)
        result = self.check(self.evidence)
        self.assertIn('object_pose_changed', result['failed_checks'])
        self.assertIn('unshifted_hand_changed', result['failed_checks'])

    def test_cli_run_preserves_inputs_status_and_refuses_stale_geometry_or_overwrite(self):
        frozen = self.root / 'counterfactual/v1'
        samples = frozen / 'samples'
        samples.mkdir(parents=True)
        entries = []
        for kind, severity in [('clean', 'none'), ('spatial_contact_shift', 'low')]:
            sample_id = f'fixture__{kind}__{severity}'
            path = samples / (sample_id + '.npz')
            np.savez(path, joints=np.zeros((30, 24, 3)), distances_m=np.zeros((30, 2)), obj_trans=np.zeros((30, 3)))
            entries.append(dict(self.entry, sample_id=sample_id, sha256=sha(path), source_sequence='fixture', type=kind))
        (frozen / 'index.jsonl').write_text('\n'.join(json.dumps(e) for e in entries), encoding='utf-8')
        (frozen / 'manifest.json').write_text(json.dumps({'index_sha256': sha(frozen / 'index.jsonl')}), encoding='utf-8')
        (self.root / 'counterfactual/diagnose_spatial.py').write_text('# fixture', encoding='utf-8')
        reviews = self.root / 'reviews.jsonl'
        reviews.write_text(json.dumps(dict(sample_id=entries[1]['sample_id'], sample_sha256=entries[1]['sha256'],
                                          round='primary', decision='uncertain')), encoding='utf-8')
        status = self.root / 'status.json'
        status.write_text(json.dumps({'source_reviews_sha256': sha(reviews),
                                      'excluded_from_confirmed_negative_ids': ['unrelated_existing_hold'],
                                      'budget_additional_primary_remaining_cap': 12}), encoding='utf-8')
        before = {p: sha(p) for p in self.root.rglob('*') if p.is_file()}
        output = self.root / 'report'
        result = run(self.root, reviews, status, output)
        self.assertEqual(result['sample_count'], 1)
        self.assertEqual(result['all_semantic_holds_preserved'], ['unrelated_existing_hold'])
        self.assertEqual(result['remaining_primary_budget_cap'], 12)
        self.assertEqual(result['new_labels'], 0)
        self.assertTrue(all(sha(p) == digest for p, digest in before.items()))
        with self.assertRaises(ValueError):
            run(self.root, reviews, status, output)
        (samples / (entries[1]['sample_id'] + '.npz')).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            run(self.root, reviews, status, self.root / 'rejected')
        self.assertFalse((self.root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
