"""Read-only preflight of frozen spatial candidates. Never emits training labels."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from counterfactual.diagnose_spatial import measure, sha


SEGMENTS = ('shoulder_to_elbow', 'elbow_to_wrist', 'wrist_to_hand')
SCHEMA = 'spatial_candidate_evidence_v1'


def text_value(value):
    return isinstance(value, str) and bool(value.strip())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_reference(ref, evidence_dir):
    """Hash verification establishes provenance, not truth of the cited claim."""
    if not isinstance(ref, dict) or not all(text_value(ref.get(k)) for k in ('path', 'sha256', 'locator')):
        return False
    path = Path(ref['path'])
    if not path.is_absolute():
        path = evidence_dir / path
    return path.is_file() and sha(path) == ref['sha256']


def preflight(entry, geometry, evidence, evidence_dir, held=False):
    missing, failed = [], []
    # Nothing in this result grants a label or bypasses the existing review/freeze gate.
    out = {'sample_id': entry['sample_id'], 'sample_sha256': entry['sha256'],
           'semantic_label': None, 'eligible_for_training': False,
           'semantic_truth_verified': False, 'held': held,
           'missing_evidence': missing, 'failed_checks': failed, 'geometry': geometry}
    if evidence is None:
        missing.extend(['provenanced_action_phase_window', 'action_conditioned_target_region',
                        'arm_tolerances_with_basis', 'pose_plausibility_evidence'])
    else:
        if not isinstance(evidence, dict) or evidence.get('schema') != SCHEMA:
            raise ValueError('Unsupported evidence schema')
        if evidence.get('sample_id') != entry['sample_id'] or evidence.get('sample_sha256') != entry['sha256']:
            raise ValueError('Evidence is bound to a different sample or stale hash')
        phase = evidence.get('phase') or {}
        if not isinstance(phase, dict):
            raise ValueError('phase must be an object')
        if not (text_value(phase.get('action')) and text_value(phase.get('name'))
                and phase['action'] != 'unknown' and phase['name'] != 'unknown'
                and verify_reference(phase.get('reference'), evidence_dir)):
            missing.append('provenanced_action_phase_window')
        start, end = phase.get('start_frame'), phase.get('end_frame')
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < entry['frames']:
            missing.append('valid_inclusive_phase_window')
        elif not start <= entry['parameters']['cut_frame'] <= end or end != entry['frames'] - 1:
            # v1 keeps the shifted trajectory through the final frame; a short focus
            # clip cannot certify the remainder. A future bounded generator can differ.
            failed.append('perturbation_extends_outside_evidenced_phase')

        region = evidence.get('target_region') or {}
        if not isinstance(region, dict):
            raise ValueError('target_region must be an object')
        if not (text_value(region.get('name')) and text_value(region.get('incompatibility_rationale'))
                and text_value(region.get('alternative_explanations_considered'))
                and verify_reference(region.get('reference'), evidence_dir)):
            missing.append('action_conditioned_target_region')
        if region.get('action') != phase.get('action') or region.get('phase') != phase.get('name'):
            failed.append('region_action_phase_mismatch')
        target = region.get('target_vertex')
        if type(target) is not int or target != entry['parameters']['target_vertex']:
            failed.append('target_vertex_mismatch')
        # A vertex id only means anything in the exact mesh file. This reference
        # preserves that declaration; it does not automatically map anatomical regions.
        if not verify_reference(region.get('mesh_reference'), evidence_dir):
            missing.append('target_mesh_reference')

        tolerance = evidence.get('arm_tolerances') or {}
        if not isinstance(tolerance, dict):
            raise ValueError('arm_tolerances must be an object')
        limits = tolerance.get('max_length_change_m') or {}
        if not isinstance(limits, dict):
            raise ValueError('max_length_change_m must be an object')
        if not verify_reference(tolerance.get('reference'), evidence_dir):
            missing.append('arm_tolerance_basis')
        for segment in SEGMENTS:
            limit = limits.get(segment)
            if not finite_number(limit) or limit < 0:
                missing.append('arm_tolerance:' + segment)
            elif geometry['arm_segment_max_length_change_m'][segment] > limit:
                failed.append('arm_length_change:' + segment)
        if not verify_reference(evidence.get('pose_plausibility_reference'), evidence_dir):
            missing.append('pose_plausibility_evidence')

    # Existing geometric cues only; no invented arm threshold or semantic criterion.
    if geometry['shifted_hand_near_fraction'] < .7:
        failed.append('shifted_hand_contact_fraction_below_0_7')
    if geometry['max_other_hand_displacement_m'] > 1e-6:
        failed.append('unshifted_hand_changed')
    if not geometry['object_pose_arrays_unchanged']:
        failed.append('object_pose_changed')
    if held:
        failed.append('existing_semantic_hold')
    out['state'] = ('failed_checks' if failed else 'missing_evidence' if missing
                    else 'preflight_pass_semantics_unverified')
    return out


def run(root, reviews, status_path, output, evidence_path=None):
    if output.exists():
        raise ValueError('Output already exists; preserve previous runs')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if sha(reviews) != status['source_reviews_sha256']:
        raise ValueError('Review/status hash mismatch')
    frozen = root / 'counterfactual/v1'
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    if sha(frozen / 'index.jsonl') != manifest['index_sha256']:
        raise ValueError('Frozen index hash mismatch')
    entries = [json.loads(line) for line in (frozen / 'index.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id = {e['sample_id']: e for e in entries}
    votes = [json.loads(line) for line in reviews.read_text(encoding='utf-8').splitlines()]
    for vote in votes:
        if vote['sample_id'] not in by_id or vote['sample_sha256'] != by_id[vote['sample_id']]['sha256']:
            raise ValueError('Review sample hash mismatch')
    evidence_by_id = {}
    if evidence_path:
        for row in json.loads(evidence_path.read_text(encoding='utf-8')):
            sample_id = row['sample_id']
            if sample_id in evidence_by_id or sample_id not in by_id or by_id[sample_id]['type'] != 'spatial_contact_shift':
                raise ValueError('Duplicate or out-of-scope evidence id')
            evidence_by_id[sample_id] = row
    results = []
    held = set(status.get('excluded_from_confirmed_negative_ids', []))
    for entry in entries:
        if entry['type'] != 'spatial_contact_shift':
            continue
        clean_id = entry['source_sequence'] + '__clean__none'
        candidate_path = frozen / 'samples' / (entry['sample_id'] + '.npz')
        clean_path = frozen / 'samples' / (clean_id + '.npz')
        if sha(candidate_path) != entry['sha256'] or sha(clean_path) != by_id[clean_id]['sha256']:
            raise ValueError('Frozen sample hash mismatch')
        with np.load(candidate_path, allow_pickle=False) as z, np.load(clean_path, allow_pickle=False) as c:
            geometry = measure(c['joints'], z['joints'], z['distances_m'], entry['parameters'])
            pose_keys = [k for k in c.files if k.startswith('obj')]
            geometry['object_pose_arrays_unchanged'] = bool(pose_keys) and all(
                k in z.files and np.array_equal(z[k], c[k]) for k in pose_keys)
        result = preflight(entry, geometry, evidence_by_id.get(entry['sample_id']),
                           evidence_path.parent if evidence_path else root, entry['sample_id'] in held)
        result['existing_reviews_reference_only'] = [v for v in votes if v['sample_id'] == entry['sample_id']]
        results.append(result)
    summary = {'schema': 'spatial_preflight_report_v1', 'sample_count': len(results),
               'states': dict(Counter(r['state'] for r in results)),
               'missing_evidence_counts': dict(Counter(k for r in results for k in r['missing_evidence'])),
               'failed_check_counts': dict(Counter(k for r in results for k in r['failed_checks'])),
               'supplied_evidence_records': len(evidence_by_id),
               'reviewed_samples_reference_only': sum(bool(r['existing_reviews_reference_only']) for r in results),
               'all_semantic_holds_preserved': sorted(held),
               'remaining_primary_budget_cap': status['budget_additional_primary_remaining_cap'],
               'new_human_tasks': 0, 'new_labels': 0, 'human_budget_consumed': 0,
               'training_started': False, 'remaining_budget_reallocated': False,
               'input_hashes': {'reviews': sha(reviews), 'review_status': sha(status_path),
                                'frozen_index': sha(frozen / 'index.jsonl'),
                                'frozen_manifest': sha(frozen / 'manifest.json'),
                                'preflight_code': sha(Path(__file__)),
                                'geometry_code': sha(root / 'counterfactual/diagnose_spatial.py'),
                                'evidence': sha(evidence_path) if evidence_path else None}}
    output.mkdir(parents=True)
    for name, data in [('summary.json', summary), ('candidates.json', results), ('preserved_review_status.json', status)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, help='Optional JSON array of hash-bound evidence records')
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1], args.reviews, args.status,
                         args.output, args.evidence), ensure_ascii=False, indent=2))
