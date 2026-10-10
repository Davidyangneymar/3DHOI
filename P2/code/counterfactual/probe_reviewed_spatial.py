"""Probe only already-reviewed spatial cases; save diagnostics, never new NPZ/labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from counterfactual.bounded_arm_probe import probe
from counterfactual.diagnose_spatial import sha
from counterfactual.protocol_checks import canonical_hash
from omomo_contact import extract_distances
from plot_omomo_alignment import DATA_ROOT


def run(root, reviews_path, status_path, output):
    if output.exists():
        raise ValueError('Output exists; previous runs must be preserved')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if sha(reviews_path) != status['source_reviews_sha256']:
        raise ValueError('Review/status hash mismatch')
    frozen = root / 'counterfactual/v1'
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    if sha(frozen / 'index.jsonl') != manifest['index_sha256']:
        raise ValueError('Frozen index hash mismatch')
    entries = {e['sample_id']: e for e in map(json.loads, (frozen / 'index.jsonl').read_text(encoding='utf-8').splitlines())}
    votes = [json.loads(line) for line in reviews_path.read_text(encoding='utf-8').splitlines()]
    for vote in votes:
        if vote['sample_id'] not in entries or entries[vote['sample_id']]['sha256'] != vote['sample_sha256']:
            raise ValueError('Review bound to unknown or stale sample')
    ids = sorted({v['sample_id'] for v in votes if v['round'] == 'primary'
                  and entries[v['sample_id']]['type'] == 'spatial_contact_shift'})
    results, mesh_hashes = [], {}
    for sample_id in ids:
        entry = entries[sample_id]
        records = [v for v in votes if v['sample_id'] == sample_id]
        primaries = [v for v in records if v['round'] == 'primary']
        secondaries = [v for v in records if v['round'] == 'secondary']
        if len(primaries) != 1 or not secondaries:
            raise ValueError('Probe requires one primary and existing secondary record')
        if any(v['reviewer'] == primaries[0]['reviewer'] for v in secondaries):
            raise ValueError('Secondary reviewer ID must differ from primary')
        clean_id = entry['source_sequence'] + '__clean__none'
        original_path = frozen / 'samples' / (clean_id + '.npz')
        candidate_path = frozen / 'samples' / (sample_id + '.npz')
        if sha(original_path) != entries[clean_id]['sha256'] or sha(candidate_path) != entry['sha256']:
            raise ValueError('Frozen geometry hash mismatch')
        with np.load(original_path, allow_pickle=False) as c, np.load(candidate_path, allow_pickle=False) as z:
            original, old = c['joints'].copy(), z['joints'].copy()
            window = {key: c[key].copy() for key in c.files if key.startswith('obj')}
        result = {'sample_id': sample_id, 'sample_sha256': entry['sha256'],
                  'clean_sha256': entries[clean_id]['sha256'],
                  'existing_review_records': records,
                  'record_hashes': [canonical_hash(v) for v in records],
                  'annotation_window_is_not_verified_single_phase': True,
                  'target_source': 'existing_frozen_v1_hand_trajectory; no new semantic region selected',
                  'semantic_label': None, 'eligible_for_training': False}
        if len({(v['action'], v['phase']) for v in records}) != 1:
            result['diagnostic'] = {'state': 'skipped_action_phase_disagreement'}
            results.append(result)
            continue
        starts, ends = [v['start_frame'] for v in records], [v['end_frame'] for v in records]
        if any(type(v) is not int for v in starts + ends) or any(
                not 0 <= a <= b < len(original) for a, b in zip(starts, ends)):
            raise ValueError('Invalid review evidence window')
        # Intersection is merely a conservative experimental support window. It
        # does not elevate an uncertain review into a verified phase annotation.
        start = max(*starts, entry['parameters']['cut_frame'])
        end = min(ends)
        ramp = entry['parameters']['ramp_frames']
        result['experimental_window'] = {'start_frame': start, 'end_frame': end, 'ramp_frames': ramp}
        if end - start < 2 * ramp:
            result['diagnostic'] = {'state': 'skipped_window_too_short'}
            results.append(result)
            continue
        changed, diagnostic = probe(original, old, entry['parameters']['hand'], start, end, ramp)
        result['diagnostic'] = diagnostic
        hand = 0 if entry['parameters']['hand'] == 'left' else 1
        arm_ids = np.array([16, 18, 20, 22]) + hand
        result['frozen_v1_max_segment_change_m'] = float(np.abs(
            np.linalg.norm(np.diff(old[:, arm_ids], axis=1), axis=-1)
            - np.linalg.norm(np.diff(original[:, arm_ids], axis=1), axis=-1)).max())
        if changed is not None:
            name = entry['object']
            names = ([f'{name}_cleaned_simplified_top.obj', f'{name}_cleaned_simplified_bottom.obj']
                     if name in ('mop', 'vacuum') else [f'{name}_cleaned_simplified.obj'])
            for filename in names:
                path = DATA_ROOT / 'captured_objects' / filename
                mesh_hashes[str(path)] = sha(path)
            distances, _, _ = extract_distances(window, changed, name)
            plateau = distances[start + ramp:end - ramp + 1]
            diagnostic['plateau_frames'] = len(plateau)
            diagnostic['plateau_shifted_hand_near_fraction'] = float((plateau[:, hand] < .05).mean())
            diagnostic['plateau_other_hand_near_fraction'] = float((plateau[:, 1 - hand] < .05).mean())
            diagnostic['distance_threshold_m_geometric_only'] = .05
        results.append(result)
    hashes = {'reviews': sha(reviews_path), 'status': sha(status_path),
              'frozen_manifest': sha(frozen / 'manifest.json'), 'frozen_index': sha(frozen / 'index.jsonl')}
    for path in ('counterfactual/bounded_arm_probe.py', 'counterfactual/probe_reviewed_spatial.py',
                 'counterfactual/protocol_checks.py', 'omomo_contact.py', 'plot_omomo_alignment.py'):
        hashes[path] = sha(root / path)
    summary = {'date_local': '2026-10-04', 'scope': 'already-reviewed spatial cases, diagnostic only',
               'reviewed_cases': len(results),
               'states': dict(Counter(r['diagnostic']['state'] for r in results)),
               'failure_reasons_by_frame': dict(Counter(f['reason'] for r in results
                                                       for f in r['diagnostic'].get('failed_frames', []))),
               'existing_primary_decisions_unchanged': dict(Counter(
                   next(v['decision'] for v in r['existing_review_records'] if v['round'] == 'primary') for r in results)),
               'new_labels': 0, 'new_dataset_samples': 0, 'new_human_tasks': 0,
               'human_budget_consumed': 0, 'remaining_primary_budget_cap': status['budget_additional_primary_remaining_cap'],
               'training_started': False, 'gpu_used': False, 'input_hashes': hashes,
               'mesh_hashes': mesh_hashes,
               'limitations': ['no joint-angle limits or rotation reconstruction',
                               'no collision or penetration test', 'no semantic target-region certification',
                               'joint-position solutions do not establish motion quality',
                               'infeasible frames reject entire probe; no clamping or partial success export']}
    output.mkdir(parents=True)
    for filename, data in [('summary.json', summary), ('cases.json', results), ('preserved_review_status.json', status)]:
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', required=True, type=Path)
    p.add_argument('--status', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1], a.reviews, a.status, a.output), ensure_ascii=False, indent=2))
