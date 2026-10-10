"""Post-hoc rotation-only control for previously recorded FPS tolerance failures.

This explains a numerical path difference. It does not repair the frozen adapter,
relax the original tolerances, or turn failed comparisons into passes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from counterfactual.critic_api import CriticScorer, FEATURE_VERSION, sha256
from counterfactual.diagnose_input_invariance import (
    LIMITS, feature_delta, metadata, reencode_integer_fps, score_delta,
)
from counterfactual.generator_adapter import load_features, normalize_time
from models.interaction_critic import ROOT


def project_rotations(sample):
    """Change rotation arrays only; no position, scale, or time normalization."""
    return {key: Rotation.from_matrix(value).as_matrix() if key.endswith('_rot_mat')
            else value.copy() for key, value in sample.items()}


def run(source, output):
    if output.exists():
        raise ValueError('Choose a new directory; original evidence is immutable')
    protocol = json.loads((source/'protocol.json').read_text())
    result = json.loads((source/'summary.json').read_text())
    original_hashes = {str(p): sha256(p) for p in (
        source/'protocol.json', source/'summary.json', source/'comparisons.jsonl')}
    watched = {ROOT/name: digest for name, digest in protocol['watched_sha256'].items()}
    watched.update({Path(name): digest for name, digest in original_hashes.items()})
    watched[ROOT/'counterfactual/diagnose_input_invariance.py'] = protocol['diagnostic_script_sha256']
    if LIMITS != protocol['limits_fixed_before_run']:
        raise ValueError('Diagnostic tolerances differ from original protocol')
    for path, digest in watched.items():
        if sha256(path) != digest:
            raise ValueError('Source evidence changed: '+str(path))
    ids = {r['sample_id'] for r in result['failed_examples'] if r['variant'].startswith('encode_')}
    if not ids:
        raise ValueError('No FPS failures to diagnose')
    index = [json.loads(line) for line in (ROOT/'counterfactual/v1/index.jsonl').read_text().splitlines()]
    entries = [r for r in index if r['sample_id'] in ids]
    if len(entries) != len(ids):
        raise ValueError('Failed sample missing from frozen index')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    scorer = CriticScorer()
    rows = []
    metas = {}
    with tempfile.TemporaryDirectory(prefix='hoi_projection_control_') as td:
        candidate_path = Path(td)/'candidate.npz'
        for entry in entries:
            path = ROOT/'counterfactual/v1/samples'/(entry['sample_id']+'.npz')
            if sha256(path) != entry['sha256']:
                raise ValueError('Frozen sample changed: '+entry['sample_id'])
            with np.load(path, allow_pickle=False) as z:
                sample = {k: np.asarray(z[k], dtype=np.float64) for k in z.files
                          if k == 'joints' or k.endswith(('_trans', '_rot_mat', '_scale'))}
            if entry['object'] not in metas:
                metas[entry['object']] = metadata(entry['object'])
            meta = metas[entry['object']]
            raw, _ = load_features(path, meta)
            projected = project_rotations(sample)
            np.savez(candidate_path, **projected)
            projected_features, _ = load_features(candidate_path, meta)
            arrays = [raw, projected_features]
            details = []
            for factor in (2, 4):
                encoded = reencode_integer_fps(sample, factor)
                aligned = normalize_time(encoded, 30.*factor)
                np.savez(candidate_path, **encoded)
                features, _ = load_features(candidate_path, {**meta, 'fps': 30.*factor})
                arrays.append(features)
                details.append({
                    'fps': 30*factor,
                    'projected_vs_time_geometry_max_by_array': {
                        k: float(np.max(np.abs(projected[k]-aligned[k]))) for k in sample},
                    'original_vs_time_feature': feature_delta(raw, features),
                    'projected_vs_time_feature': feature_delta(projected_features, features),
                })
            scores = scorer.score_many(arrays, feature_version=FEATURE_VERSION)
            for detail, score in zip(details, scores[2:]):
                detail['original_vs_time_score'] = score_delta(scores[0], score)
                detail['projected_vs_time_score'] = score_delta(scores[1], score)
            rows.append({
                'sample_id': entry['sample_id'], 'object': entry['object'], 'sample_sha256': entry['sha256'],
                'rotation_diagnostics': {k: {
                    'original_orthogonality_max_abs': float(np.max(np.abs(v.swapaxes(-1, -2)@v-np.eye(3)))),
                    'original_det_minus_one_max_abs': float(np.max(np.abs(np.linalg.det(v)-1))),
                    'projection_max_abs': float(np.max(np.abs(projected[k]-v))),
                } for k, v in sample.items() if k.endswith('_rot_mat')},
                'nonrotation_arrays_unchanged': all(np.array_equal(sample[k], projected[k])
                                                  for k in sample if not k.endswith('_rot_mat')),
                'original_vs_projected_feature': feature_delta(raw, projected_features),
                'original_vs_projected_score': score_delta(scores[0], scores[1]),
                'time_comparisons': details,
            })
    for path, digest in watched.items():
        if sha256(path) != digest:
            raise ValueError('Input changed during control: '+str(path))
    comparisons = [c for r in rows for c in r['time_comparisons']]
    summary = {
        'scope': 'post_hoc_control_on_all_previously_failed_FPS_samples_only',
        'sample_count': len(rows), 'object_counts': dict(Counter(r['object'] for r in rows)),
        'comparison_count': len(comparisons), 'original_report_hashes': original_hashes,
        'script_sha256': sha256(Path(__file__)), 'original_tolerances_and_results_unchanged': True,
        'nonrotation_arrays_unchanged': all(r['nonrotation_arrays_unchanged'] for r in rows),
        'max_projected_vs_time_geometry_abs': max(max(c['projected_vs_time_geometry_max_by_array'].values()) for c in comparisons),
        'max_projected_vs_time_feature_abs': max(c['projected_vs_time_feature']['max_continuous_abs'] for c in comparisons),
        'max_projected_vs_time_p_error_abs': max(c['projected_vs_time_score']['p_error_abs'] for c in comparisons),
        'max_projected_vs_time_rule_score_abs': max(c['projected_vs_time_score']['rule_score_abs'] for c in comparisons),
        'projected_vs_time_contact_bit_changes': sum(c['projected_vs_time_feature']['contact_bit_changes'] for c in comparisons),
        'original_vs_projected_feature_failures': sum(not r['original_vs_projected_feature']['within_tolerance'] for r in rows),
        'projected_vs_time_feature_failures': sum(not c['projected_vs_time_feature']['within_tolerance'] for c in comparisons),
        'projected_vs_time_score_failures': sum(not c['projected_vs_time_score']['within_tolerance'] for c in comparisons),
        'source_and_core_hashes_unchanged': True, 'training': False, 'new_labels': 0,
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (output/'cases.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
