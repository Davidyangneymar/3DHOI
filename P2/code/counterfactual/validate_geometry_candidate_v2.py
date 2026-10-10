"""Geometry acceptance only: fixed synthetic inputs, no labels, fitting or scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from counterfactual.critic_api import sha256
from counterfactual.diagnose_input_invariance import VARIANTS, rigid_transform, reencode_integer_fps, metadata, feature_delta
from counterfactual.generator_adapter_v2 import load_aligned_geometry
from counterfactual.geometry_candidate_v2 import FEATURE_VERSIONS, candidate_metadata, build_candidate
from counterfactual.geometry_features import features as legacy_features
from counterfactual.geometry_features_v2 import CHANNELS, UNITS, validate_features
from models.interaction_critic import ROOT


LIMITS = {'continuous_abs': 2e-5, 'binary_changes': 0}
BINARY = (2, 3, 13, 14, 15, 16)
CONTINUOUS = tuple(i for i in range(17) if i not in BINARY)


def compare(a, b):
    validate_features(a); validate_features(b)
    if a.shape != b.shape:
        return {'passed': False, 'shape_equal': False}
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    continuous = float(delta[:, CONTINUOUS].max())
    binary = int(np.count_nonzero(a[:, BINARY] != b[:, BINARY]))
    return {'passed': continuous <= LIMITS['continuous_abs'] and binary == 0,
            'shape_equal': True, 'max_continuous_abs': continuous, 'binary_changes': binary,
            'max_abs_by_channel': dict(zip(CHANNELS, map(float, delta.max(axis=0))))}


def method_comparison(entry, vertex, surface, vertex_trace, surface_trace):
    common = vertex_trace['site_step_valid'] & surface_trace['site_step_valid']
    rows = {}
    for name, raw, trace in [('vertex', vertex, vertex_trace), ('surface', surface, surface_trace)]:
        values = trace['same_part_step_m'][common]
        rows[name] = {
            'distance_mean_m': float(trace['distance_m'].mean()),
            'distance_max_m': float(trace['distance_m'].max()),
            'near_hand_frames': int(np.count_nonzero(raw[:, 2:4])),
            'valid_site_steps': int(np.count_nonzero(trace['site_step_valid'])),
            'part_changes': int(np.count_nonzero(trace['part_changed'])),
            'site_step_common_valid_p98_m': float(np.quantile(values, .98)) if len(values) else None,
            'site_step_common_valid_max_m': float(values.max()) if len(values) else None,
        }
    return {'sample_id': entry['sample_id'], 'type': entry['type'], 'object': entry['object'],
            'split': entry['split'], 'source_sequence': entry['source_sequence'],
            'sample_sha256': entry['sha256'], 'common_valid_steps': int(common.sum()),
            'surface_minus_vertex_distance_max_m': float((surface_trace['distance_m']-vertex_trace['distance_m']).max()),
            'contact_bit_changes_between_methods': int(np.count_nonzero(vertex[:, 2:4] != surface[:, 2:4])),
            'methods': rows}


def run(output):
    if output.exists():
        raise ValueError('Choose a new report directory')
    previous = ROOT/'reports/b_diagnostics_20261005/receipt.json'
    preserved = json.loads(previous.read_text())['preserved_sha256']
    preserved['reports/b_diagnostics_20261005/receipt.json'] = sha256(previous)
    scripts = [ROOT/'counterfactual'/name for name in (
        'generator_adapter_v2.py', 'geometry_features_v2.py', 'contact_surface_v2.py', 'contact_vertex_v2.py',
        'geometry_candidate_v2.py', 'validate_geometry_candidate_v2.py', 'mesh_surface_evidence.py')]
    scripts.append(ROOT/'plot_omomo_alignment.py')
    before = {**preserved, **{str(p.relative_to(ROOT)): sha256(p) for p in scripts}}
    for name, digest in before.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen evidence changed: '+name)
    entries = [json.loads(line) for line in (ROOT/'counterfactual/v1/index.jsonl').read_text().splitlines()]
    output.mkdir(parents=True)
    protocol = {'scope': 'candidate_geometry_only', 'samples': len(entries), 'variants': VARIANTS,
                'limits_fixed_before_run': LIMITS, 'feature_versions': FEATURE_VERSIONS,
                'methods': ['vertex', 'surface'], 'code_and_preserved_sha256': before,
                'no_method_selected': True, 'new_human_labels': 0, 'training': False,
                'checkpoint_scoring': False, 'classification_improvement_verified': False,
                'reference_fps_status': 'inferred', 'fps_encoding': 'integer interpolation retaining observed anchors'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    records = []; pairs = []; metadata_cache = {}; elapsed = Counter(); frame_count = 0
    base_arrays = {method: [] for method in FEATURE_VERSIONS}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='hoi_candidate_acceptance_') as td, \
            (output/'comparisons.jsonl').open('x', encoding='utf-8') as stream:
        temporary = Path(td)/'changed.npz'
        for idx, entry in enumerate(entries, 1):
            source = ROOT/'counterfactual/v1/samples'/(entry['sample_id']+'.npz')
            if sha256(source) != entry['sha256']:
                raise ValueError('Frozen sample hash changed: '+entry['sample_id'])
            if entry['object'] not in metadata_cache:
                metadata_cache[entry['object']] = candidate_metadata(metadata(entry['object']), 'vertex')
            meta = metadata_cache[entry['object']]
            aligned, _ = load_aligned_geometry(source, meta)
            frame_count += len(aligned['joints'])
            with np.load(source, allow_pickle=False) as z:
                original = {k: np.asarray(z[k], dtype=np.float64) for k in z.files
                            if k == 'joints' or k.endswith(('_trans', '_rot_mat', '_scale'))}
            bases = {}; traces = {}
            for method in FEATURE_VERSIONS:
                tick = time.perf_counter()
                bases[method], traces[method] = build_candidate(aligned, entry['object'], method, expected_mesh_sha256=meta['mesh_sha256'])
                base_arrays[method].append(bases[method])
                elapsed[method] += time.perf_counter()-tick
            pairs.append(method_comparison(entry, bases['vertex'], bases['surface'], traces['vertex'], traces['surface']))
            t = traces['vertex']
            legacy_base = legacy_features(aligned, aligned['joints'], t['distance_m'], t['nearest_world_m'], t['nearest_part'], entry['object'])
            for variant in VARIANTS:
                if variant['kind'] == 'rigid':
                    changed = rigid_transform(original, variant['yaw_degrees'], variant['translation_m']); fps = 30.
                else:
                    changed = reencode_integer_fps(original, variant['factor']); fps = 30.*variant['factor']
                np.savez(temporary, **changed)
                adapted, _ = load_aligned_geometry(temporary, {**meta, 'fps': fps})
                for method in FEATURE_VERSIONS:
                    tick = time.perf_counter()
                    raw, detail = build_candidate(adapted, entry['object'], method, expected_mesh_sha256=meta['mesh_sha256'])
                    elapsed[method] += time.perf_counter()-tick
                    row = {'sample_id': entry['sample_id'], 'method': method, 'variant': variant['name'], **compare(bases[method], raw)}
                    if method == 'vertex':
                        legacy = legacy_features(adapted, adapted['joints'], detail['distance_m'], detail['nearest_world_m'], detail['nearest_part'], entry['object'])
                        row['adapter_only_legacy_layout_comparison'] = feature_delta(legacy_base, legacy)
                    records.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            if idx % 20 == 0 or idx == 1:
                print(f'Validated {idx}/{len(entries)} samples; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    for name, digest in before.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Input or source changed during acceptance: '+name)
    by_variant = {}
    for method in FEATURE_VERSIONS:
        for variant in VARIANTS:
            rows = [r for r in records if r['method'] == method and r['variant'] == variant['name']]
            key = method+':'+variant['name']
            by_variant[key] = {'comparisons': len(rows), 'failures': sum(not r['passed'] for r in rows),
                               'max_continuous_abs': max((r['max_continuous_abs'] for r in rows if r['shape_equal']), default=None),
                               'binary_changes': sum(r.get('binary_changes', 0) for r in rows)}
    legacy_rows = [r['adapter_only_legacy_layout_comparison'] for r in records if r['method'] == 'vertex']
    clean_pairs = [p for p in pairs if p['type'] == 'clean']
    from counterfactual.contact_surface_v2 import object_part_surfaces
    mesh_summaries = {name: {label: surface.summary() for label, surface, _ in object_part_surfaces(name)} for name in metadata_cache}
    ranges = {}
    for method, arrays in base_arrays.items():
        all_frames = np.concatenate(arrays)
        ranges[method] = [{'channel': channel, 'unit': unit, 'min': float(all_frames[:,i].min()),
                           'max': float(all_frames[:,i].max()), 'p99': float(np.quantile(all_frames[:,i], .99))}
                          for i, (channel, unit) in enumerate(zip(CHANNELS, UNITS))]
    summary = {'status': 'passed' if all(r['passed'] for r in records) and all(r['within_tolerance'] for r in legacy_rows) else 'differences_require_review',
               'samples': len(entries), 'source_sequences': len({e['source_sequence'] for e in entries}),
               'objects': len(metadata_cache), 'original_frames': frame_count, 'comparisons': len(records),
               'by_variant': by_variant,
               'adapter_only_legacy_layout_failures': sum(not r['within_tolerance'] for r in legacy_rows),
               'adapter_only_legacy_layout_max_abs': max((r['max_continuous_abs'] for r in legacy_rows if r['shape_equal']), default=None),
               'method_comparison_samples': len(pairs), 'clean_comparison_samples': len(clean_pairs),
               'contact_bit_changes_between_methods': sum(p['contact_bit_changes_between_methods'] for p in pairs),
               'surface_distance_exceeds_vertex_by_more_than_1e_minus8_m_samples': sum(p['surface_minus_vertex_distance_max_m'] > 1e-8 for p in pairs),
               'elapsed_seconds_by_method': dict(elapsed), 'total_seconds': time.perf_counter()-start,
               'code_and_preserved_hashes_unchanged': True, 'training_ready': False,
               'training_blocker': 'val_needs_both_classes_after_withholding_high',
               'new_labels': 0, 'synthetic_budget_consumed': 0, 'real_budget_consumed': 0,
               'no_method_selected': True, 'new_checkpoint_available': False,
               'classification_improvement_verified': False}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (output/'method_comparison.json').write_text(json.dumps(pairs, indent=2, allow_nan=False), encoding='utf-8')
    (output/'feature_ranges.json').write_text(json.dumps(ranges, indent=2, allow_nan=False), encoding='utf-8')
    (output/'mesh_topology.json').write_text(json.dumps(mesh_summaries, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
