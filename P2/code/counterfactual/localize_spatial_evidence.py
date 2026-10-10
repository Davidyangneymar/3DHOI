"""Localize two existing semantic leads on full meshes; no labels or review tasks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from counterfactual.diagnose_spatial import sha
from counterfactual.mesh_surface_evidence import read_triangle_obj, topology, TriangleSurface, world_to_local, motion_cues


CASES = ['sub13_whitechair_000__spatial_contact_shift__medium',
         'sub15_largetable_008__spatial_contact_shift__medium']


def distribution(values):
    x = np.asarray(values, dtype=float)
    return {'min': float(x.min()), 'median': float(np.median(x)), 'max': float(x.max())}


def plot_case(result, vertices, faces, output):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    anchor = np.asarray(result['region_coordinates']['clean_nearest_triangle_point_at_cut_local'])
    target = np.asarray(result['region_coordinates']['target_vertex_local'])
    for ax, pair in zip(axes[0], [(0, 1), (0, 2)]):
        # All original faces are used; projection occlusion does not imply volume.
        ax.add_collection(PolyCollection(vertices[faces][:, :, pair], facecolors='#d0d8df',
                                         edgecolors='none', rasterized=True))
        ax.scatter(*anchor[list(pair)], c='#007d92', s=65, label='Clean surface point at cut', zorder=4)
        ax.scatter(*target[list(pair)], c='#c43a35', s=120, marker='*', label='Frozen target vertex', zorder=5)
        ax.autoscale(); ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Local ' + 'xyz'[pair[0]]); ax.set_ylabel('Local ' + 'xyz'[pair[1]])
        ax.set_title('Full triangle mesh projection (not an inside/outside test)')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5,-.16), ncol=1)
    n = len(result['series']['clean_surface_distance_m'])
    time = np.arange(n)
    axes[1, 0].plot(time, result['series']['clean_surface_distance_m'], label='Clean: surface')
    axes[1, 0].plot(time, result['series']['frozen_surface_distance_m'], label='Frozen: surface')
    axes[1, 0].plot(time, result['series']['frozen_vertex_distance_m'], ':', label='Frozen: vertex')
    axes[1, 0].axhline(.05, c='gray', ls='--', lw=.8, label='0.05 m cue')
    axes[1, 0].set_title('Shifted-hand joint: unsigned distance (m)')
    axes[1, 0].legend(fontsize=8,loc='upper right')
    cues = result['motion_cues']
    axes[1, 1].plot(time, cues['object_lowest_world_z_m'], label='Object lowest vertex')
    axes[1, 1].plot(time, cues['foot_joint_min_world_z_m'], label='Lowest foot joint; NOT ground')
    axes[1, 1].set_title('World z (m); no registered floor/support surface')
    axes[1, 1].legend(fontsize=8,loc='upper right')
    axes[2, 0].plot(time[1:], cues['object_translation_step_m'])
    axes[2, 0].set_title('Object translation increment (m/frame)')
    axes[2, 1].plot(time[1:], cues['object_rotation_step_deg'])
    axes[2, 1].set_title('Object rotation increment (degrees/frame)')
    lo, hi = result['review_overlap_frames']
    for ax in axes[1:].flat:
        ax.axvspan(lo, hi, color='#e7bd40', alpha=.16)
        ax.axvline(result['cut_frame'], color='#777777', ls=':', lw=.8)
        ax.set_xlabel('Frame'); ax.grid(alpha=.15)
    fig.suptitle(result['sample_id'] + '\nShaded: existing review overlap; dotted vertical line: frozen cut', fontsize=13)
    fig.savefig(output, dpi=145)
    plt.close(fig)


def run(root, reviews_path, status_path, output):
    if output.exists():
        raise ValueError('Output exists; do not overwrite previous evidence')
    status = json.loads(status_path.read_text(encoding='utf-8'))
    if sha(reviews_path) != status['source_reviews_sha256']:
        raise ValueError('Review/status hash mismatch')
    frozen = root / 'counterfactual/v1'
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    if sha(frozen / 'index.jsonl') != manifest['index_sha256']:
        raise ValueError('Frozen index hash mismatch')
    entries = {e['sample_id']: e for e in map(json.loads, (frozen / 'index.jsonl').read_text().splitlines())}
    votes = [json.loads(line) for line in reviews_path.read_text(encoding='utf-8').splitlines()]
    results, plot_inputs = [], []
    for sample_id in CASES:
        entry = entries[sample_id]
        records = [v for v in votes if v['sample_id'] == sample_id]
        if not records or any(v['sample_sha256'] != entry['sha256'] for v in records):
            raise ValueError('Missing or stale review records')
        clean_id = entry['source_sequence'] + '__clean__none'
        clean_path = frozen / 'samples' / (clean_id + '.npz')
        path = frozen / 'samples' / (sample_id + '.npz')
        if sha(path) != entry['sha256'] or sha(clean_path) != entries[clean_id]['sha256']:
            raise ValueError('Sample hash mismatch')
        mesh = root / 'data/captured_objects' / (entry['object'] + '_cleaned_simplified.obj')
        vertices, faces = read_triangle_obj(mesh)
        surface = TriangleSurface(vertices, faces)
        with np.load(clean_path, allow_pickle=False) as c, np.load(path, allow_pickle=False) as z:
            hand = 0 if entry['parameters']['hand'] == 'left' else 1
            r, t, s = c['obj_rot_mat'].astype(float), c['obj_trans'].astype(float), c['obj_scale'].astype(float)
            if not all(np.array_equal(c[k], z[k]) for k in c.files if k.startswith('obj')):
                raise ValueError('Expected unchanged object pose')
            clean_local = world_to_local(c['joints'][:, 22 + hand], r, t, s)
            frozen_local = world_to_local(z['joints'][:, 22 + hand], r, t, s)
            cd, cp, ci = surface.query(clean_local)
            fd, fp, fi = surface.query(frozen_local)
            cd *= np.abs(s); fd *= np.abs(s)
            cues = motion_cues(vertices, r, t, s, c['joints'])
            vertex_distance = z['distances_m'][:, hand].astype(float)
        cut = entry['parameters']['cut_frame']
        target_id = entry['parameters']['target_vertex']
        if not 0 <= target_id < len(vertices):
            raise ValueError('Target vertex out of mesh bounds')
        target = vertices[target_id]
        lo = max(v['start_frame'] for v in records); hi = min(v['end_frame'] for v in records)
        if not 0 <= lo <= hi < len(cd):
            raise ValueError('Invalid review overlap')
        bbox_min, bbox_size = vertices.min(axis=0), np.ptp(vertices, axis=0)
        result = {'sample_id': sample_id, 'object': entry['object'], 'sample_sha256': entry['sha256'],
                  'clean_sha256': entries[clean_id]['sha256'], 'mesh_path': str(mesh.resolve()), 'mesh_sha256': sha(mesh),
                  'topology': topology(vertices, faces), 'existing_reviews_reference_only': records,
                  'review_overlap_frames': [lo, hi], 'cut_frame': cut,
                  'region_coordinates': {'target_vertex_index_zero_based': target_id,
                      'target_vertex_local': target.tolist(),
                      'clean_nearest_triangle_point_at_cut_local': cp[cut].tolist(),
                      'clean_nearest_triangle_index_at_cut': int(ci[cut]),
                      'target_bbox_coordinates': ((target - bbox_min) / np.maximum(bbox_size, 1e-12)).tolist(),
                      'clean_surface_point_bbox_coordinates': ((cp[cut] - bbox_min) / np.maximum(bbox_size, 1e-12)).tolist(),
                      'surface_point_to_target_distance_at_cut_m': float(abs(s[cut]) * np.linalg.norm(target - cp[cut])),
                      'bbox_coordinates_are_not_semantic_regions': True},
                  'review_window_metrics': {
                      'clean_surface_distance_m': distribution(cd[lo:hi+1]),
                      'frozen_surface_distance_m': distribution(fd[lo:hi+1]),
                      'vertex_minus_surface_distance_m': distribution(vertex_distance[lo:hi+1] - fd[lo:hi+1]),
                      'object_lowest_z_m': distribution(cues['object_lowest_world_z_m'][lo:hi+1]),
                      'translation_increment_m': distribution(cues['object_translation_step_m'][lo:hi]) if hi > lo else None,
                      'rotation_increment_deg': distribution(cues['object_rotation_step_deg'][lo:hi]) if hi > lo else None},
                  'series': {'clean_surface_distance_m': cd.tolist(), 'frozen_surface_distance_m': fd.tolist(),
                             'frozen_vertex_distance_m': vertex_distance.tolist(),
                             'frozen_nearest_triangle_index': fi.tolist(),
                             'frozen_nearest_triangle_point_local': fp.tolist()},
                  'motion_cues': cues, 'semantic_label': None, 'eligible_for_training': False}
        results.append(result); plot_inputs.append((vertices, faces))
    inputs = {'reviews': sha(reviews_path), 'review_status': sha(status_path),
              'frozen_index': sha(frozen / 'index.jsonl'), 'frozen_manifest': sha(frozen / 'manifest.json')}
    for code in ('counterfactual/mesh_surface_evidence.py', 'counterfactual/localize_spatial_evidence.py'):
        inputs[code] = sha(root / code)
    summary = {'date_local': '2026-10-04', 'cases': len(results), 'input_hashes': inputs,
               'new_human_tasks': 0, 'new_labels': 0, 'training_started': False,
               'remaining_primary_budget_cap': status['budget_additional_primary_remaining_cap'],
               'inside_outside_available': False, 'physical_support_available': False,
               'full_mesh_used_without_repair': True,
               'signed_distance_and_support_must_not_be_inferred_from_unsigned_cues': True}
    output.mkdir(parents=True)
    for result, (vertices, faces) in zip(results, plot_inputs):
        plot_case(result, vertices, faces, output / (result['object'] + '_evidence.png'))
    for name, data in [('summary.json', summary), ('cases.json', results), ('preserved_review_status.json', status)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[1], args.reviews, args.status, args.output), indent=2))
