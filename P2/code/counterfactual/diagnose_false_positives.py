"""Historical clean false-positive diagnostics; never relabel, fit or calibrate.

The frozen rule's local-site feature is in unscaled mesh coordinates. Report
that observation without modifying the feature contract or the saved model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from counterfactual.audit_reviews import audit
from counterfactual.critic_api import CHANNELS, CriticScorer, FEATURE_VERSION, sha256, validate_features
from counterfactual.protocol_checks import canonical_hash
from counterfactual.review_integrity import load_review_status
from models.interaction_critic import ROOT, resample, rule_score


UNITS = ['m', 'm', 'binary', 'binary', 'm/frame', 'm/frame', 'm/frame',
         'm/frame', 'm/frame', 'native_mesh_units/frame', 'native_mesh_units/frame', 'm', 'm']
GROUPS = {'distance_and_contact': [0, 1, 2, 3], 'world_motion': [4, 5, 6],
          'relative_motion': [7, 8], 'local_site': [9, 10], 'grasp_shift': [11, 12]}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def intervals(mask):
    """Zero-based inclusive ranges, without turning geometry into phase labels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError('Expected one-dimensional interval mask')
    padded = np.r_[False, mask, False].astype(np.int8)
    return [{'start_frame': int(a), 'end_frame': int(b - 1)}
            for a, b in zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1))]


def rule_components(raw):
    raw = validate_features(raw)
    touching = raw[:, 2:4].max(axis=1) > 0
    late = int(len(raw) * .65)
    release_mask = np.any(raw[:, :2] > .075, axis=1)
    release_mask[:late] = False
    unsupported = (raw[:, 6] > .008) & ~touching
    site98 = float(np.quantile(raw[touching, 9:11].max(axis=1), .98)) if touching.any() else 0.
    parts = {'late_separation': float(release_mask[late:].mean() / .45),
             'moving_without_proximity': float(unsupported.mean() / .08),
             'local_site_jump': site98 / .10}
    return {'components': parts, 'score': float(max(parts.values())),
            'dominant_components': [k for k, v in parts.items() if v == max(parts.values())],
            'late_separation_intervals': intervals(release_mask),
            'moving_without_proximity_intervals': intervals(unsupported),
            'local_site_above_rule_scale_intervals': intervals(touching & (raw[:, 9:11].max(axis=1) > .1)),
            'local_site_98th_percentile_native_units_per_frame': site98}


def site_geometry(arrays):
    """Compare discrete nearest sites and hands in each object's rotating frame.

    Cross-part differences have no common origin and are excluded. These are
    geometric proxies, not measured contact slip, collision, or material points.
    """
    nearest = np.asarray(arrays['nearest_xyz'], dtype=np.float64)
    hands = np.asarray(arrays['joints'], dtype=np.float64)[:, [22, 23]]
    parts = np.asarray(arrays['nearest_part'])
    if np.any(~np.isin(parts, [0, 1])):
        raise ValueError('Unsupported nearest part index')
    native, metric, hand_metric = (np.zeros_like(nearest) for _ in range(3))
    scale_at_site = np.zeros(parts.shape)
    for pid in np.unique(parts):
        prefix = 'obj' if pid == 0 else 'obj_bottom'
        rk = 'obj_rot_mat' if pid == 0 else 'obj_bottom_rot_mat'
        scale = np.abs(np.asarray(arrays[prefix + '_scale'], dtype=np.float64))
        if np.any(scale <= 1e-9) or not np.isfinite(scale).all():
            raise ValueError('Degenerate object scale')
        translation = np.asarray(arrays[prefix + '_trans'], dtype=np.float64)
        rotation = np.asarray(arrays[rk], dtype=np.float64)
        take = parts == pid
        site_local_m = np.einsum('thi,tij->thj', nearest - translation[:, None], rotation)
        hand_local_m = np.einsum('thi,tij->thj', hands - translation[:, None], rotation)
        metric[take], hand_metric[take] = site_local_m[take], hand_local_m[take]
        native[take] = (site_local_m / scale[:, None, None])[take]
        scale_at_site[take] = np.broadcast_to(scale[:, None], parts.shape)[take]
    same_part = np.r_[np.ones((1, 2), dtype=bool), parts[1:] == parts[:-1]]
    def step(a):
        return np.where(same_part, np.linalg.norm(np.diff(a, axis=0, prepend=a[:1]), axis=-1), 0.)
    return {'native_step': step(native), 'site_step_m': step(metric), 'hand_step_m': step(hand_metric),
            'same_part': same_part, 'scale_at_site': scale_at_site}


def profile(arrays):
    """Training-only reference, with equal 96-frame weight per sequence."""
    if not arrays:
        raise ValueError('Empty reference group')
    model_arrays = np.stack([resample(a) for a in arrays])
    return {'count': len(arrays), 'frame_quantiles': np.quantile(model_arrays, [.01, .5, .95, .99], axis=(0, 1)),
            'sample_means': model_arrays.mean(axis=1),
            'sample_p98': np.quantile(model_arrays, .98, axis=1)}


def channel_comparison(raw, references, mean, std):
    sampled = resample(raw)
    out = []
    for c, name in enumerate(CHANNELS):
        seq_mean, seq98 = float(sampled[:, c].mean()), float(np.quantile(sampled[:, c], .98))
        item = {'channel': name, 'unit': UNITS[c], 'resampled_mean': seq_mean, 'resampled_p98': seq98,
                'model_normalized_mean': float((seq_mean - mean[0, c]) / std[0, c]), 'reference': {}}
        for group, ref in references.items():
            lo, median, p95, hi = ref['frame_quantiles'][:, c]
            # Strict/weak ranks expose ties (notably all-one contacts).
            item['reference'][group] = {'count': ref['count'], 'frame_q01_q50_q95_q99': [float(v) for v in (lo, median, p95, hi)],
                'frames_outside_q01_q99_fraction': float(((sampled[:, c] < lo) | (sampled[:, c] > hi)).mean()),
                'sample_mean_rank_fraction_lt_le': [float((ref['sample_means'][:, c] < seq_mean).mean()), float((ref['sample_means'][:, c] <= seq_mean).mean())],
                'sample_p98_rank_fraction_lt_le': [float((ref['sample_p98'][:, c] < seq98).mean()), float((ref['sample_p98'][:, c] <= seq98).mean())]}
        out.append(item)
    return out


def sensitivity(raw, scorer, clean_median):
    """Off-manifold feature replacement: network sensitivity, not physical repair."""
    arrays = []
    for ids in GROUPS.values():
        changed = raw.copy()
        changed[:, ids] = clean_median[ids]
        arrays.append(changed)
    baseline = scorer.score_features(raw, feature_version=FEATURE_VERSION)['p_error']
    scores = scorer.score_many(arrays, feature_version=FEATURE_VERSION)
    return [{'group': name, 'channels': [CHANNELS[c] for c in GROUPS[name]],
             'replacement': 'constant median of the frozen training-clean 96-frame reference',
             'p_error': s['p_error'], 'delta_from_original': s['p_error'] - baseline}
            for name, s in zip(GROUPS, scores)]


def plot_case(path, sid, raw, geometry, prediction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames = np.arange(len(raw))
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for h, color in [(0, '#d95f02'), (1, '#1b9e77')]:
        hand = ('left', 'right')[h]
        axes[0].plot(frames, raw[:, h], color=color, label=hand)
        axes[1].plot(frames, raw[:, 9 + h], color=color, label=hand)
        axes[2].plot(frames, geometry['site_step_m'][:, h], color=color, label=f'{hand} nearest-site')
        axes[2].plot(frames, geometry['hand_step_m'][:, h], color=color, ls='--', alpha=.7, label=f'{hand} hand')
        axes[3].plot(frames, raw[:, 11 + h], color=color, label=hand)
    axes[0].axhline(.05, color='black', ls='--', label='proximity threshold')
    axes[1].axhline(.1, color='black', ls='--', label='frozen rule scale (native units)')
    for ax, label in zip(axes, ['Hand-to-vertex distance (m)', 'Clipped site step (native units)', 'Object-frame step (m/frame)', 'Initial grasp reference shift (m)']):
        ax.set_ylabel(label); ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Original sequence frame (zero-based; no verified frame-rate conversion)')
    fig.suptitle(f"{sid}\np_error={prediction['p_error']:.4f}; rule={prediction['rule_score']:.4f}; diagnostic only")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(root, reviews_path, status_path, output):
    if output.exists():
        raise ValueError('Output exists; preserve previous diagnostic runs')
    root = Path(root)
    frozen = root / 'counterfactual/v1'
    parent_manifest = read_json(frozen / 'manifest.json')
    if sha256(frozen / 'index.jsonl') != parent_manifest['index_sha256']:
        raise ValueError('Frozen index changed')
    entries = read_lines(frozen / 'index.jsonl')
    queue_path = root / 'counterfactual/v1.1-reviewed-subset/queue.json'
    queue = read_json(queue_path)
    if {e['sample_id']: e['sha256'] for e in queue} != {e['sample_id']: e['sha256'] for e in entries}:
        raise ValueError('Review queue is not the full frozen population')
    votes = read_lines(reviews_path)
    status = load_review_status(status_path, reviews_path)
    current_audit = audit(queue, votes, review_status=status)
    saved_audit = read_json(root / 'counterfactual/v1.1-reviewed-subset/audit.json')
    if canonical_hash(current_audit) != canonical_hash(saved_audit):
        raise ValueError('Recomputed audit differs from current frozen review audit')
    historical = read_json(root / 'reports/critic_metrics.json')
    regression = read_json(root / 'reports/code_audit_20261004/historical_regression_final.json')
    if historical['dataset_manifest_sha256'] != sha256(frozen / 'manifest.json'):
        raise ValueError('Historical metrics refer to another dataset')
    by_id, arrays = {}, {}
    for e in entries:
        path = frozen / 'samples' / (e['sample_id'] + '.npz')
        if sha256(path) != e['sha256']:
            raise ValueError('Sample changed: ' + e['sample_id'])
        with np.load(path, allow_pickle=False) as z:
            arrays[e['sample_id']] = validate_features(z['features']).copy()
        by_id[e['sample_id']] = e
    train = [e for e in entries if e['split'] == 'train' and e['severity'] != 'high']
    refs = {name: profile([arrays[e['sample_id']] for e in train if label is None or e['label'] == label])
            for name, label in [('train_all', None), ('train_clean', 0), ('train_synthetic_error', 1)]}
    torch.set_num_threads(2)
    scorer = CriticScorer(root / 'models/interaction_critic_v1.pt')
    if scorer.checkpoint_sha256 != regression['checkpoint_sha256'] or scorer.threshold != historical['critic_threshold']:
        raise ValueError('Frozen model/threshold mismatch')
    scores = scorer.score_many([arrays[e['sample_id']] for e in entries], feature_version=FEATURE_VERSION)
    predictions = {e['sample_id']: score for e, score in zip(entries, scores)}
    parity = max(abs(predictions[r['sample_id']]['p_error'] - r['critic_score']) for r in historical['test_predictions'])
    if parity > 1e-6:
        raise ValueError('Historical score parity failed')
    clean_test = [e for e in entries if e['split'] == 'test' and e['label'] == 0]
    false_positive = [e for e in clean_test if predictions[e['sample_id']]['error_flag']]
    if {e['sample_id'] for e in false_positive} != {e['sample_id'] for e in regression['false_positive_examples']}:
        raise ValueError('Historical false-positive membership changed')
    cases, plot_inputs = [], []
    for e in false_positive:
        sid, raw = e['sample_id'], arrays[e['sample_id']]
        with np.load(frozen / 'samples' / (sid + '.npz'), allow_pickle=False) as z:
            geometry = site_geometry(z)
        component = rule_components(raw)
        if not np.isclose(component['score'], rule_score(raw), rtol=2e-7):
            raise ValueError('Rule decomposition failed')
        top_frames = np.argsort(-raw[:, 9:11].max(axis=1), kind='stable')[:5]
        peaks = []
        for frame in top_frames:
            h = int(np.argmax(raw[frame, 9:11]))
            peaks.append({'frame': int(frame), 'hand': ('left', 'right')[h],
                          'stored_native_site_step': float(raw[frame, 9 + h]),
                          'reconstructed_unclipped_native_site_step': float(geometry['native_step'][frame, h]),
                          'site_step_object_frame_m': float(geometry['site_step_m'][frame, h]),
                          'hand_step_object_frame_m': float(geometry['hand_step_m'][frame, h]),
                          'object_scale': float(geometry['scale_at_site'][frame, h]),
                          'context_range': {'start_frame': max(0, int(frame) - 8), 'end_frame': min(len(raw) - 1, int(frame) + 8)}})
        records = [v for v in votes if v['sample_id'] == sid]
        comparisons = channel_comparison(raw, refs, scorer.mean, scorer.std)
        dominant_deviations = sorted(comparisons, key=lambda c: c['reference']['train_clean']['frames_outside_q01_q99_fraction'], reverse=True)[:5]
        case = {'sample': e, 'prediction': predictions[sid], 'historical_label_error': 0,
                'current_review_status': current_audit['statuses'][sid], 'existing_review_records': records,
                'review_record_hashes': [canonical_hash(v) for v in records],
                'single_phase_not_certified': True, 'semantic_label_changed': False,
                'rule_decomposition': component, 'proximity_fraction_by_hand': raw[:, 2:4].mean(axis=0).astype(float).tolist(),
                'maximum_distance_m_by_hand': raw[:, :2].max(axis=0).astype(float).tolist(),
                'scale_range': [float(geometry['scale_at_site'].min()), float(geometry['scale_at_site'].max())],
                'part_switch_count': int((~geometry['same_part']).sum()),
                'stored_site_clip_ceiling_count': int((raw[:, 9:11] == 2).sum()),
                'peak_site_frames': peaks, 'channels': comparisons,
                'largest_training_clean_distribution_deviations': [c['channel'] for c in dominant_deviations],
                'feature_replacement_sensitivity_nonphysical': sensitivity(raw, scorer, refs['train_clean']['frame_quantiles'][1]),
                'plot': sid + '.png'}
        cases.append(case); plot_inputs.append((sid, raw, geometry, predictions[sid]))
    clean_comparison = [{'sample_id': e['sample_id'], 'review_status': current_audit['statuses'][e['sample_id']],
                         **predictions[e['sample_id']]} for e in clean_test]
    hashes = {str(p.relative_to(root)): sha256(p) for p in [frozen / 'manifest.json', frozen / 'index.jsonl',
        root / 'models/interaction_critic_v1.pt', root / 'counterfactual/build_v1.py', queue_path,
        root / 'counterfactual/v1.1-reviewed-subset/audit.json', root / 'reports/critic_metrics.json',
        root / 'reports/code_audit_20261004/historical_regression_final.json']}
    hashes.update({'reviews': sha256(reviews_path), 'review_status': sha256(status_path),
                   'diagnostic_source': sha256(Path(__file__))})
    summary = {'date_local': '2026-10-05', 'scope': 'historical_clean_false_positives_only',
        'historical_clean_test_count': len(clean_test), 'historical_false_positive_count': len(cases),
        'current_review_status_of_historical_false_positives': dict(Counter(c['current_review_status'] for c in cases)),
        'training_reference_samples': {k: v['count'] for k, v in refs.items()},
        'all_sample_hashes_verified': len(entries), 'historical_predictions_compared': len(historical['test_predictions']),
        'max_historical_probability_difference': parity, 'threshold_unchanged': scorer.threshold,
        'input_hashes': hashes, 'human_budget_consumed': 0,
        'remaining_synthetic_primary_budget': status['budget_additional_primary_remaining_cap'],
        'training_started': False, 'threshold_fitted': False, 'new_human_tasks': 0, 'new_labels': 0,
        'frozen_data_changed': False, 'gpu_used': False,
        'limits': ['Selected historical cases, not fresh testing or real-generator transfer.',
                  'Four existing valid votes and one uncertain vote do not independently certify physical truth.',
                  'The local-site channels use unscaled mesh units, not metres; their meaning is preserved in v1.',
                  'Nearest-site jumps and feature replacement are not measured contact slip or causal explanations.',
                  'Sensitivity inputs are nonphysical; no changed input is released as an improved sample.',
                  'All 13-channel summaries use frozen training-only references; no threshold is retuned.']}
    output.mkdir(parents=True)
    for name, data in [('summary.json', summary), ('cases.json', cases), ('held_out_clean_comparison.json', clean_comparison),
                       ('current_reviews_audit.json', current_audit), ('preserved_review_status.json', status)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for sid, raw, geometry, prediction in plot_inputs:
        plot_case(output / (sid + '.png'), sid, raw, geometry, prediction)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', required=True, type=Path)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.reviews, args.status, args.output), ensure_ascii=False, indent=2))
