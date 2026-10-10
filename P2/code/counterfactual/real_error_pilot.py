"""Lock a <=20-output plan, prepare blind raw-motion review, then evaluate once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from counterfactual.benchmark import evaluate_predictions, validate_manifest
from counterfactual.critic_api import CriticScorer, FEATURE_VERSION, sha256, LEGACY_V1_CHECKPOINT_SHA256
from counterfactual.generator_adapter import load_features
from counterfactual.protocol_checks import canonical_hash, require_text
from models.interaction_critic import MODEL

CAP = 20
POLICY = 'real_error_pilot_v1'
ERROR_TYPES = ('none', 'premature_release', 'unsupported_motion', 'incompatible_contact',
               'motion_artifact', 'other', 'uncertain')
ACTIONS = ('carry', 'hold', 'lift', 'place', 'push', 'pull', 'other', 'unknown')
PHASES = ('approach', 'grasp', 'transport', 'sustained_hold', 'placing', 'release', 'other', 'unknown')
REASONS = ('phase_consistent', 'required_contact_broken', 'unsupported_controlled_motion',
           'action_incompatible_region', 'visible_motion_artifact', 'insufficient_evidence', 'other')
CODE_FILES = ('critic_api.py', 'generator_adapter.py', 'geometry_features.py', 'benchmark.py',
              'protocol_checks.py', 'real_error_pilot.py')


def code_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in CODE_FILES} | {
        name: sha256(Path(__file__).resolve().parents[1]/name)
        for name in ('models/interaction_critic.py', 'omomo_contact.py', 'plot_omomo_alignment.py')}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def validate_plan(plan):
    if plan.get('protocol_version') != POLICY or plan.get('review_cap') != CAP:
        raise ValueError('Unsupported pilot policy or changed review cap')
    if plan.get('data_domain') != 'real_generator' or plan.get('use_for_calibration') is not False:
        raise ValueError('Pilot must be real_generator and cannot be calibration data')
    if plan.get('unfiltered_by_critic') is not True:
        raise ValueError('A must declare outputs unfiltered by Critic')
    provenance = plan.get('generator', {})
    require_text(provenance.get('revision'), 'generator revision')
    if not re.fullmatch('[0-9a-f]{64}', provenance.get('checkpoint_sha256', '')):
        raise ValueError('Actual generator checkpoint SHA256 required')
    for key in ('training_manifest_sha256', 'pretraining_exposure'):
        if key not in provenance:
            raise ValueError('Declare training provenance; use null for unknown: ' + key)
    slots = plan.get('slots')
    if not isinstance(slots, list) or not 1 <= len(slots) <= CAP:
        raise ValueError('Plan must contain 1..20 fixed slots')
    if len(slots) != CAP:
        require_text(plan.get('shortfall_reason'), 'Reason for smaller pilot')
    ids, groups = set(), {}
    for slot in slots:
        for key in ('sample_id', 'condition_id', 'source_group', 'action', 'object', 'condition_text'):
            require_text(slot.get(key), key)
        if slot['sample_id'] in ids:
            raise ValueError('Duplicate planned sample ID')
        ids.add(slot['sample_id'])
        if type(slot.get('seed')) is not int or type(slot.get('candidate_index')) is not int:
            raise ValueError('Explicit integer seeds and candidate indices required')
        groups.setdefault(slot['condition_id'], []).append(slot)
    for slots_for_condition in groups.values():
        if len(slots_for_condition) != 2 or {s['candidate_index'] for s in slots_for_condition} != {0, 1}:
            raise ValueError('Each condition needs exactly two fixed seed slots')
        if len({s['seed'] for s in slots_for_condition}) != 2:
            raise ValueError('Seeds must differ within a condition')
        for key in ('source_group', 'action', 'object', 'condition_text'):
            if len({s[key] for s in slots_for_condition}) != 1:
                raise ValueError('Condition metadata differs between seeds')


def lock_plan(draft_path, output, checkpoint=MODEL):
    if output.exists():
        raise ValueError('Plan output already exists')
    plan = json.loads(draft_path.read_text(encoding='utf-8-sig'))
    validate_plan(plan)
    if plan.get('status') != 'draft':
        raise ValueError('Expected draft plan; never overwrite a locked plan')
    scorer = CriticScorer(checkpoint)
    if scorer.checkpoint_sha256 != LEGACY_V1_CHECKPOINT_SHA256:
        raise ValueError('First transfer pilot must use the unchanged frozen v1 checkpoint')
    plan.update(status='frozen_before_scoring', frozen_at=datetime.now(timezone.utc).isoformat(),
                checkpoint_sha256=scorer.checkpoint_sha256, threshold_error=scorer.threshold,
                rule_threshold=1.0, code_sha256=code_hashes())
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, plan)
    return plan


def validate_locked(plan, checkpoint=MODEL):
    validate_plan(plan)
    if plan.get('status') != 'frozen_before_scoring':
        raise ValueError('Lock generation plan before preparing annotations')
    if sha256(checkpoint) != plan.get('checkpoint_sha256') or plan.get('code_sha256') != code_hashes():
        raise ValueError('Frozen checkpoint/scoring code changed; do not silently reuse pilot')
    scorer = CriticScorer(checkpoint)
    if scorer.checkpoint_sha256 != LEGACY_V1_CHECKPOINT_SHA256:
        raise ValueError('Pilot requires frozen v1 checkpoint')
    if plan.get('threshold_error') != scorer.threshold or plan.get('rule_threshold') != 1.0:
        raise ValueError('Frozen thresholds changed')


def delivery_rows(plan, delivered, manifest_dir):
    slots = {s['sample_id']: s for s in plan['slots']}
    if len(delivered) != len(slots) or {d.get('sample_id') for d in delivered} != set(slots):
        raise ValueError('Delivery must account for every planned slot, including failures; no refill')
    rows, failures = [], []
    for item in delivered:
        sid = item['sample_id']; slot = slots[sid]
        if item.get('status') == 'failed':
            require_text(item.get('failure_reason'), 'generation failure reason')
            failures.append({**slot, 'failure_reason': item['failure_reason']})
            continue
        if item.get('status') != 'generated':
            raise ValueError('Delivery status must be generated or failed')
        if item.get('data_domain', 'real_generator') != 'real_generator' or item.get('split', 'diagnostic') != 'diagnostic':
            raise ValueError('Pilot delivery must use real_generator / diagnostic')
        if any(item.get(k) != v for k, v in slot.items()):
            raise ValueError('Delivered identity/condition/seed differs from locked plan')
        if item.get('label_error') is not None or any(k in item for k in ('p_error', 'p_valid', 'rule_score')):
            raise ValueError('Review input must be unlabelled and score-free')
        meta = item.get('metadata', {})
        if meta.get('input_kind') != 'raw':
            raise ValueError('Blind pilot requires actual raw geometry, not features-only input')
        path = Path(item['path'])
        path = path if path.is_absolute() else manifest_dir / path
        rows.append({**slot, 'path': str(path.resolve()), 'metadata': meta, 'split': 'diagnostic',
                     'data_domain': 'real_generator', 'label_error': None})
    if rows:
        validate_manifest(rows)
    return rows, failures


def focus_range(sample, fps):
    """Motion change only; never interpreted as a semantic failure detector."""
    n = len(sample['joints'])
    speeds = np.linalg.norm(np.diff(sample['joints'][:, [22, 23]], axis=0), axis=-1).max(axis=1)
    center = int(np.argmax(speeds)) + 1
    radius = max(1, int(round(fps)))
    return {'start': max(0, center-radius), 'end': min(n-1, center+radius),
            'basis': 'largest_hand_step; navigation_only; original_frames'}


def prepare(plan_path, delivery_path, output, checkpoint=MODEL):
    from omomo_contact import object_parts
    from plot_omomo_alignment import PARENTS
    from plotly.offline import get_plotlyjs
    if output.exists():
        raise ValueError('Pilot directory already exists; preserve existing annotations')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    validate_locked(plan, checkpoint)
    rows, failures = delivery_rows(plan, read_lines(delivery_path), delivery_path.parent)
    payload, loaded, seen_files, seen_features = [], [], set(), set()
    for i, row in enumerate(rows):
        path = Path(row['path'])
        _, info = load_features(path, row['metadata'])
        if info['input_sha256'] in seen_files or info['feature_content_sha256'] in seen_features:
            raise ValueError('Duplicate file/features in pilot; investigate without replacing planned slots')
        seen_files.add(info['input_sha256']); seen_features.add(info['feature_content_sha256'])
        with np.load(path, allow_pickle=False) as z:
            sample = {k: z[k] for k in z.files if k == 'joints' or k.startswith('obj_')}
        focus = focus_range(sample, row['metadata']['fps'])
        entry = {k: row[k] for k in ('sample_id', 'action', 'object', 'condition_text')}
        entry.update(input_sha256=info['input_sha256'], frames=len(sample['joints']),
                     fps=row['metadata']['fps'], focus=focus,
                     geometry={k: v.round(6).tolist() for k, v in sample.items()})
        payload.append(entry)
        loaded.append({**row, 'path': f'samples/{i:02d}.npz', 'input_provenance': info,
                       'frames': entry['frames'], 'review_focus': focus})
    meshes = {name: [{'prefix': prefix, 'vertices': verts[::max(1, int(np.ceil(len(verts)/700)))].round(6).tolist()}
                     for _, verts, prefix in object_parts(name)] for name in {r['object'] for r in rows}}
    output.mkdir(parents=True)
    (output/'samples').mkdir()
    for original, row in zip(rows, loaded):
        shutil.copyfile(original['path'], output/row['path'])
        if sha256(output/row['path']) != row['input_provenance']['input_sha256']:
            raise ValueError('Input changed while preparing pilot; discard incomplete output')
    shutil.copyfile(plan_path, output/'generation_plan.json')
    shutil.copyfile(delivery_path, output/'delivery.jsonl')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in loaded), encoding='utf-8')
    dump(output/'generation_failures.json', failures)
    identity = canonical_hash({'plan': sha256(plan_path), 'inputs': [(r['sample_id'], r['input_provenance']['input_sha256']) for r in loaded]})
    ui_payload = {'pilot_sha256': identity, 'entries': payload, 'meshes': meshes,
                  'parents': [int(v) for v in PARENTS], 'review_cap': CAP, 'planned_count': len(plan['slots']),
                  'failed_count': len(failures)}
    # ASCII escaping prevents user-controlled strings from ending script tags.
    (output/'queue.js').write_text('window.pilot=' + json.dumps(ui_payload, ensure_ascii=True).replace('<', '\\u003c') + ';', encoding='utf-8')
    for source, target in [('real_pilot_template.html','index.html'), ('real_pilot_form.js','real_pilot_form.js'), ('real_pilot_app.js','real_pilot_app.js')]:
        shutil.copyfile(Path(__file__).with_name(source), output/target)
    (output/'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    hashes = {p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file()}
    receipt = {'protocol_version': POLICY, 'pilot_sha256': identity, 'files_sha256': hashes,
               'prepared_count': len(rows), 'failed_count': len(failures), 'review_cap': CAP,
               'real_transfer_completed': False, 'independent_review_verified': False,
               'prepared_at': datetime.now(timezone.utc).isoformat()}
    dump(output/'pilot.json', receipt)
    return receipt


def validate_annotations(records, rows, pilot_hash):
    by_id = {r['sample_id']: r for r in rows}
    if len(records) > CAP or len(records) != len(by_id) or {r.get('sample_id') for r in records} != set(by_id):
        raise ValueError('Exactly one primary record per prepared sample required; uncertain consumes budget')
    reviewers = set()
    for record in records:
        row = by_id[record['sample_id']]
        if record.get('pilot_sha256') != pilot_hash or record.get('input_sha256') != row['input_provenance']['input_sha256']:
            raise ValueError('Annotation refers to another pilot or input')
        require_text(record.get('reviewer'), 'reviewer'); reviewers.add(record['reviewer'])
        if record.get('round') != 'primary':
            raise ValueError('This pilot imports primary review only; secondary adjudication is separate')
        label = record.get('label_error')
        if not (type(label) is int and label in (0, 1)) and label != 'uncertain':
            raise ValueError('Label must be integer 0/1 or uncertain')
        if record.get('action') not in ACTIONS or record.get('phase') not in PHASES:
            raise ValueError('Choose action and interaction phase')
        if label != 'uncertain' and (record['action'] == 'unknown' or record['phase'] == 'unknown'):
            raise ValueError('Unknown semantics require uncertain label')
        a, b = record.get('start_frame'), record.get('end_frame')
        if type(a) is not int or type(b) is not int or not 0 <= a <= b < row['frames']:
            raise ValueError('Evidence range must use valid original frame indices')
        reasons = record.get('reason_codes')
        if not isinstance(reasons, list) or not reasons or any(r not in REASONS for r in reasons):
            raise ValueError('Choose semantic reasons')
        error_reasons = {'required_contact_broken','unsupported_controlled_motion','action_incompatible_region','visible_motion_artifact'}
        if label == 0 and any(r not in ('phase_consistent','other') for r in reasons):
            raise ValueError('Normal conclusion conflicts with error/insufficient-evidence reasons')
        if label == 1 and (not (set(reasons) & (error_reasons | {'other'})) or any(r in ('phase_consistent','insufficient_evidence') for r in reasons)):
            raise ValueError('Definitive error needs matching semantic evidence')
        kind = record.get('error_type')
        if kind not in ERROR_TYPES or (label == 0 and kind != 'none') or (label == 1 and kind in ('none','uncertain')) or (label == 'uncertain' and kind != 'uncertain'):
            raise ValueError('Error type and conclusion disagree')
        if label == 1 and record.get('legal_alternative') != 'excluded':
            raise ValueError('Definitive error must exclude legitimate release/support/inertia')
        if 'other' in (record['action'], record['phase'], kind) or 'other' in reasons or kind == 'incompatible_contact':
            require_text(record.get('note'), 'Explanation for other choice')
    if len(reviewers) > 1:
        raise ValueError('Pilot is a single-reviewer primary batch; reconcile reviewer identity explicitly')


def evaluate(pilot_dir, annotation_path, output, checkpoint=MODEL):
    if output.exists():
        raise ValueError('Evaluation output exists; do not overwrite the first experiment')
    receipt = json.loads((pilot_dir/'pilot.json').read_text(encoding='utf-8'))
    for name, digest in receipt['files_sha256'].items():
        target = (pilot_dir/name).resolve()
        if not target.is_relative_to(pilot_dir.resolve()) or sha256(target) != digest:
            raise ValueError('Prepared pilot changed: ' + name)
    plan = json.loads((pilot_dir/'generation_plan.json').read_text(encoding='utf-8'))
    validate_locked(plan, checkpoint)
    rows = read_lines(pilot_dir/'manifest.jsonl'); records = read_lines(annotation_path)
    expected, failures = delivery_rows(plan, read_lines(pilot_dir/'delivery.jsonl'), pilot_dir)
    if len(rows) != len(expected) or receipt['prepared_count'] != len(rows) or receipt['failed_count'] != len(failures):
        raise ValueError('Prepared pilot does not match fixed delivery slots')
    for i, (row, original) in enumerate(zip(rows, expected)):
        if row['path'] != f'samples/{i:02d}.npz' or any(row.get(k) != v for k,v in original.items() if k != 'path'):
            raise ValueError('Prepared row differs from locked plan/delivery')
    expected_identity = canonical_hash({'plan': sha256(pilot_dir/'generation_plan.json'),
        'inputs': [(r['sample_id'],r['input_provenance']['input_sha256']) for r in rows]})
    if receipt['pilot_sha256'] != expected_identity or json.loads((pilot_dir/'generation_failures.json').read_text(encoding='utf-8')) != failures:
        raise ValueError('Pilot identity/failure accounting changed')
    validate_annotations(records, rows, receipt['pilot_sha256'])
    labels = {r['sample_id']: r for r in records}
    torch.set_num_threads(2)
    scorer = CriticScorer(checkpoint)
    arrays, provenance = [], []
    for row in rows:
        raw, info = load_features(pilot_dir/row['path'], row['metadata'])
        if info != row['input_provenance']:
            raise ValueError('Prepared feature provenance changed')
        arrays.append(raw); provenance.append(info)
    scores = scorer.score_many(arrays, feature_version=FEATURE_VERSION) if arrays else []
    predictions = []
    for row, score, info in zip(rows, scores, provenance):
        record = labels[row['sample_id']]
        predictions.append({**row, **score, 'label_error': record['label_error'],
                            'error_types': [record['error_type']] if record['label_error'] == 1 else [],
                            'input_provenance': info, 'annotation': record,
                            'use_for_calibration': False, 'protocol_version': POLICY})
    metrics = evaluate_predictions(predictions, scorer.threshold)
    values = [r['label_error'] for r in records]
    classes = {v for v in values if type(v) is int}
    summary = {'experiment_status': 'pilot_completed' if classes == {0,1} else 'executed_insufficient_label_classes',
               'scope': 'single_reviewer_small_pilot; not population or generalization proof',
               'planned': len(plan['slots']), 'generated': len(rows), 'generation_failures': receipt['failed_count'],
               'primary_review_count': len(records), 'review_cap': CAP,
               'uncertain_count': values.count('uncertain'),
               'uncertain_fraction': values.count('uncertain')/len(values) if values else None,
               'independent_review_verified': False, 'unseen_pair_verified': False,
               'use_for_calibration': False, 'threshold_tuned': False, 'quality_improvement_verified': False,
               'checkpoint_sha256': scorer.checkpoint_sha256, 'threshold_error': scorer.threshold,
               'pilot_sha256': receipt['pilot_sha256'], 'annotations_sha256': sha256(annotation_path),
               'code_sha256': code_hashes(), 'metrics': metrics}
    output.mkdir(parents=True)
    shutil.copyfile(annotation_path, output/'primary_annotations.jsonl')
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf-8')
    dump(output/'report.json', summary)
    (output/'README.md').write_text('# 真实生成错误小批迁移诊断\n\n'
        f"状态：{summary['experiment_status']}。计划 {summary['planned']}，成功生成 {len(rows)}，失败 {receipt['failed_count']}，不确定 {summary['uncertain_count']}。\n\n"
        '使用冻结旧模型、旧阈值和规则；没有校准或训练。逐例结果见 predictions.jsonl，指标及按来源分组的区间见 report.json。\n\n'
        '单人、最多20条试标；未完成独立二审，不证明普遍迁移或生成收益。单一类别的 AUROC / 平衡准确率为 null。失败生成单列，不作为交互错误标签。\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock-plan'); lock.add_argument('draft', type=Path)
    prep = sub.add_parser('prepare'); prep.add_argument('plan', type=Path); prep.add_argument('delivery', type=Path)
    score = sub.add_parser('evaluate'); score.add_argument('pilot', type=Path); score.add_argument('annotations', type=Path)
    for child in (lock, prep, score):
        child.add_argument('--output', type=Path, required=True)
        child.add_argument('--checkpoint', type=Path, default=MODEL)
    args = parser.parse_args()
    if args.command == 'lock-plan': result = lock_plan(args.draft,args.output,args.checkpoint)
    elif args.command == 'prepare': result = prepare(args.plan,args.delivery,args.output,args.checkpoint)
    else: result = evaluate(args.pilot,args.annotations,args.output,args.checkpoint)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files_sha256','metrics','slots')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
