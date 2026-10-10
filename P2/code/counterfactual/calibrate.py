"""Dev-only monotonic Platt calibration; zero-shot scores are retained separately."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from counterfactual.benchmark import binary_metrics
from counterfactual.critic_api import sha256
from counterfactual.protocol_checks import canonical_hash, require_text, valid_error_label


def validate_scores(rows):
    ids, models, versions, domains = set(), set(), set(), set()
    for r in rows:
        if r.get('use_for_calibration') is False:
            raise ValueError('This frozen pilot explicitly forbids calibration, even if split is renamed')
        for key in ('sample_id', 'source_group', 'condition_id', 'checkpoint_sha256', 'feature_version', 'data_domain'):
            require_text(r.get(key), key)
        if r['data_domain'] == 'unspecified':raise ValueError('Calibration requires an explicitly declared data domain')
        if r['sample_id'] in ids:
            raise ValueError('Duplicate sample_id')
        ids.add(r['sample_id']); models.add(r['checkpoint_sha256']); versions.add(r['feature_version']); domains.add(r['data_domain'])
        p, rule = r.get('p_error'), r.get('rule_score')
        if not isinstance(p, (int,float)) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('p_error must be finite and in [0,1]')
        if not isinstance(rule, (int,float)) or not np.isfinite(rule):
            raise ValueError('rule_score must be finite')
        if not valid_error_label(r.get('label_error')):
            raise ValueError('Invalid label_error')
    if len(models) != 1 or len(versions) != 1 or len(domains) != 1:
        raise ValueError('One nonempty checkpoint/feature version/data domain per calibration experiment')


def choose_threshold(labels, scores, max_fpr=.2, min_recall=.6):
    if not 0 <= max_fpr <= 1 or not 0 <= min_recall <= 1:
        raise ValueError('Invalid FPR/recall constraints')
    scores = np.asarray(scores, dtype=float)
    choices = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    feasible = []
    for t in choices:
        m = binary_metrics(labels, scores, t)
        if m['false_positive_rate'] is not None and m['recall_error'] is not None and m['false_positive_rate'] <= max_fpr and m['recall_error'] >= min_recall:
            feasible.append(m)
    if not feasible:
        return {'status':'no_feasible_threshold','threshold':None}
    selected = max(feasible, key=lambda m:(m['balanced_accuracy'], -m['false_positive_rate'], m['threshold']))
    return {'status':'selected_on_dev','threshold':selected['threshold'], 'dev_metrics':selected}


def fit_calibration(rows, max_fpr=.2, min_recall=.6):
    validate_scores(rows)
    if any(r.get('split') != 'dev' for r in rows):
        raise ValueError('Calibration accepts DEV ONLY; test/diagnostic rows are forbidden')
    labelled = [r for r in rows if r.get('label_error') in (0,1)]
    y = np.asarray([r['label_error'] for r in labelled], dtype=float)
    if set(y) != {0,1}:
        raise ValueError('Both labelled classes are required in dev')
    if len({r['source_group'] for r in labelled}) < 2:
        raise ValueError('At least two source groups are required')
    p = np.clip([r['p_error'] for r in labelled], 1e-6, 1-1e-6)
    logits = np.log(p/(1-p))
    # Strictly positive slope preserves score ordering; weak regularization is fixed a priori.
    def objective(theta):
        a,b = theta; z=a*logits+b
        loss=np.mean(np.logaddexp(0,z)-y*z)+.001*((a-1)**2+b*b)
        residual=expit(z)-y
        gradient=np.array([np.mean(residual*logits)+.002*(a-1),np.mean(residual)+.002*b])
        return loss,gradient
    fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(1e-4,100),(-100,100)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f'Calibration optimization failed: {fit.message}')
    a,b=map(float,fit.x);calibrated=expit(a*logits+b)
    critic=choose_threshold(y,calibrated,max_fpr,min_recall)
    rule=choose_threshold(y,[r['rule_score'] for r in labelled],max_fpr,min_recall)
    return {'version':'monotonic_platt_v1','checkpoint_sha256':rows[0]['checkpoint_sha256'],
            'data_domain':rows[0]['data_domain'],
            'feature_version':rows[0]['feature_version'],'slope':a,'intercept':b,'clip_epsilon':1e-6,
            'regularization':.001,'fit_split':'dev','dev_predictions_sha256':canonical_hash(rows),
            'dev_sample_ids':sorted(r['sample_id'] for r in rows),
            'dev_source_groups':sorted({r['source_group'] for r in rows}),
            'dev_condition_ids':sorted({r['condition_id'] for r in rows}),
            'dev_input_sha256':sorted({r.get('input_provenance',{}).get('input_sha256') for r in rows} - {None}),
            'dev_feature_content_sha256':sorted({r.get('input_provenance',{}).get('feature_content_sha256') for r in rows} - {None}),
            'labelled_count':len(labelled),'excluded_count':len(rows)-len(labelled),
            'constraints':{'max_fpr':max_fpr,'min_recall':min_recall},
            'critic_threshold_selection':critic,'rule_threshold_selection':rule,
            'dev_brier_before':float(np.mean((p-y)**2)), 'dev_brier_after':float(np.mean((calibrated-y)**2)),
            'generalization_verified':False}


def apply_calibration(rows, artifact):
    validate_scores(rows)
    if artifact.get('version') != 'monotonic_platt_v1' or artifact.get('fit_split') != 'dev':
        raise ValueError('Unsupported calibration artifact')
    if any(r.get('split') != 'test' for r in rows):
        raise ValueError('Apply accepts held-out TEST ONLY')
    if any(r['checkpoint_sha256'] != artifact['checkpoint_sha256'] or r['feature_version'] != artifact['feature_version'] or r['data_domain'] != artifact['data_domain'] for r in rows):
        raise ValueError('Calibration/checkpoint/feature mismatch')
    for field,key in [('sample_id','dev_sample_ids'),('source_group','dev_source_groups'),('condition_id','dev_condition_ids')]:
        if {r[field] for r in rows} & set(artifact[key]):
            raise ValueError(f'Dev/test leakage through {field}')
    for field in ('input_sha256','feature_content_sha256'):
        incoming={r.get('input_provenance',{}).get(field) for r in rows} - {None}
        if incoming & set(artifact.get('dev_'+field,[])):
            raise ValueError('Dev/test leakage through ' + field)
    a,b=float(artifact['slope']),float(artifact['intercept'])
    if not np.isfinite([a,b]).all() or a <= 0: raise ValueError('Invalid monotonic calibration parameters')
    threshold=artifact['critic_threshold_selection']['threshold']
    result=[]
    for r in rows:
        p=float(np.clip(r['p_error'],1e-6,1-1e-6));c=float(expit(a*np.log(p/(1-p))+b))
        result.append({**r,'p_error_calibrated':c,'p_valid_calibrated':1-c,
                       'calibrated_error_flag':None if threshold is None else c >= threshold,
                       'calibration_sha256':canonical_hash(artifact)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','apply']);p.add_argument('predictions',type=Path)
    p.add_argument('--calibration',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    rows=[json.loads(s) for s in args.predictions.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if args.mode=='fit':
        result=fit_calibration(rows);result['input_file_sha256']=sha256(args.predictions)
        text=json.dumps(result,indent=2,allow_nan=False)
    else:
        if args.calibration is None:p.error('--calibration required')
        result=apply_calibration(rows,json.loads(args.calibration.read_text(encoding='utf-8')))
        text=''.join(json.dumps(r,allow_nan=False)+'\n' for r in result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:f.write(text)
    print(f'{args.mode}: wrote {args.output}; no real-transfer conclusion implied')


if __name__=='__main__':main()
