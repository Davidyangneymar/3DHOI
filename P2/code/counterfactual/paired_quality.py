"""Evaluate fixed-pool selections with independent labels and a frozen C protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from counterfactual.benchmark import select_candidates, validate_manifest
from counterfactual.protocol_checks import canonical_hash, require_text, valid_error_label

METHODS = ('baseline_first', 'random', 'rule', 'critic')


def pool_identity(rows):
    return [{k:r[k] for k in ('sample_id','condition_id','source_group','split','candidate_index')}
            | {'input_sha256':r['input_provenance']['input_sha256']}
            for r in sorted(rows,key=lambda x:x['sample_id'])]


def paired_interval(values, groups, repeats=2000, seed=20261001):
    values,groups=np.asarray(values,dtype=float),np.asarray(groups)
    if values.shape!=groups.shape or values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Finite paired observations and matching groups required')
    names=np.unique(groups)
    mean=float(values.mean()) if len(values) else None
    if len(names)<2:return {'mean_difference':mean,'low':None,'high':None,'group_count':len(names),'n':len(values)}
    rng=np.random.default_rng(seed);indices={g:np.flatnonzero(groups==g) for g in names}
    boot=[float(values[np.concatenate([indices[g] for g in rng.choice(names,len(names),replace=True)])].mean()) for _ in range(repeats)]
    return {'mean_difference':mean,'low':float(np.quantile(boot,.025)), 'high':float(np.quantile(boot,.975)),
            'group_count':len(names),'n':len(values),'repeats':repeats,'seed':seed}


def evaluate_quality(rows, annotations, protocol, repeats=2000):
    validate_manifest(rows)
    if not rows or any(r['split']!='test' for r in rows):raise ValueError('Independent quality evaluation accepts TEST ONLY')
    if protocol.get('status')!='frozen':raise ValueError('C quality protocol is not frozen')
    for key in ('protocol_id','owner','label_protocol','frozen_at'):
        require_text(protocol.get(key),key)
    if protocol.get('candidate_pool_sha256')!=canonical_hash(pool_identity(rows)):
        raise ValueError('Candidate pool does not match frozen C protocol')
    metrics=protocol.get('quality_metrics')
    if not isinstance(metrics,list) or not metrics:raise ValueError('C must define at least one independent quality guardrail')
    if len({m.get('name') for m in metrics})!=len(metrics):raise ValueError('Duplicate metric name')
    for m in metrics:
        require_text(m.get('name'),'metric name')
        if m.get('direction') not in {'lower','higher'} or not isinstance(m.get('margin'),(float,int)) or not np.isfinite(m['margin']) or m['margin']<0:
            raise ValueError('Metric requires lower/higher direction and finite nonnegative noninferiority margin')
    minimum=protocol.get('min_independent_groups')
    if type(minimum) is not int or minimum<2:raise ValueError('C must set min_independent_groups >= 2')
    candidates={r['sample_id']:r for r in rows};labels={}
    for r in annotations:
        sid=r.get('sample_id')
        if sid not in candidates or sid in labels:raise ValueError('Unknown or duplicate annotation sample')
        if not valid_error_label(r.get('label_error')):raise ValueError('Invalid independent label')
        if r.get('label_source')!='independent_blind':raise ValueError('Independent blind label provenance required')
        require_text(r.get('reviewer'),'reviewer')
        if r.get('input_sha256')!=candidates[sid]['input_provenance']['input_sha256']:
            raise ValueError('Independent label refers to a different candidate file')
        labels[sid]=r
    if set(labels)!=set(candidates):raise ValueError('All pool candidates need independent annotation, including uncertain/null')
    selections=select_candidates(rows)
    accepted=[];excluded=[]
    missing_by_method={method:0 for method in METHODS}
    for selection in selections:
        for method in METHODS:
            if labels[selection[method]].get('label_error') not in (0,1):missing_by_method[method]+=1
        if any(labels[selection[m]].get('label_error') not in (0,1) for m in METHODS):
            excluded.append(selection['condition_id'])
        else:accepted.append(selection)
    groups=[]
    for s in accepted:
        gs={candidates[s[m]]['source_group'] for m in METHODS}
        if len(gs)!=1:raise ValueError('Candidates for one condition must share source_group')
        groups.append(next(iter(gs)))
    rates={m:float(np.mean([labels[s[m]]['label_error'] for s in accepted])) if accepted else None for m in METHODS}
    comparisons={}
    for comparator in ('baseline_first','random','rule'):
        difference=[labels[s['critic']]['label_error']-labels[s[comparator]]['label_error'] for s in accepted]
        comparisons[comparator]=paired_interval(difference,groups,repeats)
    guardrails={}
    for metric in metrics:
        name=metric['name'];deltas=[];missing=[]
        for s in accepted:
            a=labels[s['critic']].get('quality',{}).get(name)
            b=labels[s['baseline_first']].get('quality',{}).get(name)
            if not isinstance(a,(float,int)) or not isinstance(b,(float,int)) or not np.isfinite([a,b]).all():
                missing.append(s['condition_id']);continue
            deltas.append((a-b)*(1 if metric['direction']=='lower' else -1))
        if missing:
            guardrails[name]={'passed':False,'status':'missing_metric_values','missing_conditions':missing}
        else:
            interval=paired_interval(deltas,groups,repeats)
            guardrails[name]={**interval,'margin':metric['margin'],'difference_definition':'positive means worse',
                              'passed':interval['high'] is not None and interval['high']<=metric['margin']}
    primary=comparisons['baseline_first'];enough=primary['group_count']>=minimum
    supported=bool(not excluded and enough and primary['high'] is not None and primary['high']<0 and all(g['passed'] for g in guardrails.values()))
    return {'protocol_sha256':canonical_hash(protocol),'candidate_pool_sha256':canonical_hash(pool_identity(rows)),
            'conditions_total':len(selections),'paired_complete_conditions':len(accepted),'excluded_conditions':excluded,
            'uncertain_or_unlabelled_selected_by_method':missing_by_method,'error_rates_common_conditions':rates,
            'critic_minus_comparator_error_rate':comparisons,'quality_guardrails':guardrails,
            'enough_independent_groups':enough,'reranking_benefit_supported':supported,
            'generator_training_improvement_verified':False,
            'limitation':'Complete-case analysis; any excluded condition blocks benefit claim. Provenance declarations are not independently authenticated.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path);p.add_argument('annotations',type=Path);p.add_argument('protocol',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    result=evaluate_quality(read(args.predictions),read(args.annotations),json.loads(args.protocol.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(f'Wrote {args.output}')


if __name__=='__main__':main()
