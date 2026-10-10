"""Verify new inference against all historical test predictions without retraining."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from counterfactual.critic_api import CriticScorer, FEATURE_VERSION, sha256
from models.interaction_critic import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('Output exists')
    entries=[json.loads(s) for s in (ROOT/'counterfactual/v1/index.jsonl').read_text().splitlines()]
    historical=json.loads((ROOT/'reports/critic_metrics.json').read_text())
    if historical['dataset_manifest_sha256'] != sha256(ROOT/'counterfactual/v1/manifest.json'):
        raise ValueError('Historical report refers to a different dataset')
    torch.set_num_threads(2);scorer=CriticScorer()
    arrays=[]
    for e in entries:
        with np.load(ROOT/'counterfactual/v1/samples'/f"{e['sample_id']}.npz") as z: arrays.append(z['features'])
    scores=scorer.score_many(arrays,feature_version=FEATURE_VERSION)
    predictions={e['sample_id']:s for e,s in zip(entries,scores)}
    errors=[abs(predictions[r['sample_id']]['p_error']-r['critic_score']) for r in historical['test_predictions']]
    if max(errors)>1e-6: raise ValueError(f'Legacy parity failed: {max(errors)}')
    false_positives=[{**e,**s} for e,s in zip(entries,scores) if e['split']=='test' and e['label']==0 and s['error_flag']]
    snapshot=ROOT/'artifacts/baseline_v1_20261001/snapshot.json'
    hashes=json.loads(snapshot.read_text())['sha256']
    changed=[name for name,digest in hashes.items() if sha256(ROOT/name)!=digest]
    if changed: raise ValueError(f'Baseline changed: {changed}')
    report={'status':'passed','scored_samples':len(entries),'historical_test_predictions':len(errors),
            'max_probability_difference':max(errors),'checkpoint_sha256':scorer.checkpoint_sha256,
            'baseline_snapshot_unchanged':True,'held_out_clean_count':sum(e['split']=='test' and e['label']==0 for e in entries),
            'held_out_clean_false_positives':len(false_positives),
            'false_positive_by_object':dict(Counter(e['object'] for e in false_positives)),
            'false_positive_examples':false_positives,
            'limitation':'Historical synthetic regression only; not fresh test evidence, calibration or real transfer'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='false_positive_examples'},indent=2))


if __name__=='__main__':main()
