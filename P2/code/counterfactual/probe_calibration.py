"""Historical synthetic diagnostic only: can calibration alone reduce old false alarms?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from counterfactual.benchmark import binary_metrics
from counterfactual.calibrate import fit_calibration, apply_calibration
from counterfactual.critic_api import CriticScorer, FEATURE_VERSION, sha256
from models.interaction_critic import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Output exists')
    entries=[json.loads(s) for s in (ROOT/'counterfactual/v1/index.jsonl').read_text().splitlines()]
    chosen=[e for e in entries if e['split'] in {'val','test'} and e['severity']!='high']
    arrays=[]
    for e in chosen:
        path=ROOT/'counterfactual/v1/samples'/f"{e['sample_id']}.npz"
        if sha256(path)!=e['sha256']:raise ValueError('Frozen sample hash changed')
        with np.load(path) as z:arrays.append(z['features'])
    torch.set_num_threads(2);scorer=CriticScorer();scores=scorer.score_many(arrays,feature_version=FEATURE_VERSION)
    rows=[{**r,'sample_id':e['sample_id'],'source_group':e['source_sequence'],'condition_id':e['source_sequence'],
           'split':'dev' if e['split']=='val' else 'test','label_error':e['label'],'data_domain':'synthetic_v1'}
          for e,r in zip(chosen,scores)]
    dev=[r for r in rows if r['split']=='dev'];test=[r for r in rows if r['split']=='test']
    calibration=fit_calibration(dev);calibrated=apply_calibration(test,calibration)
    y=[r['label_error'] for r in test]
    threshold=calibration['critic_threshold_selection']['threshold']
    rule_threshold=calibration['rule_threshold_selection']['threshold']
    report={'domain':'historical_synthetic_diagnostic','dev_n':len(dev),'test_n':len(test),
            'dataset_manifest_sha256':sha256(ROOT/'counterfactual/v1/manifest.json'),
            'checkpoint_sha256':scorer.checkpoint_sha256,
            'zero_shot_historical':binary_metrics(y,[r['p_error'] for r in test],scorer.threshold),
            'dev_calibrated_historical':None if threshold is None else binary_metrics(y,[r['p_error_calibrated'] for r in calibrated],threshold),
            'fixed_rule':binary_metrics(y,[r['rule_score'] for r in test],1.),
            'dev_threshold_rule':None if rule_threshold is None else binary_metrics(y,[r['rule_score'] for r in test],rule_threshold),
            'real_transfer_verified':False,'fresh_test':False,'human_semantic_labels_verified':False,
            'limitations':['Existing v1 labels only; semantic reliability remains unverified',
                           'Old validation already selected model and original threshold; calibration fit is exploratory',
                           'Old test already exposed; not new blind test evidence',
                           'No retraining or modification of v1 weights; synthetic calibration cannot be applied to real_generator domain']}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'calibration_synthetic_only.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    (args.output/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
