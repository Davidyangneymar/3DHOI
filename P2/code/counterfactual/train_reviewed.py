"""CPU training gated on a reviewed freeze. Does not select models using test scores."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from counterfactual.benchmark import binary_metrics
from counterfactual.audit_reviews import audit as recompute_audit
from counterfactual.critic_api import FEATURE_VERSION, sha256, validate_features
from counterfactual.review_integrity import load_review_status, validate_reviewed_index
from counterfactual.reviewed_subset import select_reviewed_subset, training_eligibility, validate_parent_queue
from models.interaction_critic import InteractionCritic, resample, threshold_from_validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261002, 20261003])
    args = p.parse_args()
    if args.output.exists() or len(set(args.seeds)) != len(args.seeds):
        p.error("Use a new output directory and unique seeds")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    audit = json.loads((args.dataset / "audit.json").read_text())
    subset_mode = manifest.get('review_scope') == 'reviewed-subset'
    if manifest.get('review_scope', 'full') not in ('full', 'reviewed-subset'):
        p.error('Unknown review scope')
    if (not subset_mode and not audit.get("ready_for_curated_freeze")) or manifest.get("feature_version") != FEATURE_VERSION:
        p.error("Dataset has not passed the semantic review gate")
    if 'review_status_sha256' not in manifest:
        p.error('Dataset semantic review gate lacks hash-bound semantic status')
    for file, key in [('index.jsonl','index_sha256'),('audit.json','audit_sha256'),('review_records.jsonl','review_records_sha256'),('review_status.json','review_status_sha256'),('queue.json','queue_sha256'),('adjudications.jsonl','adjudications_sha256')]:
        if sha256(args.dataset / file) != manifest[key]:
            p.error(f"Hash mismatch: {file}")
    def read_jsonl(name):return [json.loads(s) for s in (args.dataset/name).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    status=load_review_status(args.dataset/'review_status.json',args.dataset/'review_records.jsonl')
    actual_audit=recompute_audit(json.loads((args.dataset/'queue.json').read_text(encoding='utf-8')),
                                 read_jsonl('review_records.jsonl'),read_jsonl('adjudications.jsonl'),status)
    if not subset_mode and not actual_audit['ready_for_curated_freeze']:p.error('Recomputed semantic review gate is closed')
    if actual_audit!=audit:p.error('Stored audit disagrees with actual review history')
    entries = [json.loads(s) for s in (args.dataset / "index.jsonl").read_text().splitlines()]
    selected_audit = actual_audit
    if subset_mode:
        selection = select_reviewed_subset(json.loads((args.dataset/'queue.json').read_text()), actual_audit)
        if sha256(args.dataset/'subset.json') != manifest.get('subset_sha256') or selection != json.loads((args.dataset/'subset.json').read_text()):
            p.error('Subset selection differs from full audit and evidence policy')
        selected_audit = selection
    validate_reviewed_index(entries,selected_audit)
    # Both release modes inherit the complete parent's split and review queue.
    # A smaller self-consistent audit must never redefine secondary obligations.
    from models.interaction_critic import ROOT
    parent = ROOT / 'counterfactual/v1'
    if sha256(parent/'manifest.json') != manifest.get('parent_manifest_sha256'):
        p.error('Frozen parent manifest mismatch')
    parent_manifest = json.loads((parent/'manifest.json').read_text())
    if sha256(parent/'index.jsonl') != parent_manifest['index_sha256']:
        p.error('Frozen parent index mismatch')
    validate_parent_queue(json.loads((args.dataset/'queue.json').read_text()),
                          [json.loads(s) for s in (parent/'index.jsonl').read_text().splitlines()])
    eligibility = training_eligibility(entries)
    if 'training_eligibility_sha256' in manifest:
        if sha256(args.dataset/'training_eligibility.json') != manifest['training_eligibility_sha256'] or eligibility != json.loads((args.dataset/'training_eligibility.json').read_text()):
            p.error('Training eligibility differs from actual reviewed samples')
        if manifest.get('training_ready') is not eligibility['training_ready']:
            p.error('Training readiness declaration differs from recomputed gate')
    if not eligibility['training_ready']:
        p.error('Training gate closed: ' + ', '.join(eligibility['blockers']))
    source_splits = {}
    arrays = []
    for e in entries:
        source_splits.setdefault(e['source_sequence'], set()).add(e['split'])
        path = args.dataset / 'samples' / (e['sample_id'] + '.npz')
        if sha256(path) != e['sha256']:
            p.error(f"Sample changed: {path}")
        with np.load(path, allow_pickle=False) as z:
            arrays.append(resample(validate_features(z['features'])))
    if any(len(s) != 1 for s in source_splits.values()):
        p.error("Source leakage")
    x, y = np.stack(arrays), np.array([e['label_error'] for e in entries])
    ids = {s: np.array([i for i,e in enumerate(entries) if e['split']==s and e['severity']!='high'],dtype=int) for s in ['train','val']}
    for split, indices in ids.items():
        if set(y[indices]) != {0,1}:
            p.error(f"{split} needs both classes after withholding high severity")
    mean = x[ids['train']].mean(axis=(0,1),keepdims=True)
    std = np.maximum(x[ids['train']].std(axis=(0,1),keepdims=True),1e-4)
    xt, yt = torch.tensor((x-mean)/std), torch.tensor(y,dtype=torch.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {'seeds':args.seeds,'epochs_max':100,'patience':15,'batch_size':32,'learning_rate':.001,
              'weight_decay':.005,'withheld_severity':'high','model_selection':'minimum validation loss',
              'threshold_selection':'maximum validation balanced accuracy','device':'cpu',
              'feature_version':FEATURE_VERSION,'dataset_manifest_sha256':sha256(args.dataset/'manifest.json'),
              'trainer_sha256':sha256(Path(__file__)), 'test_used_for_selection':False,
              'test_status':manifest.get('test_status'), 'unseen_pair_verified':False}
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    torch.set_num_threads(2)
    summaries=[]
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        net=InteractionCritic()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.005)
        train,val=ids['train'],ids['val']
        weight=torch.tensor([(y[train]==0).sum()/(y[train]==1).sum()],dtype=torch.float32)
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        best,best_state,stale=float('inf'),None,0
        history=[]
        for epoch in range(100):
            net.train()
            for batch in np.array_split(np.random.permutation(train),max(1,int(np.ceil(len(train)/32)))):
                optimizer.zero_grad(); loss=loss_fn(net(xt[batch]),yt[batch])
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward();optimizer.step()
            net.eval()
            with torch.no_grad(): vloss=loss_fn(net(xt[val]),yt[val]).item()
            if not np.isfinite(vloss): raise ValueError('Nonfinite validation loss')
            history.append({'epoch':epoch+1,'validation_loss':vloss})
            if vloss<best-1e-4:
                best=vloss;best_state={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
            else: stale+=1
            if stale>=15: break
        net.load_state_dict(best_state);net.eval()
        with torch.no_grad(): scores=torch.sigmoid(net(xt[val])).numpy()
        threshold=threshold_from_validation(y[val],scores)
        destination=args.output/f'seed_{seed}';destination.mkdir()
        torch.save({'state_dict':net.state_dict(),'mean':mean[0],'std':std[0],'threshold':threshold,
                    'channels':13,'seed':seed,'feature_version':FEATURE_VERSION,
                    'dataset_manifest_sha256':config['dataset_manifest_sha256']},destination/'critic.pt')
        result={'seed':seed,'best_validation_loss':best,'validation':binary_metrics(y[val],scores,threshold),
                'history':history,'checkpoint_sha256':sha256(destination/'critic.pt')}
        (destination/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        summaries.append({'seed':seed,'best_validation_loss':best,'threshold':threshold})
        print(f'seed={seed}: best_validation_loss={best:.6f}',flush=True)
    chosen=min(summaries,key=lambda r:(r['best_validation_loss'],r['seed']))
    (args.output/'selection.json').write_text(json.dumps({'selection':chosen,'all_seeds':summaries,
        'criterion':'validation_loss_only','real_transfer_verified':False},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
