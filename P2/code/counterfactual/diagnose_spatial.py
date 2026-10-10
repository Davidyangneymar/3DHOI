"""Read-only spatial perturbation diagnostics; geometric flags are not labels."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(original, changed, distances, params):
    a=np.asarray(original);b=np.asarray(changed);d=np.asarray(distances)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(24,3) or d.shape!=(len(a),2):
        raise ValueError('Invalid joint/distance shape')
    if not all(np.isfinite(v).all() for v in (a,b,d)):raise ValueError('Nonfinite geometry')
    hand={'left':0,'right':1}.get(params['hand'])
    if hand is None:raise ValueError('Unknown shifted hand')
    cut=params['cut_frame'];start=cut+params['ramp_frames']
    if not 0<=cut<=start<len(a):raise ValueError('Invalid intervention window')
    near=d[start:]<.05
    limb=(16+hand,18+hand,20+hand,22+hand)
    differences={}
    for name,parent,child in [('shoulder_to_elbow',limb[0],limb[1]),('elbow_to_wrist',limb[1],limb[2]),('wrist_to_hand',limb[2],limb[3])]:
        old=np.linalg.norm(a[:,child]-a[:,parent],axis=-1)
        new=np.linalg.norm(b[:,child]-b[:,parent],axis=-1)
        differences[name]=float(np.max(np.abs(new-old)))
    delta=np.linalg.norm(b-a,axis=-1)
    changed_ids=np.flatnonzero(delta.max(axis=0)>1e-6).tolist()
    either=float(near.any(axis=1).mean());shifted=float(near[:,hand].mean())
    return {'shifted_hand':params['hand'],'post_ramp_start':start,'post_ramp_frames':len(near),
      'either_hand_near_fraction':either,'shifted_hand_near_fraction':shifted,
      'other_hand_near_fraction':float(near[:,1-hand].mean()),
      'changed_joint_ids':changed_ids,'max_shifted_hand_displacement_m':float(delta[:,22+hand].max()),
      'max_other_hand_displacement_m':float(delta[:,23-hand].max()),
      'arm_segment_max_length_change_m':differences,
      'either_gate_passes_but_shifted_hand_below_0_7':either>=.7 and shifted<.7,
      'upper_arm_or_forearm_change_over_0_10m':max(differences['shoulder_to_elbow'],differences['elbow_to_wrist'])>.10,
      'semantic_label_generated':False}


def run(root,reviews,status_path,output):
    if output.exists():raise ValueError('Output exists; preserve prior diagnosis')
    status=json.loads(status_path.read_text(encoding='utf-8'))
    if sha(reviews)!=status['source_reviews_sha256']:raise ValueError('Latest review hash mismatch')
    frozen=root/'counterfactual/v1'
    manifest=json.loads((frozen/'manifest.json').read_text())
    if sha(frozen/'index.jsonl')!=manifest['index_sha256']:raise ValueError('Frozen index hash mismatch')
    entries=[json.loads(s) for s in (frozen/'index.jsonl').read_text().splitlines()]
    by_id={e['sample_id']:e for e in entries}
    votes=[json.loads(s) for s in reviews.read_text(encoding='utf-8').splitlines()]
    primary={r['sample_id']:r for r in votes if r['round']=='primary'}
    result=[]
    for e in entries:
        if e['type']!='spatial_contact_shift':continue
        clean_id=e['source_sequence']+'__clean__none'
        path=frozen/'samples'/f"{e['sample_id']}.npz";clean=frozen/'samples'/f'{clean_id}.npz'
        if sha(path)!=e['sha256'] or sha(clean)!=by_id[clean_id]['sha256']:raise ValueError('Sample hash mismatch')
        with np.load(path,allow_pickle=False) as z,np.load(clean,allow_pickle=False) as c:
            metrics=measure(c['joints'],z['joints'],z['distances_m'],e['parameters'])
            metrics['object_pose_arrays_unchanged']=all(np.array_equal(z[k],c[k]) for k in z.files if k.startswith('obj'))
        result.append({'sample_id':e['sample_id'],'source_sequence':e['source_sequence'],'object':e['object'],
          'severity':e['severity'],'sample_sha256':e['sha256'],'clean_sha256':by_id[clean_id]['sha256'],
          'parameters':e['parameters'],'primary_review':primary.get(e['sample_id']),
          'secondary_reviews':[r for r in votes if r['sample_id']==e['sample_id'] and r['round']=='secondary'],
          'geometry':metrics})
    reviewed=[r for r in result if r['primary_review']]
    summary={'scope':'142 frozen spatial variants; geometry only; observed labels from current 7 reviewed variants',
      'sample_count':len(result),'reviewed_count':len(reviewed),
      'reviewed_decisions':dict(Counter(r['primary_review']['decision'] for r in reviewed)),
      'builder_matches_frozen_manifest':sha(root/'counterfactual/build_v1.py')==manifest['builder_sha256'],
      'either_hand_gate_pass_count':sum(r['geometry']['either_hand_near_fraction']>=.7 for r in result),
      'shifted_hand_below_0_7_despite_either_gate_count':sum(r['geometry']['either_gate_passes_but_shifted_hand_below_0_7'] for r in result),
      'arm_segment_change_over_0_10m_count':sum(r['geometry']['upper_arm_or_forearm_change_over_0_10m'] for r in result),
      'object_pose_unchanged_count':sum(r['geometry']['object_pose_arrays_unchanged'] for r in result),
      'unshifted_hand_unchanged_count':sum(r['geometry']['max_other_hand_displacement_m']<1e-6 for r in result),
      'max_segment_changes_m':{key:max(r['geometry']['arm_segment_max_length_change_m'][key] for r in result) for key in ['shoulder_to_elbow','elbow_to_wrist','wrist_to_hand']},
      'input_reviews_sha256':sha(reviews),'semantic_status_sha256':sha(status_path),
      'frozen_manifest_sha256':sha(frozen/'manifest.json'),'frozen_index_sha256':sha(frozen/'index.jsonl'),
      'builder_sha256':sha(root/'counterfactual/build_v1.py'),'diagnostic_code_sha256':sha(Path(__file__)),
      'new_human_labels':0,'human_budget_consumed':0,'training_started':False,'data_modified':False}
    output.mkdir(parents=True)
    for name,data in [('summary.json',summary),('all_spatial_geometry.json',result),('reviewed_spatial_cases.json',reviewed)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary,reviewed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s,reviewed=run(Path(__file__).resolve().parents[1],a.reviews,a.status,a.output)
    print(json.dumps(s,ensure_ascii=False,indent=2))
    for r in reviewed:
        g=r['geometry'];print(r['sample_id'],r['primary_review']['decision'],'shifted_near=',round(g['shifted_hand_near_fraction'],3),'other_near=',round(g['other_hand_near_fraction'],3))
