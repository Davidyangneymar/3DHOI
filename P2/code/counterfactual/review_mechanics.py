"""Objective clip statistics for navigation only; never generate semantic labels."""
import hashlib
import numpy as np


def summarize_arrays(distances, translation, start, end):
    d=np.asarray(distances,dtype=float);t=np.asarray(translation,dtype=float)
    if d.ndim!=2 or d.shape[1]!=2 or t.shape!=(len(d),3):
        raise ValueError('Expected two hand distances and xyz translation per frame')
    if type(start)is not int or type(end)is not int or not 0<=start<=end<len(d):
        raise ValueError('Invalid inclusive evidence range')
    if not np.isfinite(d).all() or not np.isfinite(t).all() or (d<0).any():
        raise ValueError('Nonfinite geometry or negative distance')
    clip=d[start:end+1]
    steps=np.linalg.norm(np.diff(t[start:end+1],axis=0),axis=1)
    return {'version':'objective_clip_v1','range':{'start':start,'end':end},'frame_count':len(clip),
            'hand_median_distance_m':np.median(clip,axis=0).tolist(),
            'both_hands_far_frames':int((clip>=.05).all(axis=1).sum()),
            'object_translation_path_m':float(steps.sum()),
            'object_max_translation_step_m':float(steps.max()) if len(steps) else 0.,
            'definition':'nearest-mesh-vertex distances; >=0.05m is geometric only; obj_trans translation within inclusive range; no force/rotation/support inference',
            'semantic_label_generated':False}


def mechanics_for(entry,path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=entry['sha256']:raise ValueError('Frozen sample hash mismatch')
    f=entry['review_focus']
    with np.load(path,allow_pickle=False) as z:
        if len(z['distances_m'])!=entry['frames']:raise ValueError('Frame count mismatch')
        result=summarize_arrays(z['distances_m'],z['obj_trans'],f['start'],f['end'])
    return {**result,'sample_sha256_verified_at_preparation':digest}
