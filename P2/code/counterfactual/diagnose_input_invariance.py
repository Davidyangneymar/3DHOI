"""Read-only metamorphic checks of raw geometry -> geom13 -> frozen scores.

Variants live only in a temporary directory. This is numerical invariance testing,
not new physical/semantic labels, real generator transfer, or threshold tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from counterfactual.critic_api import CHANNELS, CriticScorer, FEATURE_VERSION, sha256
from counterfactual.generator_adapter import REFERENCE_FPS, load_features, mesh_paths, normalize_time
from models.interaction_critic import ROOT


VARIANTS = (
    {'name': 'translate_xy', 'kind': 'rigid', 'yaw_degrees': 0., 'translation_m': [2., -1.5, 0.]},
    {'name': 'yaw_37', 'kind': 'rigid', 'yaw_degrees': 37., 'translation_m': [0., 0., 0.]},
    {'name': 'yaw_90', 'kind': 'rigid', 'yaw_degrees': 90., 'translation_m': [0., 0., 0.]},
    {'name': 'yaw_minus73', 'kind': 'rigid', 'yaw_degrees': -73., 'translation_m': [0., 0., 0.]},
    {'name': 'encode_60fps', 'kind': 'time', 'factor': 2},
    {'name': 'encode_120fps', 'kind': 'time', 'factor': 4},
)
# Numerical tolerances fixed before executing any dataset comparison.
LIMITS = {'continuous_feature_abs': 2e-5, 'p_error_abs': 1e-5, 'rule_score_abs': 1e-4,
          'contact_bit_changes': 0, 'decision_changes': 0}


def rigid_transform(sample, yaw_degrees, translation):
    """Move human and EVERY object part in one right-handed Z-up world frame."""
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,) or not np.isfinite(t).all() or not np.isfinite(yaw_degrees):
        raise ValueError('Finite yaw and 3-vector translation required')
    q = Rotation.from_euler('z', yaw_degrees, degrees=True).as_matrix()
    result = {}
    for key, value in sample.items():
        a = np.asarray(value, dtype=np.float64)
        if key == 'joints' or key.endswith('_trans'):
            result[key] = a @ q.T + t
        elif key.endswith('_rot_mat'):
            result[key] = q @ a
        elif key.endswith('_scale'):
            result[key] = a.copy()
        else:
            raise ValueError('Unsupported raw array: ' + key)
    return result


def reencode_integer_fps(sample, factor):
    """Same duration; every original frame survives at index factor*i.

    Positions/scales are linearly interpolated; rotations use SO(3) interpolation.
    This cannot establish recovery of information lost by real low-FPS capture.
    """
    if type(factor) is not int or factor < 1:
        raise ValueError('Positive integer rate factor required')
    n = len(sample['joints'])
    if n < 2:
        raise ValueError('At least two original frames required')
    old = np.arange(n, dtype=float) / REFERENCE_FPS
    new = np.arange((n-1)*factor+1, dtype=float) / (REFERENCE_FPS*factor)
    output = {}
    for key, value in sample.items():
        a = np.asarray(value, dtype=np.float64)
        if key.endswith('_rot_mat'):
            interpolated = Slerp(old, Rotation.from_matrix(a))(new).as_matrix()
        else:
            flat = a.reshape(n, -1)
            interpolated = np.stack([np.interp(new, old, flat[:,i]) for i in range(flat.shape[1])], axis=1).reshape((len(new),)+a.shape[1:])
        # Preserve the observed anchors, including original finite-precision rotations.
        interpolated[::factor] = a
        output[key] = interpolated
    return output


def feature_delta(reference, candidate):
    if reference.shape != candidate.shape:
        return {'shape_equal': False, 'reference_shape': list(reference.shape),
                'candidate_shape': list(candidate.shape), 'within_tolerance': False}
    difference = np.abs(reference.astype(np.float64)-candidate.astype(np.float64))
    continuous = [i for i in range(13) if i not in (2,3)]
    maxima = difference.max(axis=0)
    bits = int(np.count_nonzero(reference[:,2:4] != candidate[:,2:4]))
    frame, channel = np.unravel_index(np.argmax(difference), difference.shape)
    return {'shape_equal': True, 'max_abs_by_channel': dict(zip(CHANNELS, map(float, maxima))),
            'max_continuous_abs': float(maxima[continuous].max()), 'contact_bit_changes': bits,
            'largest_difference_original_reference_frame': int(frame), 'largest_difference_channel': CHANNELS[channel],
            'within_tolerance': bool(maxima[continuous].max() <= LIMITS['continuous_feature_abs'] and bits == 0)}


def score_delta(reference, candidate):
    delta = {'p_error_abs': abs(reference['p_error']-candidate['p_error']),
             'rule_score_abs': abs(reference['rule_score']-candidate['rule_score']),
             'critic_decision_changed': reference['error_flag'] != candidate['error_flag'],
             'rule_decision_changed': reference['rule_flag'] != candidate['rule_flag']}
    delta['within_tolerance'] = (delta['p_error_abs'] <= LIMITS['p_error_abs']
                               and delta['rule_score_abs'] <= LIMITS['rule_score_abs']
                               and not delta['critic_decision_changed'] and not delta['rule_decision_changed'])
    return delta


def metadata(name, fps=30.):
    return {'input_kind':'raw', 'object':name, 'units':'metres', 'coordinate_system':'right_handed_z_up',
            'joint_schema':'omomo24', 'rotation_convention':'local_to_world_column', 'mesh_frame':'omomo_original_local',
            'fps':fps, 'accept_inferred_reference_fps':True,
            'mesh_sha256':{p.name:sha256(p) for p in mesh_paths(name)}}


def aggregate(records):
    output = {}
    for variant in ('cached_vs_raw', *(v['name'] for v in VARIANTS)):
        rows = [r for r in records if r['variant'] == variant]
        completed = [r for r in rows if 'error' not in r]
        feature_rows = [r for r in completed if r['feature_delta']['shape_equal']]
        score_rows = [r for r in completed if 'score_delta' in r]
        output[variant] = {'attempted':len(rows), 'completed':len(completed),
            'load_errors':len(rows)-len(completed),
            'feature_failures':sum(not r['feature_delta']['within_tolerance'] for r in completed),
            'score_failures':sum(not r['score_delta']['within_tolerance'] for r in score_rows),
            'max_continuous_feature_abs':max((r['feature_delta']['max_continuous_abs'] for r in feature_rows),default=None),
            'contact_bit_changes':sum(r['feature_delta']['contact_bit_changes'] for r in feature_rows),
            'max_p_error_abs':max((r['score_delta']['p_error_abs'] for r in score_rows),default=None),
            'max_rule_score_abs':max((r['score_delta']['rule_score_abs'] for r in score_rows),default=None),
            'critic_decision_changes':sum(r['score_delta']['critic_decision_changed'] for r in score_rows),
            'rule_decision_changes':sum(r['score_delta']['rule_decision_changed'] for r in score_rows)}
    return output


def plot_results(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(summary)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,key,limit,title in [(axes[0],'max_continuous_feature_abs',LIMITS['continuous_feature_abs'],'Max absolute continuous-feature difference'),
                               (axes[1],'max_p_error_abs',LIMITS['p_error_abs'],'Max absolute p_error difference')]:
        values=[max(summary[n][key] or 0,1e-12) for n in names]
        ax.barh(names,values,color=['#5b7185']+['#388795']*6)
        ax.set_xscale('log');ax.axvline(limit,color='#b43838',ls='--',label='Preset tolerance')
        ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen v1 raw-adapter diagnostics (zero displayed at 1e-12; no threshold tuning)',fontsize=11)
    fig.savefig(destination,dpi=160);plt.close(fig)


def run(output, root=ROOT):
    if output.exists():
        raise ValueError('Choose a new report directory; historical evidence is immutable')
    parent = root/'counterfactual/v1'
    entries = [json.loads(s) for s in (parent/'index.jsonl').read_text().splitlines()]
    manifest = json.loads((parent/'manifest.json').read_text())
    if manifest['index_sha256'] != sha256(parent/'index.jsonl'):
        raise ValueError('Frozen parent index changed')
    torch.set_num_threads(2); scorer=CriticScorer()
    watched = [root/'models/interaction_critic_v1.pt', parent/'manifest.json', parent/'index.jsonl',
               root/'counterfactual/critic_api.py', root/'counterfactual/generator_adapter.py',
               root/'counterfactual/geometry_features.py', root/'omomo_contact.py']
    before={str(p.relative_to(root)):sha256(p) for p in watched}
    output.mkdir(parents=True)
    protocol={'scope':'numerical_invariance_of_frozen_synthetic_v1_not_real_transfer',
              'dataset_samples':len(entries),'variants':VARIANTS,'limits_fixed_before_run':LIMITS,
              'checkpoint_sha256':scorer.checkpoint_sha256,'threshold_error':scorer.threshold,
              'reference_fps':REFERENCE_FPS,'reference_fps_status':'inferred',
              'normalization_unchanged':True,'new_human_labels':0,'training':False,
              'rotation_axis':'world Z','coordinates_transform_all_parts':True,
              'time_reencoding':'integer upsampling retaining original anchors; not independent recapture',
              'watched_sha256':before,'diagnostic_script_sha256':sha256(Path(__file__))}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[];original_scores=[];aligned_geometry=[];metadata_cache={}
    with tempfile.TemporaryDirectory(prefix='hoi_invariance_') as temp:
        variant_path=Path(temp)/'variant.npz'
        for index, entry in enumerate(entries,1):
            source=parent/'samples'/(entry['sample_id']+'.npz')
            if sha256(source)!=entry['sha256']:
                raise ValueError('Frozen sample changed: '+entry['sample_id'])
            base={'sample_id':entry['sample_id'],'object':entry['object'],'type':entry['type'],
                  'split':entry['split'],'sample_sha256':entry['sha256']}
            with np.load(source,allow_pickle=False) as z:
                sample={k:np.asarray(z[k],dtype=np.float64) for k in z.files if k=='joints' or k.endswith(('_trans','_rot_mat','_scale'))}
                cached=z['features'].copy()
            try:
                if entry['object'] not in metadata_cache:metadata_cache[entry['object']]=metadata(entry['object'])
                meta=metadata_cache[entry['object']]
                raw,info=load_features(source,meta)
            except (ValueError,KeyError,OSError) as error:
                for name in ('cached_vs_raw',*(v['name'] for v in VARIANTS)):
                    records.append({**base,'variant':name,'error':'base_input: '+str(error)})
                continue
            arrays=[cached,raw];details=[{**base,'variant':'cached_vs_raw','feature_delta':feature_delta(cached,raw)}]
            descriptors=[]
            for variant in VARIANTS:
                if variant['kind']=='rigid':
                    changed=rigid_transform(sample,variant['yaw_degrees'],variant['translation_m']);fps=REFERENCE_FPS
                else:
                    changed=reencode_integer_fps(sample,variant['factor']);fps=REFERENCE_FPS*variant['factor']
                    aligned=normalize_time(changed,fps)
                    aligned_geometry.append({'sample_id':entry['sample_id'],'variant':variant['name'],
                        'original_frames':len(sample['joints']),'encoded_frames':len(changed['joints']),
                        'aligned_frames':len(aligned['joints']),
                        'max_abs_by_array':{k:float(np.max(np.abs(sample[k]-aligned[k]))) for k in sample}})
                np.savez(variant_path,**changed)
                try:
                    candidate,_=load_features(variant_path,{**meta,'fps':fps})
                except (ValueError,KeyError,OSError) as error:
                    records.append({**base,'variant':variant['name'],'error':str(error)});continue
                arrays.append(candidate)
                descriptors.append({**base,'variant':variant['name'],'feature_delta':feature_delta(raw,candidate)})
            scores=scorer.score_many(arrays,feature_version=FEATURE_VERSION)
            original_scores.append({**base,**scores[0]})
            details[0].update(score_delta=score_delta(scores[0],scores[1]),baseline_p_error=scores[0]['p_error'],candidate_p_error=scores[1]['p_error'])
            for detail,score in zip(descriptors,scores[2:]):
                detail.update(score_delta=score_delta(scores[1],score),baseline_p_error=scores[1]['p_error'],candidate_p_error=score['p_error'])
            records.extend(details+descriptors)
            if index%50==0:print(f'Checked {index}/{len(entries)}',flush=True)
    for p in watched:
        if sha256(p)!=before[str(p.relative_to(root))]:raise ValueError('Input/core file changed during run: '+str(p))
    summary=aggregate(records)
    failures=[r for r in records if 'error' in r or not r['feature_delta']['within_tolerance'] or not r['score_delta']['within_tolerance']]
    result={'status':'numerical_checks_passed' if not failures else 'differences_require_diagnosis',
            'sample_count':len(entries),'sample_objects':dict(Counter(e['object'] for e in entries)),
            'comparisons':len(records),'failed_comparisons':len(failures),'by_variant':summary,
            'raw_baseline_score_count':len(original_scores),'core_unchanged':True,
            'synthetic_review_budget_consumed':0,'real_review_budget_consumed':0,
            'real_generator_transfer_verified':False,'semantic_validity_verified':False,
            'limits':LIMITS,'failed_examples':failures}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (output/'comparisons.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    (output/'time_geometry_recovery.json').write_text(json.dumps(aligned_geometry,indent=2,allow_nan=False),encoding='utf-8')
    plot_results(summary,output/'comparison_summary.png')
    lines=['# 冻结 Critic 输入等价变换诊断','',
        '日期：2026-10-05。只读使用冻结 v1、固定模型及阈值；临时变换不写回数据集，不产生标签或训练。',
        f"覆盖 {len(entries)} 条、{len(result['sample_objects'])} 种物体，共 {len(records)} 项比较；超出预设数值容差／载入失败 {len(failures)} 项。",'',
        '| 比较 | 数量 | 特征失败 | 最大连续特征差 | 最大 p_error 差 | Critic 判定变化 | 规则判定变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f"| {name} | {s['attempted']} | {s['feature_failures']} | {s['max_continuous_feature_abs']} | {s['max_p_error_abs']} | {s['critic_decision_changes']} | {s['rule_decision_changes']} |")
    lines+=['','连续通道容差 2e-5、p_error 1e-5、规则分数 1e-4；接触位与最终判定要求不变。这些容差在运行前固定，只用于数值诊断，不是调整模型阈值。',
        '', '![差异汇总](comparison_summary.png)', '',
        '## 解释边界', '',
        '- 所有人体关节和物体各部件同时变换；旋转绕世界 Z 轴，不改变重力方向。网格局部坐标及哈希不变。',
        '- 60/120 FPS 由原 30 FPS 序列插值并保留原采样点，再走严格 raw 适配回到参考 30 FPS；验证表示等价路径，不等于原生不同相机／动捕帧率，也不验证低帧率下的信息恢复。',
        '- 最近顶点是离散查询；接触使用硬阈值。旋转矩阵投影、等距顶点与阈值附近浮点变化可能放大差异；失败不会自动改标签或阈值。',
        '- 即使所有检查通过，也只能排除这些坐标／整数倍时间编码差异，不证明物理接触真值、负例语义、真实生成迁移或未见组合收益。',
        '- 原始 30 FPS 仍为 OMOMO 发布数据的推断。固定阈值不因本报告改变，历史误报仍按原报告保留。',
        '', '复现：`python -m counterfactual.diagnose_input_invariance --output <新报告目录>`。详细比较见 `comparisons.jsonl`，输入和代码哈希、预设参数见 `protocol.json`。']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='failed_examples'},indent=2))


if __name__=='__main__':main()
