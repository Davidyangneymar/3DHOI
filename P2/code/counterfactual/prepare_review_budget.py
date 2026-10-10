"""Freeze a bounded review plan, not a fully approved training dataset."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from counterfactual.audit_reviews import audit, required_secondaries
from counterfactual.review_mechanics import mechanics_for


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def focus(entry):
    last = entry['frames'] - 1
    cut = entry.get('parameters',{}).get('cut_frame',-1)
    ramp = entry.get('parameters',{}).get('ramp_frames',0)
    if entry['type']=='clean' or not isinstance(cut,int) or not 0<=cut<=last:
        return dict(start=0,end=last,method='full_sequence',description='无可用扰动边界，保留完整回放。')
    ramp = ramp if isinstance(ramp,int) and ramp>=0 else 0
    return dict(start=max(0,cut-12),end=min(last,cut+ramp+24),event_frame=cut,
                method='construction_boundary_with_context',description='依据扰动构造起点定位，含前 12 帧及过渡后的 24 帧上下文。')


def plan(entries, records, reserved):
    audit(entries,records)
    seen={r['sample_id'] for r in records if r['round']=='primary'}
    by_id={e['sample_id']:e for e in entries}
    if len(seen)!=14 or len(reserved)!=24 or len({e['sample_id'] for e in reserved})!=24:
        raise ValueError('This approved plan requires 14 reviewed and 24 reserved samples')
    if any(e['sample_id'] in seen or by_id.get(e['sample_id'])!=e for e in reserved):
        raise ValueError('Reserved batch repeats reviewed samples or differs from full queue')
    selected=list(reserved);used={e['sample_id'] for e in selected}|seen
    source_count=Counter(e['source_sequence'] for e in selected)
    object_count=Counter(e['object'] for e in selected)
    strata=[('clean','none')]+[(kind,severity) for kind in ('premature_release','object_motion_without_contact','spatial_contact_shift') for severity in ('low','medium','high')]
    for kind,severity in strata:
        options=[e for e in entries if e['sample_id'] not in used and e['type']==kind and e['severity']==severity]
        if not options:raise ValueError('Cannot fill approved quota without changing the plan')
        e=min(options,key=lambda e:(source_count[e['source_sequence']],object_count[e['object']],hashlib.sha256(('budget34|'+e['sample_id']).encode()).hexdigest()))
        selected.append(e);used.add(e['sample_id']);source_count[e['source_sequence']]+=1;object_count[e['object']]+=1
    assert len(selected)==34
    second=required_secondaries(entries)
    return [{**e,'second_review_required':e['sample_id'] in second or e.get('second_review_required',False),'review_focus':focus(e)} for e in selected]


def build(root):
    output=root/'outputs/semantic_review_budget34_20261003'
    if output.exists():raise ValueError('Budget plan exists; do not replace an issued manifest')
    original=root/'outputs/semantic_review_v11'
    batch=root/'outputs/semantic_review_batch02_20261003'
    reviews=root/'reviews/incoming/002_action_confirmation_20261003/combined_reviews_19.jsonl'
    receipt=json.loads(reviews.with_name('check_summary.json').read_text(encoding='utf-8'))
    if sha(reviews)!=receipt['updated_combined_sha256']:raise ValueError('Prior review hash changed')
    entries=json.loads((original/'queue.json').read_text(encoding='utf-8'))
    records=[json.loads(s) for s in reviews.read_text(encoding='utf-8').splitlines() if s.strip()]
    reserved=json.loads((batch/'queue.json').read_text(encoding='utf-8'))
    selected=plan(entries,records,reserved)
    for e in selected:
        if sha(root/'counterfactual/v1/samples'/f"{e['sample_id']}.npz")!=e['sha256']:raise ValueError('Frozen NPZ hash mismatch')
        e['review_mechanics']=mechanics_for(e,root/'counterfactual/v1/samples'/f"{e['sample_id']}.npz")
    payload=json.loads((original/'queue.js').read_text(encoding='utf-8').removeprefix('window.queue=').removesuffix(';'))
    output.mkdir()
    manifest={'scope':'budgeted_human_validation_subset_only','date_local':'2026-10-03',
      'previous_primary_count':14,'additional_primary_cap':34,'total_primary_cap':48,
      'automatic_refill':False,'uncertain_counts_toward_cap':True,'full_v1_approved':False,
      'eligible_for_existing_training_freeze':False,'remaining_outside_budget':len(entries)-48,
      'planned_additional_samples':[e['sample_id'] for e in selected],
      'required_secondary_ids':[e['sample_id'] for e in selected if e['second_review_required']],
      'secondary_work_is_additional_to_primary_budget':True,
      'input_reviews_sha256':sha(reviews),'full_queue_sha256':sha(original/'queue.json'),
      'preserved_batch02_queue_sha256':sha(batch/'queue.json'),
      'type_severity_counts':dict(Counter(e['type']+'/'+e['severity'] for e in selected)),
      'unique_sources':len({e['source_sequence'] for e in selected}),
      'objects':sorted({e['object'] for e in selected}),
      'selection_note':'Reuse issued 24, add 1 clean and 1 per perturbation/severity cell with source/object diversity. Not a random benchmark.',
      'batches':[]}
    for i,group in enumerate([selected[:10],selected[10:22],selected[22:]],1):
        name=f'batch{i:02d}';dest=output/name;(dest/'samples').mkdir(parents=True)
        context={'packet_id':'budget34_20261003_'+name,'storage_key':'hoi_budget34_20261003_'+name,
                 'title':f'有限预算审核 {i}/3 · {len(group)} 条','required_reviewer':'ymh','required_round':'primary',
                 'accepted_packet_ids':['batch02_20261003'],'import_subset':True,'semantic_only':True,
                 'instructions':'你只需看回放、选动作与阶段、判断交互及理由。系统自动准备候选证据片段并处理校验和统计；保存即确认片段支持你的判断，可展开修改。最多追加首审 34 条，信息不足选不确定，不自动追加样本。旧第二批记录可在页底导入。'}
        (dest/'queue.js').write_text('window.queue='+json.dumps({**payload,'entries':group,'review_context':context},ensure_ascii=True)+';',encoding='utf-8')
        (dest/'queue.json').write_text(json.dumps(group,ensure_ascii=False,indent=2),encoding='utf-8')
        ids={e['sample_id'] for e in group}|{e['source_sequence']+'__clean__none' for e in group}
        for sid in ids:
            src=original/'samples'/f'{sid}.js';target=dest/'samples'/f'{sid}.js';shutil.copy2(src,target)
            assert sha(src)==sha(target)
        shutil.copy2(original/'plotly.min.js',dest/'plotly.min.js')
        for src,target in [('review_template.html','index.html'),('review_app.js','review_app.js'),('review_form.js','review_form.js')]:
            shutil.copy2(root/'counterfactual'/src,dest/target)
        manifest['batches'].append({'directory':name,'count':len(group),'sample_ids':[e['sample_id'] for e in group],
          'asset_sha256':{p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}})
    (output/'plan_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    m=build(Path(__file__).resolve().parents[1])
    print(json.dumps({k:v for k,v in m.items() if k not in ('batches','planned_additional_samples','required_secondary_ids')},ensure_ascii=False,indent=2))
    print('Secondary work items:',len(m['required_secondary_ids']))
