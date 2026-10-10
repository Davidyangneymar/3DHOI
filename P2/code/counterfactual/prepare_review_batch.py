"""Create a reproducible 24-item primary review packet from unreviewed samples."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from counterfactual.audit_reviews import audit, required_secondaries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_batch(entries, records, batch_id):
    audit(entries, records)
    reviewed = {r['sample_id'] for r in records if r['round'] == 'primary'}
    pending = [e for e in entries if e['sample_id'] not in reviewed]
    strata = [(kind,severity) for kind in ('premature_release','object_motion_without_contact','spatial_contact_shift')
              for severity in ('low','medium','high')]
    schedule = []
    for block in range(6):
        schedule.append(('clean','none'))
        schedule.extend(strata[(block%3)*3:(block%3)*3+3])
    quota = Counter(schedule)
    available = Counter((e['type'],e['severity']) for e in pending)
    if any(available[key] < count for key,count in quota.items()):
        raise ValueError('Not enough unreviewed samples for this fixed batch; choose a new explicit quota')
    old_objects = Counter(e['object'] for e in entries if e['sample_id'] in reviewed)
    old_sources = Counter(e['source_sequence'] for e in entries if e['sample_id'] in reviewed)
    sources, objects, selected_ids = Counter(), Counter(), set()
    selected = []
    secondary = required_secondaries(entries)
    for key in schedule:
        candidates = [e for e in pending if (e['type'],e['severity'])==key and e['sample_id'] not in selected_ids]
        def rank(e):
            tie = hashlib.sha256((batch_id+'|'+e['sample_id']).encode()).hexdigest()
            return (sources[e['source_sequence']],objects[e['object']],old_objects[e['object']],old_sources[e['source_sequence']],tie)
        e = min(candidates,key=rank)
        selected.append({**e,'second_review_required':e['sample_id'] in secondary or e.get('second_review_required',False)})
        selected_ids.add(e['sample_id']);sources[e['source_sequence']]+=1;objects[e['object']]+=1
    return selected


def build(root, reviews, output, batch_id, reviewer):
    if output.exists():
        raise ValueError('Output exists; refusing to overwrite a review packet')
    original=root/'outputs/semantic_review_v11'
    entries=json.loads((original/'queue.json').read_text(encoding='utf-8'))
    records=[json.loads(s) for s in reviews.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    selected=select_batch(entries,records,batch_id)
    for e in selected:
        if sha(root/'counterfactual/v1/samples'/f"{e['sample_id']}.npz")!=e['sha256']:
            raise ValueError('Frozen sample hash changed: '+e['sample_id'])
    payload=json.loads((original/'queue.js').read_text(encoding='utf-8').removeprefix('window.queue=').removesuffix(';'))
    payload['entries']=selected
    payload['review_context']={
        'packet_id':batch_id,'storage_key':'hoi_'+batch_id,
        'title':'第二批首审 · 24 条','required_reviewer':reviewer,'required_round':'primary',
        'instructions':'这是新的一批首审，已排除前 14 条。请依据回放选动作和证据片段的阶段；同时搬运和摇晃时，可选搬运并在备注说明摇晃。错误表示交互不合理，无效表示构造或源数据异常；信息不足选不确定。橙色点是手部关节。逐条保存，完成后导出我的记录。'}
    (output/'samples').mkdir(parents=True)
    (output/'queue.js').write_text('window.queue='+json.dumps(payload,ensure_ascii=True)+';',encoding='utf-8')
    (output/'queue.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    assets={e['sample_id'] for e in selected}|{e['source_sequence']+'__clean__none' for e in selected}
    sample_hashes={}
    for sid in sorted(assets):
        src=original/'samples'/f'{sid}.js';dst=output/'samples'/f'{sid}.js'
        shutil.copy2(src,dst)
        if sha(src)!=sha(dst):raise ValueError('Sample copy mismatch')
        sample_hashes[sid]=sha(dst)
    shutil.copy2(original/'plotly.min.js',output/'plotly.min.js')
    for src,dst in [('review_template.html','index.html'),('review_app.js','review_app.js'),('review_form.js','review_form.js')]:
        shutil.copy2(root/'counterfactual'/src,output/dst)
    reviewed={r['sample_id'] for r in records if r['round']=='primary'}
    manifest={
        'batch_id':batch_id,'batch_size':len(selected),'input_reviews_sha256':sha(reviews),
        'full_queue_sha256':sha(original/'queue.json'),'previous_primary_samples':len(reviewed),
        'unreviewed_total_at_creation':len(entries)-len(reviewed),
        'unreviewed_outside_this_batch':len(entries)-len(reviewed)-len(selected),
        'selection_method':'Fixed quotas: 6 clean, 2 per perturbation/severity cell; greedy source/object diversity; batch-id SHA256 tie-break',
        'representative_benchmark':False,'automatic_semantic_labels_created':0,
        'type_severity_counts':dict(Counter(e['type']+'/'+e['severity'] for e in selected)),
        'objects':dict(Counter(e['object'] for e in selected)),
        'unique_sources':len({e['source_sequence'] for e in selected}),
        'missing_text_count':sum(not bool(e.get('description')) for e in selected),
        'required_secondary_ids':[e['sample_id'] for e in selected if e['second_review_required']],
        'sample_assets_sha256':sample_hashes,
        'ui_assets_sha256':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'browser_visual_verification':False}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-id',required=True);p.add_argument('--reviewer',required=True)
    args=p.parse_args()
    manifest=build(Path(__file__).resolve().parents[1],args.reviews,args.output,args.batch_id,args.reviewer)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sample_assets_sha256','ui_assets_sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
