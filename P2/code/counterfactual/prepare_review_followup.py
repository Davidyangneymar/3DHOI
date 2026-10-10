"""Prepare primary clarification and independent secondary packets; never relabel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from counterfactual.audit_reviews import audit, required_secondaries
from counterfactual.protocol_checks import canonical_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_packets(entries, records, batch_id):
    audit(entries, records)  # Validate against the FULL queue before selecting subsets.
    reviewers = {r['reviewer'] for r in records}
    if len(reviewers) != 1 or any(r['round'] != 'primary' for r in records):
        raise ValueError('Expected one original primary reviewer')
    reviewer = next(iter(reviewers))
    by_id = {e['sample_id']: e for e in entries}
    required = required_secondaries(entries)
    primary = [by_id[r['sample_id']] for r in records if r['decision'] == 'invalid']
    secondary = [by_id[r['sample_id']] for r in records
                 if r['sample_id'] in required or by_id[r['sample_id']].get('second_review_required', False)]
    previous = {r['sample_id']:dict(note=r.get('note') or r['reason'], start_frame=r['start_frame'],
                                  end_frame=r['end_frame'],record_sha256=canonical_hash(r))
                for r in records if r['decision'] == 'invalid'}
    return [
        ('primary_clarification',primary,{
            'packet_id':batch_id+'_clarification','storage_key':'hoi_'+batch_id+'_clarification',
            'title':f'{len(primary)} 条首轮分类复核','required_reviewer':reviewer,'required_round':'primary',
            'previous_records':previous,
            'instructions':'只需复核这批原来标为无效的样本。错误＝交互关系不合理；无效＝数据或构造异常。手与物体分离本身不足以选无效；排除不了合法解释可选不确定。允许保留原判定，无需为了产生负例而改标签。逐条保存后导出。'}),
        ('independent_secondary',secondary,{
            'packet_id':batch_id+'_secondary','storage_key':'hoi_'+batch_id+'_secondary',
            'title':f'{len(secondary)} 条独立二审','required_round':'secondary','excluded_reviewers':[reviewer],
            'instructions':'由另一位审核人独立查看回放并判断，不参考首轮结论。请填写自己的稳定代号，逐条保存后导出。本包未包含首轮判定或说明。'})]


def build(root, reviews, output, batch_id):
    if output.exists():
        raise ValueError('Output exists; preserve earlier packets')
    original = root/'outputs/semantic_review_v11'
    entries = json.loads((original/'queue.json').read_text(encoding='utf-8'))
    records = [json.loads(s) for s in reviews.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    payload = json.loads((original/'queue.js').read_text(encoding='utf-8').removeprefix('window.queue=').removesuffix(';'))
    plans = plan_packets(entries, records, batch_id)
    output.mkdir(parents=True)
    manifest = {'input_reviews_sha256':digest(reviews),'full_queue_sha256':digest(original/'queue.json'),
                'automatic_labels_created':0,'original_records_modified':False,'packets':{}}
    for name, selected, context in plans:
        dest = output/name; (dest/'samples').mkdir(parents=True)
        packet = {**payload,'entries':selected,'review_context':context}
        (dest/'queue.js').write_text('window.queue='+json.dumps(packet,ensure_ascii=True)+';',encoding='utf-8')
        (dest/'queue.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
        assets = {e['sample_id'] for e in selected}|{e['source_sequence']+'__clean__none' for e in selected}
        for sid in sorted(assets):
            shutil.copy2(original/'samples'/f'{sid}.js',dest/'samples'/f'{sid}.js')
        shutil.copy2(original/'plotly.min.js',dest/'plotly.min.js')
        for source,target in [('review_template.html','index.html'),('review_app.js','review_app.js'),('review_form.js','review_form.js')]:
            shutil.copy2(root/'counterfactual'/source,dest/target)
        manifest['packets'][name] = {'count':len(selected),'sample_ids':[e['sample_id'] for e in selected],
                                    'assets_sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-id',required=True)
    args=p.parse_args()
    result=build(Path(__file__).resolve().parents[1],args.reviews,args.output,args.batch_id)
    print(json.dumps({k:v['count'] for k,v in result['packets'].items()},indent=2))


if __name__=='__main__':main()
