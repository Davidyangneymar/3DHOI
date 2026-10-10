"""Prepare a 14-item first batch plus objective geometry cues, not semantic labels."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

from counterfactual.audit_reviews import required_secondaries
from counterfactual.critic_api import sha256
from models.interaction_critic import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Output exists; preserve previous review packets')
    original=ROOT/'outputs/semantic_review_v11'
    entries=json.loads((original/'queue.json').read_text(encoding='utf-8'))
    second=required_secondaries(entries)
    for e in entries:e['second_review_required']=e['sample_id'] in second
    chosen=[];used_sources=set()
    for kind in ['premature_release','object_motion_without_contact','spatial_contact_shift']:
        for severity in ['low','medium','high']:
            options=[e for e in entries if e['type']==kind and e['severity']==severity]
            e=min(options,key=lambda e:(e['source_sequence'] in used_sources,e['sample_id']))
            chosen.append(e);used_sources.add(e['source_sequence'])
    report=json.loads((ROOT/'reports/critic_metrics.json').read_text())
    fp_ids={r['sample_id'] for r in report['test_predictions'] if r['label']==0 and r['critic_score']>=report['critic_threshold']}
    chosen.extend(e for e in entries if e['sample_id'] in fp_ids)
    cues=[]
    for e in entries:
        path=ROOT/'counterfactual/v1/samples'/f"{e['sample_id']}.npz"
        if sha256(path)!=e['sha256']:raise ValueError('Frozen sample hash mismatch')
        with np.load(path) as z:
            d=z['distances_m'];f=z['features']
            detached=(d>.05).all(axis=1)
            moving=f[:,6]>.008
            cues.append({'sample_id':e['sample_id'],'source_sequence':e['source_sequence'],'type':e['type'],
                         'frames':e['frames'],'text_missing':not bool(e.get('description','')),
                         'both_hands_far_and_object_moving_frames':int((detached&moving).sum()),
                         'no_hand_near_fraction':float(detached.mean()),
                         'distance_definition':'nearest mesh vertex; 0.05m threshold',
                         'motion_definition':'object displacement >0.008 metres/frame; not force/support evidence',
                         'semantic_decision':None})
    payload=json.loads((original/'queue.js').read_text(encoding='utf-8').removeprefix('window.queue=').removesuffix(';'))
    payload['entries']=chosen
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'samples').mkdir()
    assets={e['sample_id'] for e in chosen}|{e['source_sequence']+'__clean__none' for e in chosen}
    for sid in sorted(assets):shutil.copy2(original/'samples'/f'{sid}.js',args.output/'samples'/f'{sid}.js')
    shutil.copy2(original/'plotly.min.js',args.output/'plotly.min.js')
    template=Path(__file__).with_name('review_template.html')
    shutil.copy2(template,args.output/'index.html')
    for asset in ('review_form.js','review_app.js'):
        shutil.copy2(Path(__file__).with_name(asset),args.output/asset)
    (args.output/'queue.js').write_text('window.queue='+json.dumps(payload,ensure_ascii=True)+';',encoding='utf-8')
    (args.output/'queue.json').write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    (args.output/'geometry_cues_all436.json').write_text(json.dumps(cues,indent=2),encoding='utf-8')
    summary={'batch_size':len(chosen),'parent_count':len(entries),'covered_type_severity_cells':9,
             'historical_clean_false_positive_count':len(fp_ids),'unreviewed_remaining':len(entries),
             'automatic_semantic_labels_created':0,'text_missing_count':sum(e['text_missing'] for e in cues),
             'source_queue_sha256':sha256(original/'queue.json'),'template_sha256':sha256(template),
             'selection_note':'Nine type/severity cells plus five historical clean false positives; nonrandom diagnostic sampling, not a benchmark',
             'review_policy':'Merge exported records against the FULL 436-item queue; this pilot cannot freeze v1.1 by itself'}
    (args.output/'batch_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
