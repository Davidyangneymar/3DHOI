"""Freeze a reviewed subset as v1.1; never regenerate or overwrite v1."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from counterfactual.audit_reviews import audit
from models.interaction_critic import ROOT
from counterfactual.critic_api import FEATURE_VERSION, sha256
from counterfactual.review_integrity import load_review_status
from counterfactual.reviewed_subset import select_reviewed_subset, training_eligibility, validate_parent_queue

OUT = ROOT / "counterfactual" / "v1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queue", type=Path)
    p.add_argument("reviews", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--adjudications',type=Path)
    p.add_argument('--review-status',type=Path,help='Required semantic status bound to consolidated reviews')
    p.add_argument('--scope', choices=['full', 'reviewed-subset'], default='full',
                   help='Explicit partial-review archive; does not approve the full dataset')
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be a new directory")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    frozen = [json.loads(s) for s in (OUT / "index.jsonl").read_text().splitlines()]
    parent_manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(OUT / 'index.jsonl') != parent_manifest['index_sha256']:
        p.error('Frozen parent index hash mismatch')
    by_id = {e["sample_id"]: e for e in frozen}
    validate_parent_queue(queue, frozen)
    if len(queue) != len(frozen) or {e['sample_id'] for e in queue} != set(by_id):
        p.error("Review queue must cover the complete frozen v1 index")
    for e in queue:
        original = by_id[e["sample_id"]]
        if any(e.get(key) != value for key, value in original.items()):
            p.error("Review queue differs from frozen v1 index")
    records = [json.loads(s) for path in args.reviews for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    resolutions=[json.loads(s) for s in args.adjudications.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if args.adjudications else []
    report = audit(queue, records, resolutions)
    if args.scope == 'full' and not report["ready_for_curated_freeze"]:
        p.error(f"Human review gate closed: {report['counts']}")
    if args.review_status is None or len(args.reviews) != 1:
        p.error('Semantic review gate requires --review-status and one consolidated reviews file')
    status = load_review_status(args.review_status,args.reviews[0])
    report = audit(queue, records, resolutions, status)
    if args.scope == 'full' and not report['ready_for_curated_freeze']:
        p.error(f"Human review gate closed: {report['counts']}")
    selection = select_reviewed_subset(queue, report) if args.scope == 'reviewed-subset' else None
    entries = selection['eligible_samples'] if selection else report['eligible_samples']
    if not entries:
        p.error('No eligible samples for declared review scope')
    if args.scope == 'full':
        for split in ("train", "val", "test"):
            if {e["label_error"] for e in entries if e["split"] == split} != {0, 1}:
                p.error(f"Reviewed {split} split needs both classes; add reviewed data before freezing")
    eligibility = training_eligibility(entries)
    for e in entries:
        if sha256(OUT / "samples" / f"{e['sample_id']}.npz") != e["sha256"]:
            p.error("Sample hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "samples").mkdir()
    clean_index = []
    for e in entries:
        name = f"{e['sample_id']}.npz"
        shutil.copy2(OUT / "samples" / name, args.output / "samples" / name)
        clean_index.append({**e, "label": e["label_error"], "feature_version": FEATURE_VERSION})
    (args.output / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in clean_index), encoding="utf-8")
    shutil.copyfile(args.reviews[0],args.output / 'review_records.jsonl')
    shutil.copyfile(args.review_status,args.output / 'review_status.json')
    (args.output / "adjudications.jsonl").write_text(''.join(json.dumps(e)+'\n' for e in resolutions),encoding='utf-8')
    (args.output / 'queue.json').write_text(json.dumps(queue,indent=2),encoding='utf-8')
    (args.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / 'training_eligibility.json').write_text(json.dumps(eligibility, indent=2), encoding='utf-8')
    if selection:
        (args.output / 'subset.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    manifest = {"dataset": "Counterfactual Dataset v1.1 curated subset", "feature_version": FEATURE_VERSION,
                "scope": "Reviewed subset only; new phase-aware generation and normal examples not yet added",
                "split_protocol": "v1_source_sequence_disjoint", "unseen_pair_verified": False,
                "test_status": "previously_exposed_regression_only", "sample_count": len(entries),
                "by_split": dict(Counter(e['split'] for e in entries)),
                "parent_manifest_sha256": sha256(OUT / 'manifest.json'),
                "index_sha256": sha256(args.output / 'index.jsonl'),
                "review_records_sha256": sha256(args.output / 'review_records.jsonl'),
                "review_status_sha256": sha256(args.output / 'review_status.json'),
                "audit_sha256": sha256(args.output / 'audit.json'),
                "queue_sha256": sha256(args.output / 'queue.json'),
                "adjudications_sha256": sha256(args.output / 'adjudications.jsonl'),
                "builder_sha256": sha256(Path(__file__)),
                "review_inputs": {str(f.resolve()): sha256(f) for f in args.reviews}}
    manifest.update(review_scope=args.scope, training_ready=eligibility['training_ready'],
                    training_eligibility_sha256=sha256(args.output/'training_eligibility.json'))
    if selection:
        manifest.update(dataset='v1.1-reviewed-subset', subset_sha256=sha256(args.output/'subset.json'),
                        scope='Nonrandom reviewed subset; full v1 is not approved; spatial cases remain diagnostic',
                        full_queue_ready=report['ready_for_curated_freeze'], population_quality_estimate=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen {len(entries)} reviewed samples in {args.output}; not an unseen-pair benchmark")


if __name__ == "__main__":
    main()
