"""Validate human review records; report readiness without inventing annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from counterfactual.protocol_checks import canonical_hash, require_text


def required_secondaries(queue):
    required = {e['sample_id'] for e in queue if e['type']=='spatial_contact_shift'}
    strata = {}
    for e in queue:
        strata.setdefault((e['type'],e.get('severity','none')),[]).append(e['sample_id'])
    for names in strata.values():
        ordered=sorted(names,key=lambda name:hashlib.sha256(name.encode()).hexdigest())
        required.update(ordered[:math.ceil(.2*len(names))])
    return required


def audit(queue, records, adjudications=(), review_status=None):
    entries = {e["sample_id"]: e for e in queue}
    if len(entries)!=len(queue):raise ValueError('Duplicate sample in review queue')
    holds = set()
    if review_status is not None:
        declared = review_status.get('excluded_from_confirmed_negative_ids')
        if not isinstance(declared, list) or any(not isinstance(s, str) for s in declared):
            raise ValueError('Explicit semantic hold list required')
        holds = set(declared)
        if len(holds) != len(declared) or not holds <= set(entries):
            raise ValueError('Unknown or duplicate semantic hold')
    secondaries=required_secondaries(queue)
    seen, grouped = set(), {}
    for r in records:
        if r.get("sample_id") not in entries:
            raise ValueError("Unknown review sample")
        e = entries[r["sample_id"]]
        if r.get("sample_sha256") != e["sha256"]:
            raise ValueError("Review refers to a different sample hash")
        require_text(r.get('reviewer'),'reviewer');require_text(r.get('reason'),'reason')
        if r.get("round") not in {"primary", "secondary"} or r.get("decision") not in {"valid", "error", "uncertain", "invalid"}:
            raise ValueError("Invalid round or decision")
        key = (r["sample_id"], r["reviewer"], r["round"])
        if key in seen:
            raise ValueError("Duplicate reviewer record; resolve explicitly")
        seen.add(key)
        if r["decision"] in {"valid", "error"}:
            a, b = r.get("start_frame"), r.get("end_frame")
            if any(not isinstance(r.get(k), str) or not r[k].strip() or r[k].strip() == 'unknown' for k in ('action','phase')) or type(a) is not int or type(b) is not int or not 0 <= a <= b < e["frames"]:
                raise ValueError("Definitive labels require action, phase and evidence frames")
        if r["decision"] == "valid" and e["type"] != "clean":
            raise ValueError("Plausible perturbations must not silently become positives")
        grouped.setdefault(r["sample_id"], []).append(r)
    resolutions={}
    for r in adjudications:
        sid=r.get('sample_id')
        if sid not in entries or sid in resolutions:raise ValueError('Unknown or duplicate adjudication')
        originals=grouped.get(sid,[])
        if set(r.get('review_sha256',[]))!={canonical_hash(v) for v in originals} or not originals:
            raise ValueError('Adjudication must reference every original review by hash')
        require_text(r.get('adjudicator'),'adjudicator');require_text(r.get('reason'),'reason')
        require_text(r.get('adjudicated_at'),'adjudicated_at')
        if r['adjudicator'] in {v['reviewer'] for v in originals}:raise ValueError('Adjudicator must be independent of initial reviewers')
        if r.get('sample_sha256')!=entries[sid]['sha256']:raise ValueError('Adjudication sample hash mismatch')
        if r.get('decision') not in {'valid','error','uncertain','invalid'}:raise ValueError('Invalid adjudicated decision')
        if r['decision']=='valid' and entries[sid]['type']!='clean':raise ValueError('Perturbations cannot become positives')
        if r['decision'] in {'valid','error'}:
            a,b=r.get('start_frame'),r.get('end_frame')
            if any(not isinstance(r.get(k), str) or not r[k].strip() or r[k].strip() == 'unknown' for k in ('action','phase')) or type(a) is not int or type(b) is not int or not 0<=a<=b<entries[sid]['frames']:
                raise ValueError('Definitive adjudication requires action, phase and evidence frames')
        resolutions[sid]=r
    statuses, included = {}, []
    for sid, e in entries.items():
        rs = grouped.get(sid, [])
        primary = [r for r in rs if r["round"] == "primary"]
        secondary = [r for r in rs if r["round"] == "secondary"]
        if sid in holds:
            status = 'semantic_hold'
        elif not primary:
            status = "pending_primary"
        elif (e.get("second_review_required",False) or sid in secondaries) and not any(a["reviewer"] != b["reviewer"] for a in primary for b in secondary):
            status = "pending_independent_secondary"
        elif sid in resolutions:
            status = resolutions[sid]['decision']
        elif len({(r["decision"], r.get("action"), r.get("phase")) for r in rs}) != 1:
            status = "unresolved_disagreement"
        else:
            status = primary[0]["decision"]
        statuses[sid] = status
        if status in {"valid", "error"}:
            included.append({**e, "label_error": int(status == "error"), "reviews": rs,
                             "adjudication":resolutions.get(sid)})
    pending = sum(s.startswith("pending") or s in {"unresolved_disagreement", "semantic_hold"} for s in statuses.values())
    return {"ready_for_curated_freeze": pending == 0 and bool(included), "pending_or_disputed": pending,
            "included_count": len(included), "counts": {s: list(statuses.values()).count(s) for s in sorted(set(statuses.values()))},
            "statuses": statuses, "eligible_samples": included}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queue", type=Path)
    p.add_argument("reviews", type=Path, nargs="*")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--adjudications',type=Path,help='Separate third-reviewer JSONL; original votes retained')
    p.add_argument('--review-status',type=Path,help='Hash-bound semantic holds for one consolidated review file')
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    records = [json.loads(line) for path in args.reviews for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    resolutions=[json.loads(s) for s in args.adjudications.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if args.adjudications else []
    status = None
    if args.review_status:
        from counterfactual.review_integrity import load_review_status
        if len(args.reviews) != 1:p.error('--review-status requires one consolidated review file')
        status = load_review_status(args.review_status,args.reviews[0])
    report = audit(queue, records, resolutions, status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in {"statuses", "eligible_samples"}}, indent=2))


if __name__ == "__main__":
    main()
