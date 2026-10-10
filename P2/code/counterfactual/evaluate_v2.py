"""Score a JSONL generator manifest into a fresh run directory. No training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from counterfactual.benchmark import evaluate_predictions, select_candidates, validate_manifest, validate_input_provenance
from counterfactual.critic_api import CriticScorer, FEATURE_VERSION, sha256
from counterfactual.generator_adapter import load_features
from counterfactual.protocol_checks import check_pair_protocol
from models.interaction_critic import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite")
    parser.add_argument("--checkpoint", type=Path, default=MODEL)
    parser.add_argument("--rerank", action="store_true", help="Experimental selection, not proof of quality")
    parser.add_argument("--pair-protocol", type=Path, help="Optional frozen C protocol; absence means no unseen-pair claim")
    parser.add_argument('--data-domain',help='Explicit experiment domain, e.g. real_generator or synthetic_v1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    validate_manifest(rows)
    declared_domains={r['data_domain'] for r in rows if 'data_domain' in r}
    if args.data_domain is not None:
        if not args.data_domain.strip():parser.error('--data-domain must be nonempty')
        declared_domains.add(args.data_domain)
    if len(declared_domains)>1:parser.error('One data domain required per evaluation run')
    domain=next(iter(declared_domains)) if declared_domains else 'unspecified'
    if domain != 'unspecified' and args.data_domain is None and any('data_domain' not in r for r in rows):
        parser.error('Declare data_domain on every row or use --data-domain')
    rows=[{**r,'data_domain':domain} for r in rows]
    pair_check = check_pair_protocol(rows,json.loads(args.pair_protocol.read_text(encoding='utf-8'))) if args.pair_protocol else None
    if args.pair_protocol:
        assignments=json.loads(args.pair_protocol.read_text(encoding='utf-8'))['conditions']
        rows=[{**r,'pair_status':assignments[r['condition_id']]['pair_status']} for r in rows]
    torch.set_num_threads(2)
    scorer = CriticScorer(args.checkpoint)
    arrays, provenance = [], []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        raw, info = load_features(path, row["metadata"])
        arrays.append(raw); provenance.append(info)
    scores = scorer.score_many(arrays, feature_version=FEATURE_VERSION)
    predictions = [{**r, **s, "input_provenance": p} for r, s, p in zip(rows, scores, provenance)]
    validate_input_provenance(predictions)
    selections = select_candidates(predictions) if args.rerank else None
    metrics = evaluate_predictions(predictions, scorer.threshold)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    if pair_check:
        subgroups={status:evaluate_predictions([r for r in predictions if r['pair_status']==status],scorer.threshold)
                   for status in ('seen','unseen')}
        (args.output/'pair_subgroup_metrics.json').write_text(json.dumps(subgroups,indent=2,allow_nan=False),encoding='utf-8')
    if selections is not None:
        (args.output / "selections.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    (args.output / "run.json").write_text(json.dumps({"manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": scorer.checkpoint_sha256, "feature_version": FEATURE_VERSION,
        "threshold": scorer.threshold, "mode": "frozen_zero_shot", "count": len(rows),
        "data_domain": domain,
        "quality_improvement_verified": False,
        "pair_protocol_check": pair_check,
        "code_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("critic_api.py"),
                        Path(__file__).with_name("generator_adapter.py"), Path(__file__).with_name("benchmark.py"),
                        Path(__file__).with_name("geometry_features.py"), Path(__file__).with_name("protocol_checks.py")]}
        }, indent=2), encoding="utf-8")
    print(f"Scored {len(rows)} sequences into {args.output}; independent quality evaluation still required")


if __name__ == "__main__":
    main()
