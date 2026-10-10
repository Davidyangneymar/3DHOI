"""Evaluation primitives. Unknown labels are excluded, never treated as valid."""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from counterfactual.protocol_checks import valid_error_label


def binary_metrics(labels, scores, threshold: float) -> dict:
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary labels and scores")
    if not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    pred = s >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    both = tpr is not None and fpr is not None
    return {"n": len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "recall_error": tpr, "false_positive_rate": fpr,
            "false_negative_rate": 1 - tpr if tpr is not None else None,
            "balanced_accuracy": (tpr + 1 - fpr) / 2 if both else None,
            "roc_auc": float(roc_auc_score(y, s)) if both else None,
            "average_precision": float(average_precision_score(y, s)) if both else None,
            "threshold": float(threshold)}


def grouped_intervals(labels, scores, groups, threshold, repeats=1000, seed=20261001):
    y, s, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    if y.shape != s.shape or y.shape != g.shape:
        raise ValueError("Labels, scores and group IDs must align")
    names = np.unique(g)
    keys = ["roc_auc", "balanced_accuracy", "false_positive_rate", "recall_error"]
    values = {k: [] for k in keys}
    if len(names) >= 2:
        rng = np.random.default_rng(seed)
        indices = {name: np.flatnonzero(g == name) for name in names}
        for _ in range(repeats):
            ids = np.concatenate([indices[name] for name in rng.choice(names, len(names), replace=True)])
            m = binary_metrics(y[ids], s[ids], threshold)
            for key in keys:
                if m[key] is not None:
                    values[key].append(m[key])
    return {"group_count": len(names), "repeats": repeats, "seed": seed,
            "intervals": {k: {"low": float(np.quantile(v, .025)) if v else None,
                               "high": float(np.quantile(v, .975)) if v else None,
                               "valid_replicates": len(v)} for k, v in values.items()}}


def validate_manifest(rows: list[dict]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError('A nonempty manifest is required')
    ids, sources, conditions = set(), {}, {}
    condition_sources = {}
    condition_pairs = {}
    for row in rows:
        if not isinstance(row, dict):raise ValueError('Manifest rows must be objects')
        for key in ("sample_id", "condition_id", "source_group", "split", "path", "metadata"):
            if key not in row or row[key] in ("", None):
                raise ValueError(f"Missing {key}")
        for key in ('sample_id','source_group','condition_id','path'):
            if not isinstance(row[key],str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(row['metadata'],dict):
            raise ValueError('metadata must be an object')
        if 'error_types' in row and (not isinstance(row['error_types'],list) or any(not isinstance(v,str) or not v.strip() for v in row['error_types'])):
            raise ValueError('error_types must be a list of nonempty strings')
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
        if row["split"] not in {"dev", "test", "diagnostic"}:
            raise ValueError("split must be dev/test/diagnostic")
        if not valid_error_label(row.get('label_error')):
            raise ValueError("label_error must be 0/1/uncertain/null")
        if row.get('object') is not None and row['metadata'].get('object') is not None and row['object'] != row['metadata']['object']:
            raise ValueError('Manifest object disagrees with adapter metadata')
        for field in ('action','object','data_domain'):
            if field in row and (not isinstance(row[field],str) or not row[field].strip()):
                raise ValueError(field + ' must be nonempty text when supplied')
        condition_pairs.setdefault(row['condition_id'],set()).add((row.get('action'),row.get('object')))
        for field, mapping in [("source_group", sources), ("condition_id", conditions)]:
            mapping.setdefault(row[field], set()).add(row["split"])
        condition_sources.setdefault(row['condition_id'],set()).add(row['source_group'])
    if any(len(s) > 1 for s in [*sources.values(), *conditions.values()]):
        raise ValueError("Source/condition leakage across splits")
    if any(len(s)>1 for s in condition_sources.values()):
        raise ValueError('One condition cannot span multiple source groups')
    if any(len(s)>1 for s in condition_pairs.values()):
        raise ValueError('One condition cannot span multiple action/object pairs')


def validate_input_provenance(rows):
    """Catch identical files/features under renamed IDs across dev/test/diagnostic."""
    for key in ('input_sha256','feature_content_sha256'):
        splits = {}
        for row in rows:
            value = row.get('input_provenance',{}).get(key)
            if value:
                splits.setdefault(value,set()).add(row['split'])
        if any(len(v)>1 for v in splits.values()):
            raise ValueError('Identical input leakage across splits through ' + key)


def evaluate_predictions(rows, threshold, repeats=1000):
    if rows:
        validate_manifest(rows)
        validate_input_provenance(rows)
    output = {}
    for split in sorted({r["split"] for r in rows}):
        subset = [r for r in rows if r["split"] == split]
        labelled = [r for r in subset if r.get("label_error") in (0, 1)]
        y = [r["label_error"] for r in labelled]
        groups = [r["source_group"] for r in labelled]
        result = {"total": len(subset), "labelled": len(labelled),
                  "excluded_uncertain": sum(r.get("label_error") == "uncertain" for r in subset),
                  "unlabelled": sum(r.get("label_error") is None for r in subset)}
        for name, field, t in [("critic", "p_error", threshold), ("rule", "rule_score", 1.0)]:
            scores = [r[field] for r in labelled]
            result[name] = {**binary_metrics(y, scores, t),
                            "bootstrap": grouped_intervals(y, scores, groups, t, repeats)}
        result["per_error_type"] = {}
        for kind in sorted({k for r in labelled for k in r.get("error_types", [])}):
            typed = [r for r in labelled if r["label_error"] == 0 or kind in r.get("error_types", [])]
            result["per_error_type"][kind] = {
                name: binary_metrics([r["label_error"] for r in typed], [r[field] for r in typed], t)
                for name, field, t in [("critic", "p_error", threshold), ("rule", "rule_score", 1.0)]}
        output[split] = result
    return output


def select_candidates(rows, k=4, seed=20261001):
    validate_manifest(rows)
    if type(k) is not int or k < 1:raise ValueError('k must be a positive integer')
    grouped = {}
    for row in rows:
        grouped.setdefault(row["condition_id"], []).append(row)
    result = []
    for condition, candidates in sorted(grouped.items()):
        if any(type(r.get('candidate_index')) is not int for r in candidates):
            raise ValueError('candidate_index must be an integer')
        for r in candidates:
            if not np.isfinite([r['p_valid'],r['rule_score']]).all() or not 0 <= r['p_valid'] <= 1:
                raise ValueError('Invalid candidate selection scores')
        if len(candidates) != k or {r.get("candidate_index") for r in candidates} != set(range(k)):
            raise ValueError(f"{condition}: expected exactly {k} candidates indexed 0..{k - 1}")
        candidates = sorted(candidates, key=lambda r: r["candidate_index"])
        if len({r["split"] for r in candidates}) != 1:
            raise ValueError("A candidate pool cannot cross splits")
        random_index = int(hashlib.sha256(f"{seed}:{condition}".encode()).hexdigest(), 16) % k
        picks = {"baseline_first": candidates[0], "random": candidates[random_index],
                 "rule": min(candidates, key=lambda r: (r["rule_score"], r["candidate_index"])),
                 "critic": max(candidates, key=lambda r: (r["p_valid"], -r["candidate_index"]))}
        result.append({"condition_id": condition, "split": candidates[0]["split"], "k": k,
                       **{method: r["sample_id"] for method, r in picks.items()}})
    return result
