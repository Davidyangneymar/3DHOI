"""Evaluate future generator output without retraining the frozen critic.

CSV columns: path,object,label (label 1=real generator error, 0=valid).
NPZ: either features (T,13), or joints (T,24,3) plus obj_trans,
obj_rot_mat, obj_scale and optional obj_bottom_* transforms.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from counterfactual.build_v1 import features
from models.interaction_critic import InteractionCritic, MODEL, metrics, resample, rule_score
from omomo_contact import extract_distances


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path, help="CSV with path,object,label")
    p.add_argument("--output", type=Path, default=Path("reports/generator_errors_predictions.csv"))
    args = p.parse_args()
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
    net = InteractionCritic()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig", newline="")))
    predictions = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = args.manifest.parent / path
        with np.load(path) as z:
            if "features" in z:
                raw = z["features"]
            else:
                joints = z["joints"]
                window = {k: z[k] for k in z.files if k.startswith("obj_")}
                dist, nearest, part = extract_distances(window, joints, row["object"])
                raw = features(window, joints, dist, nearest, part, row["object"])
        if raw.shape[1] != 13 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid features in {path}")
        x = (resample(raw) - ckpt["mean"]) / ckpt["std"]
        with torch.no_grad():
            score = float(torch.sigmoid(net(torch.tensor(x[None]))).item())
        predictions.append({"path": str(path), "object": row["object"], "label": row.get("label", ""),
                            "critic_score": score, "critic_flag": int(score >= ckpt["threshold"]),
                            "rule_score": rule_score(raw), "rule_flag": int(rule_score(raw) >= 1.0)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]) if predictions else ["path", "object", "label", "critic_score", "critic_flag", "rule_score", "rule_flag"])
        writer.writeheader()
        writer.writerows(predictions)
    if predictions and all(x["label"] in ("0", "1") for x in predictions):
        y = np.asarray([int(x["label"]) for x in predictions])
        print(json.dumps({"critic": metrics(y, np.asarray([x["critic_score"] for x in predictions]), ckpt["threshold"]),
                          "rule": metrics(y, np.asarray([x["rule_score"] for x in predictions]), 1.0)}, indent=2))
    print(f"wrote {args.output} ({len(predictions)} samples)")


if __name__ == "__main__":
    main()
