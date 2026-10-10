"""Temporal CNN critic and fixed rule baseline for Counterfactual Dataset v1.

Train: python -m models.interaction_critic
Predict: python -m models.interaction_critic --predict path/to/sample.npz
The output is a corruption probability, not a calibrated physical validity score.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "counterfactual" / "v1"
MODEL = ROOT / "models" / "interaction_critic_v1.pt"
SEED = 20260928
STEPS = 96


class InteractionCritic(nn.Module):
    """Small temporal model for framewise hand/object geometry features."""

    def __init__(self, channels: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1)).squeeze(1)


def resample(x: np.ndarray, steps: int = STEPS) -> np.ndarray:
    pos = np.linspace(0, len(x) - 1, steps)
    return np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


def rule_score(raw: np.ndarray) -> float:
    """Three independently interpretable geometric failure rules."""
    dist = raw[:, :2]
    contact = raw[:, 2:4]
    obj_speed = raw[:, 6]
    local_speed = raw[:, 9:11]
    late = slice(int(len(raw) * 0.65), None)
    release = np.mean(np.any(dist[late] > 0.075, axis=1))
    unsupported_motion = np.mean((obj_speed > 0.008) & (contact.max(axis=1) == 0))
    shift = np.quantile(local_speed[contact.max(axis=1) > 0].max(axis=1), 0.98) if contact.any() else 0.0
    return float(max(release / 0.45, unsupported_motion / 0.08, shift / 0.10))


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    out = {"n": int(len(y)), "positive": int(y.sum()), "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else None,
           "f1": float(f1_score(y, pred, zero_division=0)), "threshold": float(threshold),
           "false_positive": int(((pred == 1) & (y == 0)).sum()), "false_negative": int(((pred == 0) & (y == 1)).sum())}
    if len(set(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["average_precision"] = float(average_precision_score(y, scores))
    return out


def threshold_from_validation(y: np.ndarray, scores: np.ndarray) -> float:
    choices = np.unique(np.r_[0.5, np.quantile(scores, np.linspace(0, 1, 101))])
    return float(max(choices, key=lambda t: (balanced_accuracy_score(y, scores >= t), -abs(t - 0.5))))


def load_data() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    entries = [json.loads(line) for line in (DATA / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    x, raw, rule = [], [], []
    for e in entries:
        with np.load(DATA / "samples" / f"{e['sample_id']}.npz") as z:
            f = z["features"]
        raw.append(f)
        x.append(resample(f))
        rule.append(rule_score(f))
    return entries, np.stack(x), np.asarray(rule), np.asarray([e["label"] for e in entries], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.predict:
        checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
        net = InteractionCritic()
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        with np.load(args.predict) as z:
            raw = z["features"]
        x = (resample(raw) - checkpoint["mean"]) / checkpoint["std"]
        with torch.no_grad():
            score = torch.sigmoid(net(torch.tensor(x[None]))).item()
        print(json.dumps({"critic_probability": score, "critic_flag": score >= checkpoint["threshold"],
                          "rule_score": rule_score(raw), "rule_flag": rule_score(raw) >= 1}))
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    entries, x, rule, y = load_data()
    train = np.asarray([i for i, e in enumerate(entries) if e["split"] == "train" and e["severity"] != "high"])
    val = np.asarray([i for i, e in enumerate(entries) if e["split"] == "val" and e["severity"] != "high"])
    test_seen = np.asarray([i for i, e in enumerate(entries) if e["split"] == "test" and e["severity"] != "high"])
    test_unseen = np.asarray([i for i, e in enumerate(entries) if e["split"] == "test" and e["severity"] in ("none", "high")])
    train_sources = {entries[i]["source_sequence"] for i in train}
    val_sources = {entries[i]["source_sequence"] for i in val}
    test_sources = {entries[i]["source_sequence"] for i in test_seen}
    assert not (train_sources & val_sources or train_sources & test_sources or val_sources & test_sources)
    mean = x[train].mean(axis=(0, 1), keepdims=True)
    std = np.maximum(x[train].std(axis=(0, 1), keepdims=True), 1e-4)
    xt = torch.tensor((x - mean) / std)
    yt = torch.tensor(y, dtype=torch.float32)
    net = InteractionCritic()
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.005)
    pos_weight = torch.tensor([max(1, (y[train] == 0).sum()) / max(1, (y[train] == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(100):
        net.train()
        permutation = np.random.permutation(train)
        for ids in np.array_split(permutation, max(1, int(np.ceil(len(permutation) / 32)))):
            opt.zero_grad()
            loss = loss_fn(net(xt[ids]), yt[ids])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = loss_fn(net(xt[val]), yt[val]).item()
        history.append({"epoch": epoch + 1, "val_loss": vloss})
        if vloss < best - 1e-4:
            best = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(xt)).numpy()
    threshold = threshold_from_validation(y[val], scores[val])
    torch.save({"state_dict": net.state_dict(), "mean": mean[0], "std": std[0],
                "threshold": threshold, "seed": SEED, "channels": 13}, MODEL)
    result = {"dataset_manifest_sha256": __import__("hashlib").sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
              "seed": SEED, "train_count": len(train), "val_count": len(val),
              "train_sources": sorted(train_sources), "val_sources": sorted(val_sources), "test_sources": sorted(test_sources),
              "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"], "critic_threshold": threshold,
              "rule_threshold": 1.0, "metrics": {}}
    for label, ids in [("validation_seen", val), ("held_out_seen_severity", test_seen), ("held_out_unseen_high_severity", test_unseen)]:
        result["metrics"][label] = {"critic": metrics(y[ids], scores[ids], threshold),
                                     "rule": metrics(y[ids], rule[ids], 1.0)}
    for kind in ("premature_release", "object_motion_without_contact", "spatial_contact_shift"):
        ids = np.asarray([i for i, e in enumerate(entries) if e["split"] == "test" and e["type"] == kind])
        result["metrics"][f"recall_{kind}"] = {"n": int(len(ids)),
            "critic": float(np.mean(scores[ids] >= threshold)) if len(ids) else None,
            "rule": float(np.mean(rule[ids] >= 1.0)) if len(ids) else None}
    result["test_predictions"] = [{"sample_id": entries[i]["sample_id"], "label": int(y[i]),
                                    "critic_score": float(scores[i]), "rule_score": float(rule[i])}
                                   for i, e in enumerate(entries) if e["split"] == "test"]
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "critic_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": result["best_epoch"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
