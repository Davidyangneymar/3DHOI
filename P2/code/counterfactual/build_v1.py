"""Build a reproducible, sequence-disjoint OMOMO grasp counterfactual dataset.

Run from the repository root: python -m counterfactual.build_v1
All distances are metres. Contacts are geometry estimates, not physical forces.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import cKDTree

from omomo_contact import extract_distances, object_parts, raw_record_as_window

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVIEW = Path(os.environ.get("COUNTERHOI_SOURCE_REVIEW", str(ROOT / "reviews" / "source_review.csv")))
OUT = ROOT / "counterfactual" / "v1"
SEED = 20260928
LEVELS = {"low": 0.10, "medium": 0.18, "high": 0.28}
TYPES = ("premature_release", "object_motion_without_contact", "spatial_contact_shift")
THRESHOLD = 0.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_sequences(rows: list[dict]) -> dict[str, str]:
    """Stratify reviewed train sequences by object; retain official test split."""
    rng = np.random.default_rng(SEED)
    groups: dict[str, list[str]] = {}
    result = {}
    for r in rows:
        if r["split"] == "test":
            result[r["sequence"]] = "test"
        else:
            groups.setdefault(r["object"], []).append(r["sequence"])
    for seqs in groups.values():
        rng.shuffle(seqs)
        n = len(seqs)
        n_test = max(1, round(n * 0.2)) if n >= 4 else 0
        n_val = max(1, round(n * 0.15)) if n >= 4 else 0
        for i, name in enumerate(seqs):
            result[name] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return result


def contact_local(window: dict, nearest: np.ndarray, part: np.ndarray, name: str) -> np.ndarray:
    local = np.zeros_like(nearest)
    for part_id, (_, _, prefix) in enumerate(object_parts(name)):
        take = part == part_id
        rot = window["obj_rot_mat"] if prefix == "obj" else window["obj_bottom_rot_mat"]
        trans = window[f"{prefix}_trans"]
        scale = window[f"{prefix}_scale"]
        p = np.einsum("thi,tij->thj", nearest - trans[:, None], rot) / np.maximum(np.abs(scale[:, None, None]), 1e-9)
        local[take] = p[take]
    return local


def features(window: dict, joints: np.ndarray, dist: np.ndarray, nearest: np.ndarray, part: np.ndarray, name: str) -> np.ndarray:
    hands = joints[:, [22, 23]]
    obj = np.asarray(window["obj_trans"])
    rel = hands - obj[:, None]
    local = contact_local(window, nearest, part, name)
    dhand = np.linalg.norm(np.diff(hands, axis=0, prepend=hands[:1]), axis=-1)
    dobj = np.linalg.norm(np.diff(obj, axis=0, prepend=obj[:1]), axis=-1)
    drel = np.linalg.norm(np.diff(rel, axis=0, prepend=rel[:1]), axis=-1)
    dlocal = np.linalg.norm(np.diff(local, axis=0, prepend=local[:1]), axis=-1)
    rotation = np.asarray(window["obj_rot_mat"])
    hand_local_world = np.einsum("thi,tij->thj", rel, rotation)
    initial_grasp = np.median(hand_local_world[:max(5, len(joints) // 5)], axis=0)
    grasp_shift = np.linalg.norm(hand_local_world - initial_grasp[None], axis=-1)
    # Clip outliers before model fitting; all channels have metre/frame units except binary contacts.
    return np.column_stack([np.clip(dist, 0, 1), (dist < THRESHOLD).astype(float),
                            np.clip(dhand, 0, 1), np.clip(dobj, 0, 1),
                            np.clip(drel, 0, 1), np.clip(dlocal, 0, 2),
                            np.clip(grasp_shift, 0, 1)]).astype(np.float32)


def corrupt(window: dict, joints: np.ndarray, nearest: np.ndarray, part: np.ndarray,
            name: str, kind: str, amount: float) -> tuple[dict, np.ndarray, dict]:
    w = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in window.items()}
    j = joints.copy()
    t = len(j)
    cut = max(10, int(t * 0.56))
    ramp = max(4, int(t * 0.09))
    alpha = np.clip((np.arange(t) - cut) / ramp, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    obj = w["obj_trans"]
    center = obj.mean(axis=0)
    direction = j[:cut, [22, 23]].mean(axis=(0, 1)) - center
    direction[2] = 0  # horizontal displacement avoids floor penetration
    direction /= max(np.linalg.norm(direction), 1e-9)
    if kind == "premature_release":
        for h, elbow in [(22, 20), (23, 21)]:
            radial = (j[cut, h] - obj[cut]).copy()
            radial[2] = 0
            radial /= max(np.linalg.norm(radial), 1e-9)
            delta = amount * alpha[:, None] * radial[None]
            j[:, h] += delta
            j[:, elbow] += 0.45 * delta
        params = {"cut_frame": cut, "ramp_frames": ramp, "hand_shift_m": amount}
    elif kind == "object_motion_without_contact":
        # Move all rigid parts together; relative articulated geometry is preserved.
        delta = -amount * alpha[:, None] * direction[None]
        w["obj_trans"] += delta
        if "obj_bottom_trans" in w:
            w["obj_bottom_trans"] += delta
        params = {"cut_frame": cut, "ramp_frames": ramp, "object_shift_m": amount}
    elif kind == "spatial_contact_shift":
        # Slide one hand to one fixed object-local surface vertex. A smooth
        # transition avoids framewise nearest-vertex snapping, and the other
        # hand remains anchored as a stable grasp reference.
        parts = object_parts(name)
        initial_local = contact_local(window, nearest, part, name)
        order = [int(hashlib.sha256(window["seq_name"].encode()).hexdigest()[0], 16) % 2]
        order.append(1 - order[0])
        candidates = []
        for h in order:
            pid = int(part[cut, h])
            _, verts, prefix = parts[pid]
            tree = cKDTree(verts)
            tangent = np.array([1.0, 0.0, 0.0]) if h == 0 else np.array([0.0, 1.0, 0.0])
            scale = np.asarray(window[f"{prefix}_scale"])
            local_shift = amount / max(abs(float(scale[cut])), 1e-6)
            anchor = initial_local[cut, h]
            for sign in (1.0, -1.0):
                desired = anchor + sign * local_shift * tangent
                _, indices = tree.query(desired, k=min(64, len(verts)))
                for idx in np.atleast_1d(indices):
                    actual = float(abs(scale[cut]) * np.linalg.norm(verts[idx] - anchor))
                    if .65 * amount <= actual <= 1.25 * amount:
                        candidates.append((np.linalg.norm(verts[idx] - desired), h, pid, prefix, int(idx), actual))
            if candidates:
                break
        if not candidates:
            params = {"cut_frame": cut, "ramp_frames": ramp, "target_shift_m": amount,
                      "actual_hand_shift_m": 0.0, "hand": -1}
            return w, j, params
        _, h, pid, prefix, idx, actual = min(candidates)
        _, verts, _ = parts[pid]
        target_local = verts[idx]
        scale = np.asarray(window[f"{prefix}_scale"])
        rot = window["obj_rot_mat"] if prefix == "obj" else window["obj_bottom_rot_mat"]
        trans = window[f"{prefix}_trans"]
        target_world = scale[:, None] * np.einsum("i,tji->tj", target_local, rot) + trans
        fixed_offset = joints[cut, 22 + h] - nearest[cut, h]
        desired_hand = target_world + fixed_offset
        delta = alpha[:, None] * (desired_hand - joints[:, 22 + h])
        for joint_id, fraction in ((22 + h, 1.0), (20 + h, .75), (18 + h, .45), (16 + h, .15)):
            j[:, joint_id] += fraction * delta
        params = {"cut_frame": cut, "ramp_frames": ramp, "target_shift_m": amount,
                  "actual_hand_shift_m": actual, "hand": "left" if h == 0 else "right",
                  "target_part": parts[pid][0], "target_vertex": idx}
    else:
        raise ValueError(kind)
    w["motion"] = j.reshape(t, 72)
    return w, j, params


def validate(kind: str, old: np.ndarray, new: np.ndarray, feat: np.ndarray, params: dict,
             original_joints: np.ndarray, changed_joints: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(new).all() or not np.isfinite(feat).all():
        return False, "non_finite"
    hand_delta = changed_joints[:, [22, 23]] - original_joints[:, [22, 23]]
    if np.max(np.linalg.norm(np.diff(hand_delta, axis=0), axis=-1)) > 0.065:
        return False, "hand_shift_too_abrupt"
    old_forearm = np.linalg.norm(original_joints[:, [22, 23]] - original_joints[:, [20, 21]], axis=-1)
    new_forearm = np.linalg.norm(changed_joints[:, [22, 23]] - changed_joints[:, [20, 21]], axis=-1)
    if np.max(np.abs(new_forearm - old_forearm)) > 0.10:
        return False, "forearm_length_change"
    cut = params["cut_frame"]
    if not np.allclose(old[:cut], new[:cut], atol=1e-6):
        return False, "prefix_changed"
    if kind == "spatial_contact_shift":
        # Surface displacement must preserve most contact and move the site.
        if np.mean((new[cut + params["ramp_frames"]:] < THRESHOLD).any(axis=1)) < 0.7:
            return False, "contact_lost"
        if params["actual_hand_shift_m"] < .65 * params["target_shift_m"]:
            return False, "shift_too_small"
    elif kind == "object_motion_without_contact":
        # Require motion while neither hand is near the object, not just a
        # detached hand during an otherwise supported two-hand manipulation.
        moving = feat[1:, 6] > 0.004
        unsupported = np.all(new[1:] > THRESHOLD, axis=1) & moving
        if int(unsupported[cut:].sum()) < 3:
            return False, "no_unsupported_motion_interval"
    else:
        gap = np.mean((new[cut + params["ramp_frames"]:] > 0.075).any(axis=1))
        if gap < 0.5:
            return False, "insufficient_separation"
    return True, "ok"


def main() -> None:
    if not SOURCE_REVIEW.exists():
        raise FileNotFoundError(SOURCE_REVIEW)
    OUT.mkdir(parents=True, exist_ok=True)
    review_bytes = SOURCE_REVIEW.read_bytes()
    rows = list(csv.DictReader(review_bytes.decode("utf-8-sig").splitlines()))
    rows = [r for r in rows if r["status"].strip().lower() == "confirmed"]
    if len({r["sequence"] for r in rows}) != len(rows):
        raise ValueError("Duplicate reviewed sequences")
    split = split_sequences(rows)
    records = {}
    for source in ("train", "test"):
        needed = {r["sequence"] for r in rows if r["split"] == source}
        raw = joblib.load(ROOT / "data" / f"{source}_diffusion_manip_seq_joints24.p")
        records.update({r["seq_name"]: r for r in raw.values() if r["seq_name"] in needed})
        del raw
    if len(records) != len(rows):
        raise ValueError(f"Found {len(records)} of {len(rows)} reviewed source records")
    (OUT / "review_labels.csv").write_bytes(review_bytes)
    entries, rejects = [], []
    for ri, row in enumerate(sorted(rows, key=lambda x: x["sequence"]), 1):
        name, obj = row["sequence"], row["object"]
        window = raw_record_as_window(records[name])
        joints = window["motion"].reshape(-1, 24, 3)
        old, nearest, part = extract_distances(window, joints, obj)
        if len(joints) < 40 or not np.isfinite(old).all():
            rejects.append({"sequence": name, "reason": "invalid_source"})
            continue
        variants = [("clean", "none", 0.0)] + [(k, s, v) for k in TYPES for s, v in LEVELS.items()]
        for kind, severity, amount in variants:
            if kind == "clean":
                w, j, params = window, joints, {"cut_frame": -1, "ramp_frames": 0}
                dist, near, ids = old, nearest, part
            else:
                w, j, params = corrupt(window, joints, nearest, part, obj, kind, amount)
                dist, near, ids = extract_distances(w, j, obj)
            feat = features(w, j, dist, near, ids, obj)
            if kind != "clean":
                ok, reason = validate(kind, old, dist, feat, params, joints, j)
                if not ok:
                    rejects.append({"sequence": name, "type": kind, "severity": severity, "reason": reason})
                    continue
            sample_id = f"{name}__{kind}__{severity}"
            path = OUT / "samples" / f"{sample_id}.npz"
            path.parent.mkdir(exist_ok=True)
            save = {"features": feat, "distances_m": dist.astype(np.float32),
                    "nearest_xyz": near.astype(np.float32), "nearest_part": ids.astype(np.int16),
                    "joints": j.astype(np.float32), "obj_trans": w["obj_trans"].astype(np.float32),
                    "obj_rot_mat": w["obj_rot_mat"].astype(np.float32),
                    "obj_scale": np.asarray(w["obj_scale"], dtype=np.float32)}
            for key in ("obj_bottom_trans", "obj_bottom_rot_mat", "obj_bottom_scale"):
                if key in w:
                    save[key] = np.asarray(w[key], dtype=np.float32)
            np.savez_compressed(path, **save)
            entries.append({"sample_id": sample_id, "source_sequence": name, "object": obj,
                            "source_split": row["split"], "split": split[name], "type": kind,
                            "severity": severity, "label": int(kind != "clean"),
                            "frames": len(j), "parameters": params, "sha256": sha256(path)})
        print(f"{ri}/{len(rows)} {name}: accepted={len(entries)} rejected={len(rejects)}", flush=True)
    # A rerun must leave exactly the indexed samples, including after a rule change.
    indexed_files = {f"{e['sample_id']}.npz" for e in entries}
    samples_dir = (OUT / "samples").resolve()
    for stale in samples_dir.glob("*.npz"):
        if stale.name not in indexed_files:
            if stale.resolve().parent != samples_dir:
                raise ValueError(f"Refusing to remove file outside samples directory: {stale}")
            stale.unlink()
    (OUT / "index.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    (OUT / "rejects.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in rejects), encoding="utf-8")
    manifest = {"dataset": "Counterfactual Dataset v1", "seed": SEED, "threshold_m": THRESHOLD,
                "severity_m": LEVELS, "types": TYPES, "review_sha256": sha256(OUT / "review_labels.csv"),
                "raw_source_sha256": {source: sha256(ROOT / "data" / f"{source}_diffusion_manip_seq_joints24.p")
                                       for source in ("train", "test")},
                "builder_sha256": sha256(Path(__file__)), "sample_count": len(entries),
                "rejected_count": len(rejects), "by_split": dict(Counter(e["split"] for e in entries)),
                "by_type": dict(Counter(e["type"] for e in entries)),
                "index_sha256": sha256(OUT / "index.jsonl")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
