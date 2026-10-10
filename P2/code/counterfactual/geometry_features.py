"""Frozen geom13 feature definition, copied verbatim from v1 for portable inference.
No dataset builder or private attachment path is imported. Units remain metres/frame.
"""
from __future__ import annotations
import numpy as np
from omomo_contact import object_parts
THRESHOLD = 0.05


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


