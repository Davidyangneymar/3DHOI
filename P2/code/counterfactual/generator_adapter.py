"""Strict adapter for known OMOMO meshes; other generator conventions need A's mapping."""
from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from counterfactual.geometry_features import features
from counterfactual.critic_api import FEATURE_VERSION, sha256, validate_features
from omomo_contact import extract_distances, object_parts
from plot_omomo_alignment import DATA_ROOT

REFERENCE_FPS = 30.0  # Inferred for released OMOMO data, NOT independently confirmed.
MAX_ALIGNED_FRAMES = 100000


def feature_content_hash(features):
    array = np.ascontiguousarray(features, dtype='<f4')
    return hashlib.sha256(str(array.shape).encode('ascii') + array.tobytes()).hexdigest()


def mesh_paths(name: str) -> list[Path]:
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError("Invalid OMOMO object identifier")
    suffixes = ["_top", "_bottom"] if name in {"mop", "vacuum"} else [""]
    return [DATA_ROOT / "captured_objects" / f"{name}_cleaned_simplified{s}.obj" for s in suffixes]


def _finite_shape(value, shape, field):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{field}: expected finite shape {shape}, got {a.shape}")
    return a


def normalize_time(sample: dict, fps: float) -> dict:
    """Interpolate geometry before feature construction; rotations use SO(3) Slerp."""
    if isinstance(fps, (bool, np.bool_)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    n = len(sample["joints"])
    if n < 2:
        raise ValueError("At least two input frames required")
    old = np.arange(n) / fps
    count = old[-1] * REFERENCE_FPS + 1e-8
    if not np.isfinite(count) or count + 1 > MAX_ALIGNED_FRAMES:
        raise ValueError('Resampling exceeds frame budget; check FPS/sequence length or split explicitly')
    new = np.arange(int(np.floor(count)) + 1) / REFERENCE_FPS
    if len(new) < 2:
        raise ValueError("Sequence is shorter than one reference-frame interval")
    if fps == REFERENCE_FPS:
        return sample
    output = {}
    for key, value in sample.items():
        if key.endswith("rot_mat"):
            output[key] = Slerp(old, Rotation.from_matrix(value))(new).as_matrix()
        else:
            flat = value.reshape(n, -1)
            output[key] = np.stack([np.interp(new, old, flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape((len(new),) + value.shape[1:])
    return output


def load_features(path: Path, metadata: dict) -> tuple[np.ndarray, dict]:
    """Metadata must explicitly declare either v1 features or canonical raw geometry."""
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if metadata.get("input_kind") == "features":
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Precomputed features require an explicit v1 feature_version")
        if "features" not in data:
            raise ValueError("Missing features array")
        raw = validate_features(data['features'])
        return raw, {"input_sha256": sha256(path), "input_kind": "features", 'feature_content_sha256': feature_content_hash(raw)}
    if metadata.get("input_kind") != "raw":
        raise ValueError("input_kind must be features or raw")
    required = {"units": "metres", "coordinate_system": "right_handed_z_up",
                "joint_schema": "omomo24", "rotation_convention": "local_to_world_column",
                "mesh_frame": "omomo_original_local"}
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{key} must explicitly be {expected}; other conventions require an adapter")
    if metadata.get("accept_inferred_reference_fps") is not True:
        raise ValueError("OMOMO reference FPS is inferred: acknowledge accept_inferred_reference_fps=true")
    paths = mesh_paths(metadata.get("object", ""))
    hashes = {p.name: sha256(p) for p in paths}
    if metadata.get("mesh_sha256") != hashes:
        raise ValueError("Mesh hashes must match the actual OMOMO local meshes")
    joints = np.asarray(data.get("joints", []))
    if joints.ndim != 3 or joints.shape[0] < 2:
        raise ValueError("joints must have shape (T>=2,24,3)")
    n = len(joints)
    sample = {"joints": _finite_shape(joints, (n, 24, 3), "joints")}
    for _, _, prefix in object_parts(metadata["object"]):
        for suffix, tail in [("trans", (3,)), ("rot_mat", (3, 3)), ("scale", ())]:
            key = f"{prefix}_{suffix}"
            if key not in data:
                raise ValueError(f"Missing {key}")
            sample[key] = _finite_shape(data[key], (n,) + tail, key)
        r = sample[f"{prefix}_rot_mat"]
        if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(r), 1, atol=1e-4):
            raise ValueError("Object rotations must be proper orthonormal matrices")
        if (sample[f"{prefix}_scale"] <= 0).any():
            raise ValueError("Zero/negative scale requires explicit invalid-frame handling; unsupported here")
    if type(metadata.get('fps')) not in (int,float):raise ValueError('fps must be an explicit numeric value, not boolean/text')
    fps = float(metadata['fps'])
    aligned = normalize_time(sample, fps)
    distance, nearest, part = extract_distances(aligned, aligned["joints"], metadata["object"])
    raw = features(aligned, aligned["joints"], distance, nearest, part, metadata["object"])
    return validate_features(raw), {"input_sha256": sha256(path), "input_kind": "raw", 'feature_content_sha256': feature_content_hash(raw),
                                   "fps_input": fps, "fps_reference": REFERENCE_FPS,
                                   "reference_fps_status": "inferred", "mesh_sha256": hashes,
                                   "input_frames": n, "aligned_frames": len(raw)}
