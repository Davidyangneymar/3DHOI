"""Versioned, CPU-only inference for the frozen geometry critic (not a loss)."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from models.interaction_critic import InteractionCritic, MODEL, resample, rule_score

FEATURE_VERSION = "omomo_geom13_v1"
API_VERSION = "critic_inference_v1"
LEGACY_V1_CHECKPOINT_SHA256 = '93797cff65bf2d7b40fee7f0169c02017fb00edb6db1a6829b98aea74159a383'
CHANNELS = ["left_distance_m", "right_distance_m", "left_contact", "right_contact",
            "left_displacement_m_per_frame", "right_displacement_m_per_frame",
            "object_displacement_m_per_frame", "left_relative_displacement",
            "right_relative_displacement", "left_contact_site_displacement",
            "right_contact_site_displacement", "left_grasp_shift", "right_grasp_shift"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_features(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 13 or x.shape[0] < 2:
        raise ValueError("features must have shape (T>=2,13)")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    if (x < 0).any() or not np.isin(x[:, 2:4], [0, 1]).all():
        raise ValueError("v1 geometry features must be nonnegative with binary contact channels")
    return x


def probabilities(logits: np.ndarray, threshold: float) -> list[dict]:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError('Expected finite one-dimensional logits')
    values = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).tolist()
    return [{"p_error": float(p), "p_valid": 1.0 - float(p),
             "error_flag": bool(p >= threshold)} for p in values]


class CriticScorer:
    """Restricted tensor/NumPy loading, including the unchanged legacy checkpoint."""

    def __init__(self, checkpoint: Path = MODEL):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256(self.checkpoint)
        from numpy.core.multiarray import _reconstruct
        safe_numpy = [_reconstruct, np.ndarray, np.dtype, type(np.dtype('float32')), type(np.dtype('float64'))]
        try:
            with torch.serialization.safe_globals(safe_numpy):
                ckpt = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError as error:
            raise ValueError('Checkpoint rejected by restricted loader; no unrestricted fallback') from error
        if not isinstance(ckpt,dict):raise ValueError('Checkpoint must be a dictionary')
        if ckpt.get("channels") != 13:
            raise ValueError("Unsupported checkpoint channel count")
        version = ckpt.get('feature_version')
        if version is None and self.checkpoint_sha256 == LEGACY_V1_CHECKPOINT_SHA256:
            version = FEATURE_VERSION
        if version != FEATURE_VERSION:
            raise ValueError("Checkpoint feature version is incompatible")
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        if self.mean.shape != (1, 13) or self.std.shape != (1, 13):
            raise ValueError("Invalid normalization shape")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid normalization values")
        self.threshold = float(ckpt["threshold"])
        probabilities(np.array([0.0]), self.threshold)
        self.net = InteractionCritic().eval()
        self.net.load_state_dict(ckpt["state_dict"])
        if any(not torch.isfinite(value).all() for value in self.net.state_dict().values()):
            raise ValueError('Nonfinite model parameters')

    def score_many(self, features: list[np.ndarray], *, feature_version: str,
                   batch_size: int = 32) -> list[dict]:
        if feature_version != FEATURE_VERSION:
            raise ValueError(f"Expected feature_version={FEATURE_VERSION}")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        arrays = [validate_features(x) for x in features]
        output = []
        with torch.inference_mode():
            for start in range(0, len(arrays), batch_size):
                chunk = arrays[start:start + batch_size]
                x = np.stack([(resample(a) - self.mean) / self.std for a in chunk])
                logits = self.net(torch.from_numpy(x)).numpy()
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite critic output")
                for prediction, raw in zip(probabilities(logits, self.threshold), chunk):
                    rule = rule_score(raw)
                    output.append({**prediction, "threshold_error": self.threshold,
                                   "rule_score": rule, "rule_flag": rule >= 1.0,
                                   "checkpoint_sha256": self.checkpoint_sha256,
                                   "feature_version": FEATURE_VERSION, "api_version": API_VERSION,
                                   "calibrated_on_real_data": False})
        return output

    def score_features(self, features: np.ndarray, *, feature_version: str) -> dict:
        return self.score_many([features], feature_version=feature_version)[0]
