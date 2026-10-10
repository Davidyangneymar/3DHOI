"""Untrained, incompatible 17-channel geometry feature candidate.

All coordinates supplied here are world metres. Unlike frozen v1, local-site
steps are measured in metres in each part's rotating frame: no division by
object scale. No continuous channel is clipped. This is an input candidate,
not a new Critic, physical contact detector, slip measurement or phase label.

Channels 9/10 are masked site steps. A zero with channels 13/14 equal to zero
means unavailable, not no slip. Availability requires proximity in both
adjacent frames and an unchanged nearest part. Channels 15/16 retain any part
switch even when contact is absent. First-frame steps and all four new bits
are zero. Grasp-reference channels 11/12 preserve v1's main-part initial
median reference; they are not action- or phase-conditioned grasp errors.

The 17-dimensional schema requires a separately fitted model and normalization.
FEATURE_VERSION here identifies the shared layout, not a contact estimator;
public vertex and triangle-surface candidate outputs must use distinct version
identifiers in the candidate API. Never feed these values to frozen v1.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FEATURE_VERSION = 'omomo_geom17_metric_v2_candidate'
CONTACT_DISTANCE_M = .05
ROTATION_ATOL = 1e-5
CHANNELS = [
    'left_distance_m', 'right_distance_m', 'left_contact', 'right_contact',
    'left_displacement_m_per_frame', 'right_displacement_m_per_frame',
    'object_displacement_m_per_frame', 'left_relative_displacement',
    'right_relative_displacement', 'left_contact_site_displacement_m_per_frame',
    'right_contact_site_displacement_m_per_frame', 'left_grasp_shift',
    'right_grasp_shift', 'left_site_step_valid', 'right_site_step_valid',
    'left_part_changed', 'right_part_changed',
]
UNITS = ['m', 'm', 'binary', 'binary', 'm/frame', 'm/frame', 'm/frame',
         'm/frame', 'm/frame', 'm/frame', 'm/frame', 'm', 'm',
         'binary', 'binary', 'binary', 'binary']


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.shape != shape:
        raise ValueError(f'{name} must be a real numeric array with shape {shape}')
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _transforms(window, frames, name):
    if not isinstance(window, Mapping):
        raise ValueError('window must map transform names to arrays')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('name must identify an object')
    prefixes = ('obj', 'obj_bottom') if name in {'mop', 'vacuum'} else ('obj',)
    out = []
    for prefix in prefixes:
        rk = prefix + '_rot_mat'
        keys = (prefix + '_trans', rk, prefix + '_scale')
        if any(key not in window for key in keys):
            raise ValueError(f'Missing transform arrays for {prefix}')
        trans = _numeric(window[keys[0]], (frames, 3), keys[0])
        rotation = _numeric(window[rk], (frames, 3, 3), rk)
        scale = _numeric(window[keys[2]], (frames,), keys[2])
        if np.any(scale <= 0):
            raise ValueError(f'{prefix} scale must be strictly positive')
        orthogonal = rotation.swapaxes(-1, -2) @ rotation
        determinant = np.linalg.det(rotation)
        if (np.max(np.abs(orthogonal - np.eye(3))) > ROTATION_ATOL
                or np.max(np.abs(determinant - 1)) > ROTATION_ATOL):
            raise ValueError(f'{rk} must contain proper rotations; no implicit projection')
        out.append((trans, rotation, scale))
    return out


def _site_inputs(window, nearest, part, distances_m, name):
    shape = np.asarray(nearest).shape
    if len(shape) != 3 or shape[1:] != (2, 3) or shape[0] < 2:
        raise ValueError('nearest must have shape (T>=2,2,3)')
    frames = shape[0]
    nearest = _numeric(nearest, (frames, 2, 3), 'nearest')
    distances = _numeric(distances_m, (frames, 2), 'distances_m')
    if np.any(distances < 0):
        raise ValueError('distances_m must be nonnegative')
    transforms = _transforms(window, frames, name)
    parts = np.asarray(part)
    if (parts.shape != (frames, 2) or parts.dtype.kind not in 'iu'
            or np.any(parts < 0) or np.any(parts >= len(transforms))):
        raise ValueError('part must contain integer indices of declared object parts')
    return nearest, parts, distances, transforms


def _site_geometry(nearest, parts, distances, transforms):
    local_sites_m = np.zeros_like(nearest)
    for pid, (trans, rotation, _) in enumerate(transforms):
        take = parts == pid
        local = np.einsum('thi,tij->thj', nearest - trans[:, None], rotation)
        local_sites_m[take] = local[take]
    changed = np.zeros(parts.shape, dtype=bool)
    changed[1:] = parts[1:] != parts[:-1]
    touching = distances < CONTACT_DISTANCE_M
    valid = np.zeros(parts.shape, dtype=bool)
    valid[1:] = touching[1:] & touching[:-1] & ~changed[1:]
    same_part_step = np.linalg.norm(np.diff(local_sites_m, axis=0, prepend=local_sites_m[:1]), axis=-1)
    same_part_step[changed] = 0.
    return {'local_sites_m': local_sites_m, 'same_part_step_m': same_part_step,
            'site_step_valid': valid, 'part_changed': changed}


def site_geometry(window, nearest, part, distances_m, name):
    """Return metric site geometry for diagnostics, without contact masking.

    ``same_part_step_m`` has shape (T,2): zero on the first frame and part
    changes; otherwise the metric difference, including non-contact steps.
    ``site_step_valid`` and ``part_changed`` are boolean (T,2) arrays.
    ``local_sites_m`` has shape (T,2,3), with each site's own part origin.
    Local coordinates from different parts must not be directly differenced.
    """
    return _site_geometry(*_site_inputs(window, nearest, part, distances_m, name))


def validate_features(raw):
    """Validate candidate values and explicit missing-step semantics."""
    shape = np.asarray(raw).shape
    if len(shape) != 2 or shape[1] != 17 or shape[0] < 2:
        raise ValueError('v2 candidate features must have shape (T>=2,17)')
    values = _numeric(raw, shape, 'features')
    if np.any(values < 0):
        raise ValueError('Candidate geometry features must be nonnegative')
    if not np.isin(values[:, [2, 3, 13, 14, 15, 16]], [0, 1]).all():
        raise ValueError('Contact, validity and part-change channels must be binary')
    changed = values[:, 15:17].astype(bool)
    valid = values[:, 13:15].astype(bool)
    contact = values[:, 2:4].astype(bool)
    expected = np.zeros_like(valid)
    expected[1:] = contact[1:] & contact[:-1] & ~changed[1:]
    if changed[0].any() or not np.array_equal(valid, expected):
        raise ValueError('Site validity must require consecutive contact and no part change')
    if np.any(values[:, 9:11][~valid] != 0):
        raise ValueError('Unavailable site steps require zero placeholders with validity zero')
    if np.any(values[0, 4:11] != 0):
        raise ValueError('First-frame displacement channels must be zero')
    # Contact is computed before float32 serialization. At the representational
    # rounding boundary, the contact bit retains the higher-precision decision.
    tolerance = np.spacing(np.float32(CONTACT_DISTANCE_M)).item()
    d = values[:, :2]
    if np.any(contact & (d > CONTACT_DISTANCE_M + tolerance)) or np.any(~contact & (d < CONTACT_DISTANCE_M - tolerance)):
        raise ValueError('Contact bits conflict with distances')
    with np.errstate(over='ignore'):
        result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Candidate values exceed finite float32 range')
    return result


def features(window, joints, dist, nearest, part, name):
    """Compute unnormalized, unclipped candidate features from metric inputs.

    All displacement channels are per input frame. FPS normalization belongs
    upstream; this function neither infers frame rate nor resamples time.
    Scale must be positive and rotations proper within 1e-5; transforms are
    validated without correction. Inputs and frozen v1 files are never changed.
    """
    nearest, parts, distances, transforms = _site_inputs(window, nearest, part, dist, name)
    frames = len(nearest)
    joints = _numeric(joints, (frames, 24, 3), 'joints')
    hands = joints[:, [22, 23]]
    translation, rotation, _ = transforms[0]
    relative = hands - translation[:, None]
    geometry = _site_geometry(nearest, parts, distances, transforms)
    def step(array):
        return np.linalg.norm(np.diff(array, axis=0, prepend=array[:1]), axis=-1)
    # This legacy reference is deliberately retained as a geometric quantity,
    # never promoted to an action-conditioned grasp stability measurement.
    hand_local_m = np.einsum('thi,tij->thj', relative, rotation)
    reference = np.median(hand_local_m[:max(5, frames // 5)], axis=0)
    grasp_shift = np.linalg.norm(hand_local_m - reference[None], axis=-1)
    valid = geometry['site_step_valid']
    metric_step = np.where(valid, geometry['same_part_step_m'], 0.)
    result = np.column_stack([distances, distances < CONTACT_DISTANCE_M,
        step(hands), step(translation), step(relative), metric_step, grasp_shift,
        valid, geometry['part_changed']])
    return validate_features(result)
