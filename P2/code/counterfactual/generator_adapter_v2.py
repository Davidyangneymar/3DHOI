"""Opt-in geometry adapter with one rotation policy for every input frame rate.

This candidate adapter does not build features, load a checkpoint, or change the
legacy adapter. SO(3) projection repairs accepted rounding error only; rotations
outside the declared tolerance are rejected before any projection takes place.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from counterfactual.generator_adapter import MAX_ALIGNED_FRAMES, REFERENCE_FPS, mesh_paths


ADAPTER_VERSION = 'omomo_raw_so3_v2_candidate'
ROTATION_ATOL = 1e-4


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_array(value, shape, field):
    if np.iscomplexobj(value):
        raise ValueError(f'{field}: complex geometry is unsupported')
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field}: expected finite numeric geometry') from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{field}: expected finite shape {shape}, got {array.shape}')
    return array.copy()


def _rotation_errors(rotation):
    return {
        'orthogonality_max_abs': float(np.max(np.abs(rotation @ rotation.swapaxes(-1, -2) - np.eye(3)))),
        'det_minus_one_max_abs': float(np.max(np.abs(np.linalg.det(rotation) - 1.))),
    }


def _validate_and_project(sample):
    if not isinstance(sample, dict) or 'joints' not in sample:
        raise ValueError('Expected a raw geometry dictionary containing joints')
    joints = np.asarray(sample['joints'])
    if joints.ndim != 3 or joints.shape[0] < 2:
        raise ValueError('joints must have shape (T>=2,24,3)')
    count = len(joints)
    prefixes = ['obj']
    if any(key.startswith('obj_bottom_') for key in sample):
        prefixes.append('obj_bottom')
    allowed = {'joints'} | {f'{prefix}_{suffix}' for prefix in prefixes
                            for suffix in ('trans', 'rot_mat', 'scale')}
    if set(sample) != allowed:
        raise ValueError(f'Raw geometry fields mismatch; missing={sorted(allowed-set(sample))}, '
                         f'unsupported={sorted(set(sample)-allowed)}')
    output = {'joints': _finite_array(joints, (count, 24, 3), 'joints')}
    diagnostics = {}
    for prefix in prefixes:
        for suffix, tail in [('trans', (3,)), ('rot_mat', (3, 3)), ('scale', ())]:
            key = f'{prefix}_{suffix}'
            output[key] = _finite_array(sample[key], (count,) + tail, key)
        rotation_key = f'{prefix}_rot_mat'
        rotation = output[rotation_key]
        before = _rotation_errors(rotation)
        # Explicit absolute tolerances: do not add np.allclose's default rtol.
        if (not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3),
                            atol=ROTATION_ATOL, rtol=0.)
                or not np.allclose(np.linalg.det(rotation), 1., atol=ROTATION_ATOL, rtol=0.)):
            raise ValueError(f'{rotation_key}: rotations must be proper orthonormal matrices '
                             f'within atol={ROTATION_ATOL}, rtol=0 before projection')
        if (output[f'{prefix}_scale'] <= 0).any():
            raise ValueError(f'{prefix}_scale: zero/negative scale is unsupported')
        projected = Rotation.from_matrix(rotation).as_matrix()
        output[rotation_key] = projected
        diagnostics[rotation_key] = {
            'before': before,
            'after_projection': _rotation_errors(projected),
            'projection_max_abs': float(np.max(np.abs(projected-rotation))),
        }
    return output, diagnostics


def _frame_count(count, fps):
    if (isinstance(fps, (bool, np.bool_))
            or not isinstance(fps, (int, float, np.integer, np.floating))):
        raise ValueError('fps must be a finite positive numeric value, not boolean/text')
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    last_reference_index = ((count-1) / fps) * REFERENCE_FPS
    if not np.isfinite(last_reference_index):
        raise ValueError('Resampling exceeds frame budget; check FPS/sequence length')
    aligned_count = int(np.floor(last_reference_index + 1e-8)) + 1
    if aligned_count > MAX_ALIGNED_FRAMES:
        raise ValueError('Resampling exceeds frame budget; check FPS/sequence length')
    if aligned_count < 2:
        raise ValueError('Sequence is shorter than one reference-frame interval')
    return fps, aligned_count


def _align_projected(projected, fps, aligned_count):
    count = len(projected['joints'])
    factor = fps / REFERENCE_FPS
    if factor >= 1 and factor.is_integer():
        # Existing observations are already on the output grid. Do not run a
        # second SO(3) conversion on them, including the 30 FPS identity path.
        return {key: value[::int(factor)].copy() for key, value in projected.items()}
    old = np.arange(count, dtype=np.float64) / fps
    new = np.arange(aligned_count, dtype=np.float64) / REFERENCE_FPS
    # The frame-count epsilon only addresses floating representation of an
    # endpoint. Clamp that sub-nanosecond excess so Slerp never extrapolates.
    new = np.minimum(new, old[-1])
    positions = new * fps
    nearest = np.rint(positions).astype(np.int64)
    at_anchor = np.isclose(positions, nearest, atol=1e-9, rtol=0.)
    at_anchor &= (nearest >= 0) & (nearest < count)
    output = {}
    for key, value in projected.items():
        if key.endswith('_rot_mat'):
            aligned = Slerp(old, Rotation.from_matrix(value))(new).as_matrix()
        else:
            flat = value.reshape(count, -1)
            aligned = np.stack([np.interp(new, old, flat[:, channel])
                                for channel in range(flat.shape[1])], axis=1)
            aligned = aligned.reshape((aligned_count,) + value.shape[1:])
        aligned[at_anchor] = value[nearest[at_anchor]]
        output[key] = aligned
    return output


def normalize_time(sample: dict, fps: float) -> dict:
    """Validate/project all parts, then align to 30 Hz without mutating input.

    Arbitrary input rates retain only the regular reference grid inside the
    original duration. An endpoint between grid ticks is not extrapolated.
    Integer multiples of 30 Hz preserve all selected observed anchors exactly
    after projection; no claim is made about information lost during capture.
    """
    projected, _ = _validate_and_project(sample)
    fps, aligned_count = _frame_count(len(projected['joints']), fps)
    return _align_projected(projected, fps, aligned_count)


def load_aligned_geometry(path: Path, metadata: dict) -> tuple[dict, dict]:
    """Load canonical raw NPZ geometry only; return geometry and audit metadata."""
    if not isinstance(metadata, dict):
        raise ValueError('Metadata must be a dictionary')
    required = {
        'adapter_version': ADAPTER_VERSION, 'input_kind': 'raw', 'units': 'metres',
        'coordinate_system': 'right_handed_z_up', 'joint_schema': 'omomo24',
        'rotation_convention': 'local_to_world_column', 'mesh_frame': 'omomo_original_local',
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f'{key} must explicitly be {expected}')
    if metadata.get('accept_inferred_reference_fps') is not True:
        raise ValueError('OMOMO reference FPS is inferred: accept_inferred_reference_fps must be true')
    if type(metadata.get('fps')) not in (int, float):
        raise ValueError('fps must be an explicit numeric value, not boolean/text')
    name = metadata.get('object')
    if not isinstance(name, str) or not name:
        raise ValueError('An explicit OMOMO object identifier is required')
    paths = mesh_paths(name)
    mesh_hashes = {mesh.name: _sha256(mesh) for mesh in paths}
    if metadata.get('mesh_sha256') != mesh_hashes:
        raise ValueError('Mesh hashes must match the actual OMOMO local meshes')
    prefixes = ['obj', 'obj_bottom'] if name in {'mop', 'vacuum'} else ['obj']
    fields = ['joints'] + [f'{prefix}_{suffix}' for prefix in prefixes
                          for suffix in ('trans', 'rot_mat', 'scale')]
    path = Path(path)
    input_hash = _sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = set(fields)-set(archive.files)
        if missing:
            raise ValueError('Missing raw geometry fields: '+', '.join(sorted(missing)))
        if len(prefixes) == 1 and any(key.startswith('obj_bottom_') for key in archive.files):
            raise ValueError('Unexpected obj_bottom geometry for a single-part object')
        sample = {key: archive[key] for key in fields}
    if _sha256(path) != input_hash:
        raise ValueError('Input changed during loading')
    projected, rotation_diagnostics = _validate_and_project(sample)
    fps, aligned_count = _frame_count(len(projected['joints']), metadata['fps'])
    aligned = _align_projected(projected, fps, aligned_count)
    for key in rotation_diagnostics:
        rotation_diagnostics[key]['after_alignment'] = _rotation_errors(aligned[key])
    return aligned, {
        'adapter_version': ADAPTER_VERSION, 'input_kind': 'raw', 'object': name,
        'input_sha256': input_hash, 'mesh_sha256': mesh_hashes,
        'fps_input': fps, 'fps_reference': REFERENCE_FPS, 'reference_fps_status': 'inferred',
        'input_frames': len(sample['joints']), 'aligned_frames': len(aligned['joints']),
        'input_duration_seconds': (len(sample['joints'])-1)/fps,
        'aligned_duration_seconds': (len(aligned['joints'])-1)/REFERENCE_FPS,
        'rotation_policy': 'validate_then_project_all_parts_before_time_alignment',
        'rotation_atol': ROTATION_ATOL, 'rotation_rtol': 0.,
        'rotation_diagnostics': rotation_diagnostics,
        'time_alignment': 'uniform_30hz_grid_with_observed_anchors_preserved_after_projection',
        'features_built': False, 'checkpoint_loaded': False,
    }
