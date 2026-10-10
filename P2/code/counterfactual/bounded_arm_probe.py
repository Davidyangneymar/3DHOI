"""Geometry-only arm probe; no semantic region selection or training labels.

The shoulder and the original wrist-to-hand vector stay fixed per frame. The
elbow lies on the intersection circle of the upper/lower arm length spheres.
The closest point to the original elbow is chosen; ambiguous cases fail closed.
This is a joint-position calculation, not a joint-limit-aware body pose solver.
"""
import numpy as np


def bounded_envelope(frames, start, end, ramp):
    """Inclusive annotation window, quintic entry/exit, exactly zero outside."""
    if any(type(v) is not int for v in (frames, start, end, ramp)):
        raise ValueError('Frame indices and ramp must be integers')
    if not 0 <= start < end < frames or ramp < 1 or end - start < 2 * ramp:
        raise ValueError('Window must contain both ramps')
    time = np.arange(frames, dtype=float)
    u = np.clip(np.minimum((time - start) / ramp, (end - time) / ramp), 0., 1.)
    return u ** 3 * (10. - 15. * u + 6. * u ** 2)


def solve_frame(arm, target_hand, eps=1e-9):
    """Return length-preserving [shoulder, elbow, wrist, hand] or explicit failure."""
    arm = np.asarray(arm, dtype=np.float64)
    target = np.asarray(target_hand, dtype=np.float64)
    if arm.shape != (4, 3) or target.shape != (3,) or not np.isfinite(arm).all() or not np.isfinite(target).all():
        raise ValueError('Invalid arm/target geometry')
    if np.array_equal(target, arm[3]):
        return arm.copy(), None
    shoulder, elbow, wrist, hand = arm
    a = np.linalg.norm(elbow - shoulder)
    b = np.linalg.norm(wrist - elbow)
    if min(a, b, np.linalg.norm(hand - wrist)) <= eps:
        return None, 'degenerate_source_segment'
    new_wrist = target - (hand - wrist)
    direction = new_wrist - shoulder
    distance = np.linalg.norm(direction)
    if distance > a + b + eps:
        return None, 'target_beyond_reach'
    if distance < abs(a - b) - eps:
        return None, 'target_inside_inner_reach_limit'
    if distance <= eps:
        return None, 'ambiguous_folded_arm'
    axis = direction / distance
    along = (a * a - b * b + distance * distance) / (2. * distance)
    center = shoulder + along * axis
    radius = np.sqrt(max(0., a * a - along * along))
    radial = elbow - center
    radial -= np.dot(radial, axis) * axis
    radial_norm = np.linalg.norm(radial)
    if radius > eps and radial_norm <= eps:
        return None, 'ambiguous_elbow_plane'
    new_elbow = center if radius <= eps else center + radius * radial / radial_norm
    return np.stack([shoulder, new_elbow, new_wrist, target]), None


def probe(original, old_candidate, hand, start, end, ramp):
    """All-or-nothing probe: one unsolved frame means no candidate is returned."""
    original = np.asarray(original, dtype=np.float64)
    old = np.asarray(old_candidate, dtype=np.float64)
    if original.shape != old.shape or original.ndim != 3 or original.shape[1:] != (24, 3):
        raise ValueError('Expected paired T x 24 x 3 joint arrays')
    if not np.isfinite(original).all() or not np.isfinite(old).all() or hand not in ('left', 'right'):
        raise ValueError('Invalid joint values or hand')
    alpha = bounded_envelope(len(original), start, end, ramp)
    h = 0 if hand == 'left' else 1
    ids = np.array([16, 18, 20, 22]) + h
    requested = original[:, ids[-1]] + alpha[:, None] * (old[:, ids[-1]] - original[:, ids[-1]])
    changed = original.copy()
    failures = []
    for frame in np.flatnonzero(alpha > 0):
        solved, reason = solve_frame(original[frame, ids], requested[frame])
        if reason:
            failures.append({'frame': int(frame), 'reason': reason})
        else:
            changed[frame, ids] = solved
    diagnostics = {'start_frame': start, 'end_frame': end, 'ramp_frames': ramp,
                   'active_frames': int((alpha > 0).sum()), 'failed_frames': failures,
                   'numerical_tolerance_m': 1e-9, 'numerical_tolerance_is_not_pose_acceptance_threshold': True,
                   'semantic_label': None, 'eligible_for_training': False,
                   'state': 'geometry_solved_semantics_unverified' if not failures else 'rejected_geometry'}
    if failures:
        return None, diagnostics
    source_lengths = np.linalg.norm(np.diff(original[:, ids], axis=1), axis=-1)
    solved_lengths = np.linalg.norm(np.diff(changed[:, ids], axis=1), axis=-1)
    displacement = changed - original
    diagnostics.update(
        max_segment_length_error_m=float(np.abs(source_lengths - solved_lengths).max()),
        max_hand_displacement_m=float(np.linalg.norm(displacement[:, ids[-1]], axis=-1).max()),
        max_joint_displacement_increment_m=float(np.linalg.norm(np.diff(displacement, axis=0), axis=-1).max()),
        outside_window_unchanged=bool(np.array_equal(changed[alpha == 0], original[alpha == 0])),
        shoulder_unchanged=bool(np.array_equal(changed[:, ids[0]], original[:, ids[0]])),
        other_joints_unchanged=bool(np.array_equal(np.delete(changed, ids[1:], axis=1),
                                                 np.delete(original, ids[1:], axis=1))))
    return changed, diagnostics
