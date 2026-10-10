"""Unsigned triangle-surface evidence. Never infers penetration or physical support."""
from pathlib import Path
import numpy as np


def read_triangle_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            if len(fields) != 4:
                raise ValueError('Only triangular OBJ faces supported; do not silently triangulate')
            ids = [int(x.split('/')[0]) for x in fields[1:]]
            if 0 in ids:
                raise ValueError('OBJ indices cannot be zero')
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in ids])
    v, f = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError('Invalid vertices')
    if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Invalid triangle indices')
    return v, f


def topology(vertices, faces):
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    direction_sum = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1., -1.))
    duplicate = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    inconsistent = int(((counts == 2) & (direction_sum != 0)).sum())
    degenerate = int((np.linalg.norm(normal, axis=1) <= 1e-14).sum())
    return {'vertices': len(vertices), 'triangles': len(faces), 'boundary_edges': boundary,
            'nonmanifold_edges': nonmanifold, 'inconsistent_two_face_edges': inconsistent,
            'duplicate_faces': duplicate, 'degenerate_faces': degenerate,
            'closed_oriented_edge_check_pass': not any((boundary, nonmanifold, inconsistent, duplicate, degenerate)),
            'self_intersections_checked': False,
            'signed_distance_available': False, 'penetration_conclusion': None,
            'note': 'No mesh repair; edge checks alone never establish a valid physical solid.'}


class TriangleSurface:
    """Exact unsigned point-to-triangle distance, brute-force over original faces."""
    def __init__(self, vertices, faces):
        self.triangles = np.asarray(vertices, dtype=float)[faces]
        self.a = self.triangles[:, 0]
        self.u = self.triangles[:, 1] - self.a
        self.v = self.triangles[:, 2] - self.a
        self.normal = np.cross(self.u, self.v)
        self.n2 = np.einsum('ij,ij->i', self.normal, self.normal)
        self.uu = np.einsum('ij,ij->i', self.u, self.u)
        self.uv = np.einsum('ij,ij->i', self.u, self.v)
        self.vv = np.einsum('ij,ij->i', self.v, self.v)
        self.den = self.uu * self.vv - self.uv ** 2

    def query(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite N x 3 points')
        distances, nearest, face_ids = [], [], []
        for p in points:
            w = p - self.a
            n_dot = np.einsum('ij,ij->i', w, self.normal)
            projection = p - (n_dot / np.where(self.n2 > 0, self.n2, 1.))[:, None] * self.normal
            uw = np.einsum('ij,ij->i', w, self.u)
            vw = np.einsum('ij,ij->i', w, self.v)
            safe = np.where(self.den > 0, self.den, 1.)
            s = (self.vv * uw - self.uv * vw) / safe
            t = (self.uu * vw - self.uv * uw) / safe
            inside = (self.den > 0) & (s >= 0) & (t >= 0) & (s + t <= 1)
            d2 = np.sum((projection - p) ** 2, axis=1)
            d2[~inside] = np.inf
            closest = projection.copy()
            for first, second in ((0, 1), (1, 2), (2, 0)):
                a = self.triangles[:, first]
                edge = self.triangles[:, second] - a
                length2 = np.einsum('ij,ij->i', edge, edge)
                frac = np.clip(np.einsum('ij,ij->i', p - a, edge) / np.where(length2 > 0, length2, 1.), 0., 1.)
                candidate = a + frac[:, None] * edge
                candidate_d2 = np.sum((candidate - p) ** 2, axis=1)
                better = candidate_d2 < d2
                d2[better] = candidate_d2[better]
                closest[better] = candidate[better]
            idx = int(np.argmin(d2))
            distances.append(float(np.sqrt(d2[idx])))
            nearest.append(closest[idx])
            face_ids.append(idx)
        return np.asarray(distances), np.asarray(nearest), np.asarray(face_ids)


def world_to_local(points, rotation, translation, scale):
    scale = np.asarray(scale)
    if not np.isfinite(scale).all() or np.any(np.abs(scale) < 1e-12):
        raise ValueError('Invalid or collapsed object scale')
    return np.einsum('ti,tij->tj', points - translation, rotation) / scale[:, None]


def motion_cues(vertices, rotation, translation, scale, joints):
    # z is the existing OMOMO world-up convention. No ground plane is inferred.
    bottom = np.asarray([(s * (vertices @ r.T) + t)[:, 2].min()
                         for r, t, s in zip(rotation, translation, scale)])
    translation_step = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    rel = np.einsum('tji,tjk->tik', rotation[:-1], rotation[1:])
    angle = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.) / 2., -1., 1.)))
    return {'object_lowest_world_z_m': bottom.tolist(),
            'foot_joint_min_world_z_m': joints[:, [10, 11], 2].min(axis=1).tolist(),
            'object_translation_step_m': translation_step.tolist(),
            'object_rotation_step_deg': angle.tolist(),
            'step_indexing': 'element i describes frame i to i+1',
            'registered_ground_plane_available': False,
            'scene_support_geometry_available': False,
            'foot_joint_is_not_sole_or_ground': True,
            'physically_supported': None, 'stable_placement': None}
