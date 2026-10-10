"""Exact unsigned nearest-triangle candidate, independent of frozen v1.

This is geometric proximity, not measured contact, penetration, or support.
Static local meshes are cached. Vertex queries supply only a valid search
upper bound; every returned nearest point is computed on original triangles.
No temporal smoothing, mesh repair, approximate queries, or vertex fallback.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from counterfactual.generator_adapter import mesh_paths
from counterfactual.mesh_surface_evidence import topology


QUERY_VERSION = "nearest_triangle_unsigned_v2"
ROTATION_ATOL = 1e-10  # Upstream v2 must project rotations before any query.


def _real_array(value, field):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{field} must contain real numeric values; no complex coercion")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return array


class TriangleSurfaceV2:
    """Exact point-to-triangle queries with conservative spatial pruning.

    The distance to a referenced vertex bounds the nearest surface distance.
    Any triangle that improves this bound has its centroid within the bound
    plus that triangle's radius. A centroid tree and an AABB lower bound only
    remove triangles that cannot improve the bound. Ties choose the earliest
    original face. Floating-point padding makes pruning conservative.
    """

    def __init__(self, vertices, faces):
        vertices = _real_array(vertices, "vertices")
        faces = np.asarray(faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0
                or not np.isfinite(vertices).all()):
            raise ValueError("Expected nonempty finite N x 3 mesh vertices")
        if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Expected nonempty integer triangular faces; no vertex fallback")
        self.vertices = vertices.copy()
        self.faces = faces.astype(np.int64, copy=True)
        self.triangles = self.vertices[self.faces]
        self.a = self.triangles[:, 0]
        self.u = self.triangles[:, 1] - self.a
        self.v = self.triangles[:, 2] - self.a
        self.normal = np.cross(self.u, self.v)
        self.n2 = np.einsum("ij,ij->i", self.normal, self.normal)
        self.center = self.triangles.mean(axis=1)
        self.radius = np.linalg.norm(self.triangles - self.center[:, None], axis=2).max(axis=1)
        self.low = self.triangles.min(axis=1)
        self.high = self.triangles.max(axis=1)
        if not all(np.isfinite(a).all() for a in (self.n2, self.radius, self.center)):
            raise ValueError("Mesh coordinates overflow finite triangle geometry")
        self.max_radius = float(self.radius.max())
        self.coordinate_scale = float(max(1., np.abs(self.vertices).max()))
        # Unreferenced vertices cannot provide a bound for this surface.
        self.vertex_tree = cKDTree(self.vertices[np.unique(self.faces)])
        self.center_tree = cKDTree(self.center)

    def summary(self):
        info = topology(self.vertices, self.faces)
        info.update(query_version=QUERY_VERSION, distance="unsigned", exact_triangle_queries=True,
                    temporal_smoothing=False, vertex_fallback=False,
                    contact_truth_available=False, support_truth_available=False,
                    unreferenced_vertices_count=int(len(self.vertices) - len(np.unique(self.faces))))
        if hasattr(self, "mesh_sha256"):
            info.update(mesh_path=self.mesh_path, mesh_sha256=self.mesh_sha256)
        return info

    def _closest(self, p, ids):
        a, u, v = self.a[ids], self.u[ids], self.v[ids]
        normal, n2 = self.normal[ids], self.n2[ids]
        w = p - a
        safe_n2 = np.where(n2 > 0, n2, 1.)
        projection = p - (np.einsum("ij,ij->i", w, normal) / safe_n2)[:, None] * normal
        s = np.einsum("ij,ij->i", np.cross(w, v), normal) / safe_n2
        t = np.einsum("ij,ij->i", np.cross(u, w), normal) / safe_n2
        inside = (n2 > 0) & (s >= 0) & (t >= 0) & (s + t <= 1.)
        d2 = np.sum((projection - p) ** 2, axis=1)
        d2[~inside] = np.inf
        nearest = projection.copy()
        triangles = self.triangles[ids]
        # All three segments are evaluated, including zero-length segments.
        # Degenerate triangles therefore reduce to their original edges/points.
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start = triangles[:, first]
            edge = triangles[:, second] - start
            length2 = np.einsum("ij,ij->i", edge, edge)
            frac = np.clip(np.einsum("ij,ij->i", p - start, edge)
                           / np.where(length2 > 0, length2, 1.), 0., 1.)
            candidate = start + frac[:, None] * edge
            candidate_d2 = np.sum((candidate - p) ** 2, axis=1)
            better = candidate_d2 < d2
            nearest[better] = candidate[better]
            d2[better] = candidate_d2[better]
        return d2, nearest

    def query(self, points, *, chunk_size=256):
        """Return distances, closest local points, and original face IDs.

        Bounded point chunks limit the centroid-query temporary storage. Faces
        are visited in original order for deterministic ties. Empty point
        inputs are accepted; an empty mesh never is.
        """
        points = _real_array(points, "query points")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite N x 3 query points")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        distance = np.empty(len(points), dtype=np.float64)
        nearest = np.empty_like(points)
        face_id = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            upper, _ = self.vertex_tree.query(chunk, eps=0.)
            padding = 64 * np.finfo(np.float64).eps * np.maximum(
                self.coordinate_scale, np.maximum(np.abs(chunk).max(axis=1), upper))
            candidates = self.center_tree.query_ball_point(
                chunk, upper + self.max_radius + padding, eps=0., return_sorted=True)
            for offset, (p, upper_bound, pad, candidate_ids) in enumerate(zip(chunk, upper, padding, candidates)):
                ids = np.asarray(candidate_ids, dtype=np.int64)
                gap = np.maximum(np.maximum(self.low[ids] - p, p - self.high[ids]), 0.)
                keep = np.einsum("ij,ij->i", gap, gap) <= (upper_bound + pad) ** 2
                ids = ids[keep]
                if not len(ids):
                    raise ArithmeticError("Conservative triangle search found no face; refusing vertex fallback")
                d2, closest = self._closest(p, ids)
                chosen = int(np.argmin(d2))
                i = start + offset
                distance[i] = np.sqrt(d2[chosen])
                nearest[i] = closest[chosen]
                face_id[i] = ids[chosen]
        if not np.isfinite(distance).all() or not np.isfinite(nearest).all():
            raise ArithmeticError("Nonfinite point-to-triangle result")
        return distance, nearest, face_id


@lru_cache(maxsize=32)
def _surface_from_snapshot(path, digest, content):
    """Parse the exact bytes that were hashed, never a second disk read."""
    vertices, faces = [], []
    for line in content.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Only triangular OBJ faces supported; no silent triangulation")
            ids = [int(x.split("/")[0]) for x in fields[1:]]
            if 0 in ids:
                raise ValueError("OBJ indices cannot be zero")
            faces.append([i - 1 if i > 0 else len(vertices) + i for i in ids])
    surface = TriangleSurfaceV2(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    surface.mesh_path = path
    surface.mesh_sha256 = digest
    return surface


def _object_part_surface_snapshots(name):
    """Internal content-bound query objects; never expose these to callers.

    Each lookup reads and hashes current file bytes; the internal cache key
    binds absolute path, content hash, and the exact parsed snapshot. Replacing
    a file in this process can never reuse stale geometry with a new hash.
    """
    paths = mesh_paths(name)
    parts = [("top", "obj"), ("bottom", "obj_bottom")] if name in {"mop", "vacuum"} else [("whole", "obj")]
    if len(paths) != len(parts):
        raise ValueError("Object part/mesh path count mismatch")
    result = []
    for (label, prefix), path in zip(parts, paths):
        path = Path(path).resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        result.append((label, _surface_from_snapshot(str(path), digest, content), prefix))
    return result


def object_part_surfaces(name):
    """Return independent surface copies for inspection, in v1 part order.

    Caller mutation of vertices, faces, derived arrays or tree data cannot
    corrupt the content-bound objects used by later geometry queries.
    """
    output = []
    for label, cached, prefix in _object_part_surface_snapshots(name):
        surface = TriangleSurfaceV2(cached.vertices, cached.faces)
        surface.mesh_path = cached.mesh_path
        surface.mesh_sha256 = cached.mesh_sha256
        output.append((label, surface, prefix))
    return output


# Public cache controls are useful for isolated tests and explicit release of
# cached local assets. They never disable the per-call content check.
object_part_surfaces.cache_clear = _surface_from_snapshot.cache_clear
object_part_surfaces.cache_info = _surface_from_snapshot.cache_info


def extract_surface_distances(window, joints, name, *, expected_mesh_sha256=None):
    """Return unsigned metres, closest world points and part IDs for both hands.

    Signature/shapes match the legacy nearest-vertex function. Rotations must
    already be proper orthonormal matrices and scales strictly positive. This
    routine never projects rotations or repairs invalid geometry silently.
    When supplied, expected_mesh_sha256 must exactly match filename-to-hash
    values of the snapshots actually queried, not another disk read.
    """
    joints = _real_array(joints, "joints")
    if (joints.ndim != 3 or joints.shape[1:] != (24, 3) or len(joints) < 1
            or not np.isfinite(joints).all()):
        raise ValueError("Expected finite T x 24 x 3 joints")
    hands, count = joints[:, [22, 23]], len(joints)
    distances = np.full((count, 2), np.inf, dtype=np.float64)
    nearest = np.empty((count, 2, 3), dtype=np.float64)
    nearest_part = np.full((count, 2), -1, dtype=np.int32)
    parts = _object_part_surface_snapshots(name)
    if not parts:
        raise ValueError("Object has no triangle surfaces")
    if expected_mesh_sha256 is not None:
        actual_hashes = {Path(surface.mesh_path).name: surface.mesh_sha256 for _, surface, _ in parts
                         if hasattr(surface, "mesh_path") and hasattr(surface, "mesh_sha256")}
        if (not isinstance(expected_mesh_sha256, dict) or len(actual_hashes) != len(parts)
                or expected_mesh_sha256 != actual_hashes):
            raise ValueError("Expected mesh hashes do not match the exact surface snapshots queried")
    for part_id, (_, surface, prefix) in enumerate(parts):
        rotation = _real_array(window[f"{prefix}_rot_mat"], f"{prefix} rotation")
        scale = _real_array(window[f"{prefix}_scale"], f"{prefix} scale")
        translation = _real_array(window[f"{prefix}_trans"], f"{prefix} translation")
        for value, shape, field in ((rotation, (count, 3, 3), "rotation"),
                                    (scale, (count,), "scale"), (translation, (count, 3), "translation")):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{prefix} {field}: expected finite shape {shape}")
        if (np.any(scale <= 0) or not np.allclose(rotation @ rotation.transpose(0, 2, 1),
                                                np.eye(3), rtol=0., atol=ROTATION_ATOL)
                or not np.allclose(np.linalg.det(rotation), 1., rtol=0., atol=ROTATION_ATOL)):
            raise ValueError("Surface query requires positive scale and proper projected rotations")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            local = np.einsum("thi,tij->thj", hands - translation[:, None], rotation) / scale[:, None, None]
        if not np.isfinite(local).all():
            raise ValueError("Hand-to-local transform overflowed finite coordinates")
        d, p, _ = surface.query(local.reshape(-1, 3))
        with np.errstate(over='ignore', invalid='ignore'):
            part_distance = scale[:, None] * d.reshape(count, 2)
            part_nearest = scale[:, None, None] * np.einsum("thi,tji->thj", p.reshape(count, 2, 3), rotation) + translation[:, None]
        if not np.isfinite(part_distance).all() or not np.isfinite(part_nearest).all():
            raise ValueError("Surface query overflowed finite output geometry")
        take = part_distance < distances  # Same first-part tie policy as v1.
        distances[take] = part_distance[take]
        nearest[take] = part_nearest[take]
        nearest_part[take] = part_id
    return distances, nearest, nearest_part
