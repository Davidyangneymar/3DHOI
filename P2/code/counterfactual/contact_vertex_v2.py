"""Content-bound nearest-vertex queries for opt-in geometry candidates.

The query formula and first-part tie policy match frozen v1. This module owns
its cache; it never reads or clears the legacy object's name-only tree cache.
Every invocation hashes mesh contents, including cache hits. Geometry must
already have proper projected rotations and positive uniform scales.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from counterfactual.generator_adapter import mesh_paths


SOURCE_QUERY = 'nearest_vertex_content_bound_v2_candidate'
QUERY_VERSION = SOURCE_QUERY
ROTATION_ATOL = 1e-10


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=20)
def _cached_mesh(path_string, expected_digest):
    """Bind parsed vertices to this exact path/content pair, never just a name."""
    path = Path(path_string)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise ValueError('Mesh changed before vertex-cache construction: '+str(path))
    vertices = []
    for line in content.decode('utf-8', errors='replace').splitlines():
        if line.startswith('v '):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError('Incomplete OBJ vertex: '+str(path))
            try:
                vertices.append([float(value) for value in fields[1:4]])
            except ValueError as error:
                raise ValueError('Invalid OBJ vertex: '+str(path)) from error
    values = np.asarray(vertices, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1:] != (3,) or len(values) == 0
            or not np.isfinite(values).all()):
        raise ValueError('Expected nonempty finite mesh vertices: '+str(path))
    if _file_hash(path) != expected_digest:
        raise ValueError('Mesh changed during vertex-cache construction: '+str(path))
    values.setflags(write=False)
    return values, cKDTree(values, copy_data=True)


def _numeric(value, shape, field):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.shape != shape:
        raise ValueError(f'{field}: expected real numeric shape {shape}')
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{field}: geometry must be finite')
    return array


def extract_vertex_distances(window, joints, name, *, expected_mesh_sha256=None):
    """Return distance metres, closest world vertices and part IDs for two hands.

    ``expected_mesh_sha256`` uses OBJ filenames as keys, matching adapter
    provenance. A mismatch or a mesh replacement during querying fails closed.
    Distances remain proximity estimates, not physical contact/penetration.
    """
    if not isinstance(window, Mapping):
        raise ValueError('window must map transform names to arrays')
    if not isinstance(name, str) or not name:
        raise ValueError('An explicit OMOMO object identifier is required')
    shape = np.asarray(joints).shape
    if len(shape) != 3 or shape[1:] != (24, 3) or shape[0] < 1:
        raise ValueError('Expected finite T>=1 x 24 x 3 joints')
    count = shape[0]
    joints = _numeric(joints, (count, 24, 3), 'joints')
    prefixes = ('obj', 'obj_bottom') if name in {'mop', 'vacuum'} else ('obj',)
    transforms = []
    for prefix in prefixes:
        keys = (prefix+'_rot_mat', prefix+'_scale', prefix+'_trans')
        if any(key not in window for key in keys):
            raise ValueError('Missing transform arrays for '+prefix)
        rotation = _numeric(window[keys[0]], (count, 3, 3), keys[0])
        scale = _numeric(window[keys[1]], (count,), keys[1])
        translation = _numeric(window[keys[2]], (count, 3), keys[2])
        if (np.any(scale <= 0)
                or not np.allclose(rotation @ rotation.swapaxes(-1, -2), np.eye(3),
                                   rtol=0., atol=ROTATION_ATOL)
                or not np.allclose(np.linalg.det(rotation), 1., rtol=0., atol=ROTATION_ATOL)):
            raise ValueError('Vertex query requires positive scale and proper projected rotations')
        transforms.append((rotation, scale, translation))
    paths = [Path(path).resolve() for path in mesh_paths(name)]
    if len(paths) != len(prefixes) or len({path.name for path in paths}) != len(paths):
        raise ValueError('Mesh parts do not match the declared object')
    signatures = {path.name: _file_hash(path) for path in paths}
    if expected_mesh_sha256 is not None and expected_mesh_sha256 != signatures:
        raise ValueError('Mesh hashes do not match the expected query contents')
    meshes = [_cached_mesh(str(path), signatures[path.name]) for path in paths]
    hands = joints[:, [22, 23]]
    distances = np.full((count, 2), np.inf, dtype=np.float64)
    nearest = np.empty((count, 2, 3), dtype=np.float64)
    nearest_part = np.full((count, 2), -1, dtype=np.int32)
    for part_id, ((vertices, tree), (rotation, scale, translation)) in enumerate(zip(meshes, transforms)):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            local = np.einsum('thi,tij->thj', hands-translation[:, None], rotation) / scale[:, None, None]
        if not np.isfinite(local).all():
            raise ValueError('Hand-to-local transform overflowed finite coordinates')
        d, indices = tree.query(local.reshape(-1, 3), eps=0.)
        if not np.isfinite(d).all() or np.any(indices >= len(vertices)):
            raise ValueError('Nearest-vertex distance overflowed finite geometry')
        local_nearest = vertices[indices.reshape(count, 2)]
        with np.errstate(over='ignore', invalid='ignore'):
            part_distance = scale[:, None]*d.reshape(count, 2)
            part_nearest = scale[:, None, None]*np.einsum('thi,tji->thj', local_nearest, rotation)+translation[:, None]
        if not np.isfinite(part_distance).all() or not np.isfinite(part_nearest).all():
            raise ValueError('Vertex query overflowed finite output geometry')
        take = part_distance < distances  # Ties retain the first declared part.
        distances[take] = part_distance[take]
        nearest[take] = part_nearest[take]
        nearest_part[take] = part_id
    for path in paths:
        if _file_hash(path) != signatures[path.name]:
            raise ValueError('Mesh changed during vertex query: '+str(path))
    return distances, nearest, nearest_part
