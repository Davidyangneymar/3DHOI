"""Geometry-only candidate API. No checkpoint, scoring rule, or training entry.

The nearest-vertex and nearest-surface alternatives have distinct public feature
versions. Both intentionally reject use through the frozen 13-channel scorer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from counterfactual.critic_api import sha256
from counterfactual.generator_adapter import feature_content_hash
from counterfactual.generator_adapter_v2 import ADAPTER_VERSION, load_aligned_geometry
from counterfactual.geometry_features_v2 import CHANNELS, UNITS, features, site_geometry
from counterfactual.contact_vertex_v2 import extract_vertex_distances


API_VERSION = 'geometry_candidate_v2.1'
FEATURE_VERSIONS = {
    'vertex': 'omomo_geom17_metric_vertex_v2_candidate',
    'surface': 'omomo_geom17_metric_surface_v2_candidate',
}
QUERY_VERSIONS = {'vertex': 'nearest_vertex_content_bound_v2_candidate', 'surface': 'nearest_triangle_unsigned_v2'}


def candidate_metadata(raw_metadata, method):
    """Explicitly opt in to a candidate; never infer a model-compatible version."""
    if method not in FEATURE_VERSIONS:
        raise ValueError('Choose vertex or surface explicitly')
    return {**raw_metadata, 'adapter_version': ADAPTER_VERSION,
            'feature_version': FEATURE_VERSIONS[method], 'contact_method': method}


def build_candidate(aligned, name, method, *, expected_mesh_sha256=None):
    if method not in FEATURE_VERSIONS:
        raise ValueError('Unsupported candidate contact method')
    if method == 'vertex':
        distance, nearest, part = extract_vertex_distances(aligned, aligned['joints'], name, expected_mesh_sha256=expected_mesh_sha256)
    else:
        from counterfactual.contact_surface_v2 import extract_surface_distances
        distance, nearest, part = extract_surface_distances(aligned, aligned['joints'], name, expected_mesh_sha256=expected_mesh_sha256)
    raw = features(aligned, aligned['joints'], distance, nearest, part, name)
    traces = site_geometry(aligned, nearest, part, distance, name)
    traces.update(distance_m=distance, nearest_world_m=nearest, nearest_part=part)
    return raw, traces


def load_candidate(path, metadata):
    method = metadata.get('contact_method')
    if method not in FEATURE_VERSIONS or metadata.get('feature_version') != FEATURE_VERSIONS[method]:
        raise ValueError('Candidate feature_version must match its explicit contact_method')
    aligned, provenance = load_aligned_geometry(Path(path), metadata)
    raw, traces = build_candidate(aligned, metadata['object'], method, expected_mesh_sha256=provenance['mesh_sha256'])
    info = {**provenance, 'api_version': API_VERSION,
            'feature_version': FEATURE_VERSIONS[method], 'contact_method': method,
            'query_version': QUERY_VERSIONS[method], 'channels': list(CHANNELS), 'units': list(UNITS),
            'features_built': True,
            'clipping': 'none', 'feature_content_sha256': feature_content_hash(raw),
            'first_frame_site_step_valid': False,
            'invalid_site_step_policy': 'zero placeholder; validity and part-change channels must be used',
            'trained_checkpoint_available': False, 'rule_thresholds_defined': False,
            'training_ready': False, 'semantic_validity_verified': False,
            'signed_distance_or_penetration': False,
            'code_sha256': {p.name: sha256(p) for p in (
                Path(__file__), Path(__file__).with_name('geometry_features_v2.py'),
                Path(__file__).with_name('generator_adapter_v2.py'),
                Path(__file__).with_name('contact_surface_v2.py') if method == 'surface'
                else Path(__file__).with_name('contact_vertex_v2.py'))}}
    return raw, info, traces


def _publish_directory_no_replace(staging, output):
    """Atomically publish a same-filesystem directory without overwriting peers.

    Windows rename rejects an existing destination. Linux needs the explicit
    RENAME_NOREPLACE operation: ordinary POSIX rename may replace an empty
    destination directory. Unsupported platforms fail closed.
    """
    if os.path.lexists(output):
        raise FileExistsError('Export destination already exists: '+str(output))
    if os.name == 'nt':
        os.rename(staging, output)
        return
    if sys.platform.startswith('linux'):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, 'renameat2', None)
        if rename is None:
            raise OSError('Atomic no-replace export requires Linux renameat2 support')
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(output))
        return
    raise OSError('Atomic no-replace directory exports are supported on Windows and Linux only')


def export_candidate(path, metadata, output):
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise ValueError('Choose a new output directory; never overwrite a frozen export')
    raw, info, traces = load_candidate(path, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only this newly created staging directory is owned/cleaned by the call.
    # All artifacts become visible at the final name in one successful rename.
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'.staging-', dir=output.parent) as folder:
        staging = Path(folder)
        np.savez_compressed(staging/'features.npz', features=raw,
                            feature_version=np.asarray(info['feature_version']),
                            adapter_version=np.asarray(ADAPTER_VERSION))
        np.savez_compressed(staging/'geometry_traces.npz', **traces)
        info['artifact_sha256'] = {name: sha256(staging/name)
                                   for name in ('features.npz', 'geometry_traces.npz')}
        (staging/'metadata.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
        try:
            _publish_directory_no_replace(staging, output)
        except FileExistsError as error:
            raise ValueError('Export destination appeared concurrently; it was not overwritten') from error
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    info = export_candidate(args.input, json.loads(args.metadata.read_text(encoding='utf-8-sig')), args.output)
    print(json.dumps({k: info[k] for k in ('feature_version', 'training_ready', 'trained_checkpoint_available')}, indent=2))
