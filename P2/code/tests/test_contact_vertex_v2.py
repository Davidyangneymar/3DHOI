import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import counterfactual.contact_vertex_v2 as vertex
from counterfactual.diagnose_input_invariance import rigid_transform
from omomo_contact import extract_distances


def geometry(parts=1):
    count = 4
    sample = {'joints': np.zeros((count, 24, 3))}
    sample['joints'][:, 22] = [.1, .1, .1]
    sample['joints'][:, 23] = [.8, .1, .1]
    for prefix in ('obj', 'obj_bottom')[:parts]:
        sample[prefix+'_rot_mat'] = Rotation.from_euler('z', np.arange(count)*8, degrees=True).as_matrix()
        sample[prefix+'_scale'] = np.ones(count)
        sample[prefix+'_trans'] = np.zeros((count, 3))
    return sample


def write_mesh(path, vertices):
    path.write_text(''.join('v '+' '.join(f'{v:g}' for v in row)+'\n' for row in vertices), encoding='utf-8')


def hashes(paths):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class VertexQueryTests(unittest.TestCase):
    def test_matches_legacy_formula_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/'top.obj', Path(td)/'bottom.obj']
            vertices = [np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                        np.array([[0., 0., -.5], [1., 0., -.5]])]
            for path, values in zip(paths, vertices):
                write_mesh(path, values)
            sample = geometry(parts=2)
            sample['obj_bottom_trans'][:, 2] = .2
            saved = copy.deepcopy(sample)
            legacy_parts = [('top', vertices[0], 'obj', cKDTree(vertices[0])),
                            ('bottom', vertices[1], 'obj_bottom', cKDTree(vertices[1]))]
            with patch.object(vertex, 'mesh_paths', return_value=paths), patch('omomo_contact.object_part_trees', return_value=legacy_parts):
                actual = vertex.extract_vertex_distances(sample, sample['joints'], 'vacuum', expected_mesh_sha256=hashes(paths))
                expected = extract_distances(sample, sample['joints'], 'vacuum')
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
            for key in sample:
                np.testing.assert_array_equal(sample[key], saved[key])

    def test_same_size_same_mtime_replacement_refreshes_cache_and_old_hash_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'mesh.obj'
            write_mesh(path, [[0, 0, 0]])
            old = hashes([path])
            stat = path.stat()
            sample = geometry()
            with patch.object(vertex, 'mesh_paths', return_value=[path]):
                before = vertex.extract_vertex_distances(sample, sample['joints'], 'chair', expected_mesh_sha256=old)
                write_mesh(path, [[2, 0, 0]])
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertEqual(path.stat().st_size, stat.st_size)
                self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
                with self.assertRaisesRegex(ValueError, 'expected query contents'):
                    vertex.extract_vertex_distances(sample, sample['joints'], 'chair', expected_mesh_sha256=old)
                after = vertex.extract_vertex_distances(sample, sample['joints'], 'chair', expected_mesh_sha256=hashes([path]))
                self.assertFalse(np.allclose(before[0], after[0]))

    def test_mesh_replaced_after_cache_lookup_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'mesh.obj'
            write_mesh(path, [[0, 0, 0]])
            expected = hashes([path])
            original_cache = vertex._cached_mesh
            def replace_after_lookup(*args):
                result = original_cache(*args)
                write_mesh(path, [[1, 0, 0]])
                return result
            sample = geometry()
            with patch.object(vertex, 'mesh_paths', return_value=[path]), patch.object(vertex, '_cached_mesh', side_effect=replace_after_lookup):
                with self.assertRaisesRegex(ValueError, 'changed during vertex query'):
                    vertex.extract_vertex_distances(sample, sample['joints'], 'chair', expected_mesh_sha256=expected)

    def test_rigid_and_equivalent_native_mesh_scale(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'mesh.obj'
            vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            write_mesh(path, vertices)
            sample = geometry()
            with patch.object(vertex, 'mesh_paths', return_value=[path]):
                baseline = vertex.extract_vertex_distances(sample, sample['joints'], 'chair')
                changed = rigid_transform(sample, 37., [2., -1.5, 0.])
                rigid = vertex.extract_vertex_distances(changed, changed['joints'], 'chair')
                q = Rotation.from_euler('z', 37., degrees=True).as_matrix()
                np.testing.assert_allclose(rigid[0], baseline[0], atol=2e-15, rtol=0.)
                np.testing.assert_allclose(rigid[1], baseline[1]@q.T+[2., -1.5, 0.], atol=2e-15, rtol=0.)
                np.testing.assert_array_equal(rigid[2], baseline[2])
                write_mesh(path, vertices*10)
                scaled = copy.deepcopy(sample)
                scaled['obj_scale'] /= 10
                equivalent = vertex.extract_vertex_distances(scaled, scaled['joints'], 'chair')
                for a, b in zip(equivalent, baseline):
                    np.testing.assert_allclose(a, b, atol=2e-15, rtol=0.)

    def test_first_part_tie_and_invalid_mesh_contents(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td)/'top.obj', Path(td)/'bottom.obj']
            for path in paths:
                write_mesh(path, [[0, 0, 0]])
            sample = geometry(parts=2)
            with patch.object(vertex, 'mesh_paths', return_value=paths):
                _, _, parts = vertex.extract_vertex_distances(sample, sample['joints'], 'vacuum')
                np.testing.assert_array_equal(parts, np.zeros((4, 2), dtype=int))
                for invalid in ('# no vertices\n', 'v 0 0\n', 'v nan 0 0\n', 'v 1+2j 0 0\n'):
                    paths[0].write_text(invalid)
                    with self.assertRaises(ValueError):
                        vertex.extract_vertex_distances(sample, sample['joints'], 'vacuum')

    def test_invalid_input_shape_complex_reflection_and_scale(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'mesh.obj'
            write_mesh(path, [[0, 0, 0]])
            with patch.object(vertex, 'mesh_paths', return_value=[path]):
                for field, bad in [('joints', np.full((4, 24, 3), 1j)), ('joints', np.zeros((0, 24, 3))),
                                   ('joints', np.full((4, 24, 3), 1e300)),
                                   ('obj_scale', np.zeros(4)), ('obj_scale', np.full(4, -1.)),
                                   ('obj_trans', np.full((4, 3), np.nan)), ('obj_trans', np.zeros((3, 3))),
                                   ('obj_rot_mat', np.tile(np.diag([-1., 1., 1.]), (4, 1, 1))),
                                   ('obj_rot_mat', np.tile(np.diag([1.+1e-6, 1., 1.]), (4, 1, 1)))]:
                    sample = geometry()
                    sample[field] = bad
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        vertex.extract_vertex_distances(sample, sample['joints'], 'chair')


if __name__ == '__main__':
    unittest.main()
