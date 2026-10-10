import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from counterfactual.contact_surface_v2 import TriangleSurfaceV2, extract_surface_distances, object_part_surfaces
from counterfactual.mesh_surface_evidence import TriangleSurface


class ContactSurfaceV2Tests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.faces = np.array([[0, 1, 2]])
        self.surface = TriangleSurfaceV2(self.vertices, self.faces)

    @staticmethod
    def geometry(points):
        count = len(points)
        joints = np.zeros((count, 24, 3))
        joints[:, 22] = points
        joints[:, 23] = points
        sample = {"obj_trans": np.zeros((count, 3)), "obj_scale": np.ones(count),
                  "obj_rot_mat": np.repeat(np.eye(3)[None], count, axis=0)}
        return sample, joints

    def query_geometry(self, sample, joints, surface=None):
        with patch("counterfactual.contact_surface_v2._object_part_surface_snapshots", return_value=[("whole", surface or self.surface, "obj")]):
            return extract_surface_distances(sample, joints, "test")

    def test_known_face_edge_vertex_and_unsigned_sides(self):
        points = [[.25, .25, .1], [.5, -.2, 0.], [2., 0., 0.], [.25, .25, -.1]]
        d, p, face = self.surface.query(points)
        np.testing.assert_allclose(d, [.1, .2, 1., .1])
        np.testing.assert_allclose(p, [[.25, .25, 0.], [.5, 0., 0.], [1., 0., 0.], [.25, .25, 0.]])
        np.testing.assert_array_equal(face, [0, 0, 0, 0])
        self.assertFalse(self.surface.summary()["support_truth_available"])

    def test_degenerate_faces_preserve_edges_and_points(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        d, p, _ = TriangleSurfaceV2(vertices, self.faces).query([[.5, 1., 0.]])
        np.testing.assert_allclose(d, [1.])
        np.testing.assert_allclose(p, [[.5, 0., 0.]])
        d, p, _ = TriangleSurfaceV2(vertices, np.array([[1, 1, 1]])).query([[.5, 1., 0.]])
        np.testing.assert_allclose(d, [np.sqrt(1.25)])
        np.testing.assert_allclose(p, [[1., 0., 0.]])

    def test_spatial_pruning_matches_brute_force_random_mesh_and_queries(self):
        rng = np.random.default_rng(871)
        vertices = rng.normal(size=(90, 3))
        faces = np.arange(90).reshape(-1, 3)
        points = rng.normal(size=(137, 3)) * 2
        expected = TriangleSurface(vertices, faces).query(points)
        surface = TriangleSurfaceV2(vertices, faces)
        actual = surface.query(points, chunk_size=7)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual[2], expected[2])

    def test_unreferenced_vertex_does_not_prune_valid_surface(self):
        surface = TriangleSurfaceV2(np.vstack([self.vertices, [.2, .2, 1.]]), self.faces)
        d, p, _ = surface.query([[.2, .2, 1.]])
        np.testing.assert_allclose(d, [1.])
        np.testing.assert_allclose(p, [[.2, .2, 0.]])

    def test_empty_faces_invalid_faces_and_missing_obj_faces_rejected(self):
        for faces in (np.empty((0, 3), int), np.array([[0., 1., 2.]]), np.array([[0, 1, 9]])):
            with self.assertRaises(ValueError):
                TriangleSurfaceV2(self.vertices, faces)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n", encoding="utf-8")
            with patch("counterfactual.contact_surface_v2.mesh_paths", return_value=[path]):
                object_part_surfaces.cache_clear()
                with self.assertRaises(ValueError):
                    object_part_surfaces("empty_test")
                object_part_surfaces.cache_clear()

    def test_cache_tracks_exact_file_content_even_same_length_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mesh.obj"
            original = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            changed = "v 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\n"
            path.write_text(original, encoding="utf-8")
            with patch("counterfactual.contact_surface_v2.mesh_paths", return_value=[path]):
                object_part_surfaces.cache_clear()
                first = object_part_surfaces("cache_test")[0][1]
                second_copy = object_part_surfaces("cache_test")[0][1]
                self.assertIsNot(first, second_copy)
                self.assertEqual(first.mesh_sha256, second_copy.mesh_sha256)
                expected = {path.name: first.mesh_sha256}
                sample, joints = self.geometry([[.2, .2, .1]])
                extract_surface_distances(sample, joints, "cache_test", expected_mesh_sha256=expected)
                path.write_text(changed, encoding="utf-8")
                second = object_part_surfaces("cache_test")[0][1]
                self.assertIsNot(first, second)
                self.assertNotEqual(first.mesh_sha256, second.mesh_sha256)
                np.testing.assert_allclose(first.query([[.2, .2, 0.]])[0], [0.])
                np.testing.assert_allclose(second.query([[.2, .2, 0.]])[0], [1.])
                with self.assertRaises(ValueError):
                    extract_surface_distances(sample, joints, "cache_test", expected_mesh_sha256=expected)
                extract_surface_distances(sample, joints, "cache_test", expected_mesh_sha256={path.name: second.mesh_sha256})
                object_part_surfaces.cache_clear()

    def test_global_rigid_and_equivalent_mesh_scale_encoding_invariant(self):
        sample, joints = self.geometry([[.25, .25, .1], [.3, .2, .2]])
        baseline = self.query_geometry(sample, joints)
        q = Rotation.from_euler("xyz", [17, -25, 61], degrees=True).as_matrix()
        translation = np.array([3., -1., .2])
        moved = {k: v.copy() for k, v in sample.items()}
        moved["obj_trans"] = sample["obj_trans"] @ q.T + translation
        moved["obj_rot_mat"] = q @ sample["obj_rot_mat"]
        actual = self.query_geometry(moved, joints @ q.T + translation)
        np.testing.assert_allclose(actual[0], baseline[0], atol=1e-14)
        np.testing.assert_allclose(actual[1], baseline[1] @ q.T + translation, atol=1e-14)
        scaled = {k: v.copy() for k, v in sample.items()}
        scaled["obj_scale"] /= 40.
        encoded = self.query_geometry(scaled, joints, TriangleSurfaceV2(self.vertices * 40., self.faces))
        np.testing.assert_allclose(encoded[0], baseline[0], atol=1e-14)
        np.testing.assert_allclose(encoded[1], baseline[1], atol=1e-14)

    def test_part_ties_keep_first_part_and_nearest_part_changes(self):
        sample, joints = self.geometry([[.2, .2, .1], [.2, .2, .3]])
        for suffix in ("trans", "rot_mat", "scale"):
            sample[f"obj_bottom_{suffix}"] = sample[f"obj_{suffix}"].copy()
        with patch("counterfactual.contact_surface_v2._object_part_surface_snapshots", return_value=[("top", self.surface, "obj"), ("bottom", self.surface, "obj_bottom")]):
            _, _, part = extract_surface_distances(sample, joints, "vacuum")
            np.testing.assert_array_equal(part, 0)
            sample["obj_bottom_trans"][:, 2] = .3
            _, _, part = extract_surface_distances(sample, joints, "vacuum")
            np.testing.assert_array_equal(part[:, 0], [0, 1])

    def test_nonproper_rotation_nonpositive_scale_and_invalid_queries_rejected(self):
        sample, joints = self.geometry([[.2, .2, .1]])
        for scale in (0., -1., np.nan):
            sample["obj_scale"][:] = scale
            with self.assertRaises(ValueError):
                self.query_geometry(sample, joints)
        sample["obj_scale"][:] = 1.
        for value in (-1., 1.000001):
            sample["obj_rot_mat"][0, 0, 0] = value
            with self.assertRaises(ValueError):
                self.query_geometry(sample, joints)
        for points in ([[np.nan, 0, 0]], [0, 0, 0]):
            with self.assertRaises(ValueError):
                self.surface.query(points)
        d, p, ids = self.surface.query(np.empty((0, 3)))
        self.assertEqual(p.shape, (0, 3))
        self.assertEqual(d.shape, ids.shape)

    def test_surface_removes_vertex_jump_but_preserves_real_motion(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        surface = TriangleSurfaceV2(vertices, np.array([[0, 1, 2], [0, 2, 3]]))
        points = np.array([[.499, .2, .01], [.501, .2, .01], [.9, .2, .01]])
        _, vertex_ids = cKDTree(vertices).query(points)
        vertex_steps = np.linalg.norm(np.diff(vertices[vertex_ids], axis=0), axis=1)
        _, nearest, _ = surface.query(points)
        surface_steps = np.linalg.norm(np.diff(nearest, axis=0), axis=1)
        np.testing.assert_allclose(vertex_steps, [1., 0.])
        np.testing.assert_allclose(surface_steps, [.002, .399], atol=1e-14)
        # No averaging: the .399 m change is retained on the same frame.
        np.testing.assert_allclose(nearest[:, :2], points[:, :2], atol=1e-14)

    def test_public_mesh_arrays_and_trees_cannot_mutate_query_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.obj'
            path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')
            with patch('counterfactual.contact_surface_v2.mesh_paths', return_value=[path]):
                first = object_part_surfaces('cache_test')[0][1]
                peer = object_part_surfaces('cache_test')[0][1]
                expected = {path.name: first.mesh_sha256}
                for key, value in first.__dict__.items():
                    if isinstance(value, np.ndarray):
                        self.assertFalse(np.shares_memory(value, peer.__dict__[key]), key)
                for key in ('vertex_tree', 'center_tree'):
                    self.assertFalse(np.shares_memory(getattr(first, key).data, getattr(peer, key).data))
                first.triangles[:, :, 2] += 1.
                first.vertices[:] += 10.
                first.u[:] = 0.
                first.center_tree.data[:] += 20.
                first.vertex_tree.data[:] += 30.
                first.faces[:] = 0
                sample, joints = self.geometry([[.2, .2, .1]])
                d, p, _ = extract_surface_distances(sample, joints, 'cache_test', expected_mesh_sha256=expected)
                np.testing.assert_allclose(d, [[.1, .1]])
                np.testing.assert_allclose(p[0], [[.2, .2, 0.], [.2, .2, 0.]])
                np.testing.assert_allclose(peer.query([[.2, .2, .1]])[0], [.1])

    def test_complex_values_rejected_before_any_real_cast(self):
        with self.assertRaisesRegex(ValueError, 'real numeric'):
            TriangleSurfaceV2(self.vertices.astype(complex)+1j, self.faces)
        with self.assertRaisesRegex(ValueError, 'real numeric'):
            self.surface.query(np.array([[.2, .2, .1]], complex)+1j)
        sample, joints = self.geometry([[.2, .2, .1]])
        with self.assertRaisesRegex(ValueError, 'real numeric'):
            self.query_geometry(sample, joints.astype(complex)+1j)
        for field in ('obj_trans', 'obj_rot_mat', 'obj_scale'):
            changed = {key: value.copy() for key, value in sample.items()}
            changed[field] = changed[field].astype(complex)+1j
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'real numeric'):
                self.query_geometry(changed, joints)

    def test_finite_inputs_cannot_return_overflowed_world_points(self):
        surface = TriangleSurfaceV2(np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]]), self.faces)
        sample, joints = self.geometry([[1e308, 0., 0.]])
        sample['obj_trans'][:, 0] = 1e308
        sample['obj_scale'][:] = 1e308
        with self.assertRaisesRegex(ValueError, 'finite output'):
            self.query_geometry(sample, joints, surface)


if __name__ == "__main__":
    unittest.main()
