import unittest
import numpy as np
from counterfactual.mesh_surface_evidence import TriangleSurface, topology, world_to_local, motion_cues


class MeshEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.f = np.array([[0, 1, 2]])

    def test_face_interior_is_closer_than_nearest_vertex(self):
        d, p, idx = TriangleSurface(self.v, self.f).query([[.25, .25, .1]])
        np.testing.assert_allclose(d, [.1])
        np.testing.assert_allclose(p, [[.25, .25, 0.]])
        np.testing.assert_array_equal(idx, [0])

    def test_edge_vertex_and_opposite_side_unsigned(self):
        d, _, _ = TriangleSurface(self.v, self.f).query([[.5, -.2, 0.], [2., 0., 0.], [.25, .25, -.1]])
        np.testing.assert_allclose(d, [.2, 1., .1])

    def test_degenerate_triangle_reduces_to_segments(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        d, _, _ = TriangleSurface(v, self.f).query([[.5, 1., 0.]])
        np.testing.assert_allclose(d, [1.])
        self.assertEqual(topology(v, self.f)['degenerate_faces'], 1)

    def test_open_and_nonmanifold_do_not_produce_penetration_label(self):
        info = topology(self.v, self.f)
        self.assertEqual(info['boundary_edges'], 3)
        self.assertFalse(info['closed_oriented_edge_check_pass'])
        self.assertIsNone(info['penetration_conclusion'])
        self.assertFalse(info['signed_distance_available'])
        duplicate = topology(self.v, np.repeat(self.f, 3, axis=0))
        self.assertEqual(duplicate['nonmanifold_edges'], 3)
        self.assertEqual(duplicate['duplicate_faces'], 2)

    def test_closed_tetrahedron_still_does_not_certify_physical_solid(self):
        v = np.vstack([self.v, [0., 0., 1.]])
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        info = topology(v, f)
        self.assertTrue(info['closed_oriented_edge_check_pass'])
        self.assertFalse(info['self_intersections_checked'])
        self.assertIsNone(info['penetration_conclusion'])

    def test_scaled_rotated_local_conversion(self):
        r = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
        t = np.array([[10., 20., 30.]])
        s = np.array([2.])
        local = np.array([[.2, .3, .4]])
        world = 2 * np.einsum('ti,tji->tj', local, r) + t
        np.testing.assert_allclose(world_to_local(world, r, t, s), local)

    def test_stationary_object_is_not_automatically_supported(self):
        r = np.repeat(np.eye(3)[None], 3, axis=0)
        t = np.tile([0., 0., 1.], (3, 1))
        cues = motion_cues(self.v, r, t, np.ones(3), np.zeros((3, 24, 3)))
        np.testing.assert_allclose(cues['object_translation_step_m'], 0.)
        self.assertIsNone(cues['physically_supported'])
        self.assertIsNone(cues['stable_placement'])


if __name__ == '__main__':
    unittest.main()
