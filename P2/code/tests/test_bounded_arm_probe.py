import unittest
import numpy as np
from counterfactual.bounded_arm_probe import bounded_envelope, probe, solve_frame


class BoundedArmTests(unittest.TestCase):
    def setUp(self):
        self.arm = np.array([[0., 0., 0.], [.3, .4, 0.], [.6, 0., 0.], [.7, 0., 0.]])

    def test_exact_lengths_target_and_shoulder_for_reachable_target(self):
        result, error = solve_frame(self.arm, np.array([.65, .05, .1]))
        self.assertIsNone(error)
        np.testing.assert_allclose(np.linalg.norm(np.diff(result, axis=0), axis=1),
                                   np.linalg.norm(np.diff(self.arm, axis=0), axis=1), atol=1e-12)
        np.testing.assert_array_equal(result[0], self.arm[0])
        np.testing.assert_array_equal(result[-1], [.65, .05, .1])

    def test_noop_preserves_original_elbow_and_all_coordinates(self):
        result, error = solve_frame(self.arm, self.arm[-1])
        self.assertIsNone(error)
        np.testing.assert_array_equal(result, self.arm)

    def test_unreachable_is_rejected_not_clamped(self):
        result, error = solve_frame(self.arm, [2., 0., 0.])
        self.assertIsNone(result)
        self.assertEqual(error, 'target_beyond_reach')

    def test_ambiguous_elbow_plane_is_rejected(self):
        straight = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.], [1.1, 0., 0.]])
        result, error = solve_frame(straight, [.8, 0., 0.])
        self.assertIsNone(result)
        self.assertEqual(error, 'ambiguous_elbow_plane')

    def test_envelope_has_exact_boundaries_and_plateau(self):
        w = bounded_envelope(40, 5, 30, 5)
        np.testing.assert_array_equal(w[:6], 0.)
        np.testing.assert_array_equal(w[30:], 0.)
        np.testing.assert_array_equal(w[10:26], 1.)
        self.assertTrue(np.all(np.diff(w[5:11]) > 0))
        self.assertTrue(np.all(np.diff(w[25:31]) < 0))

    def test_short_window_noninteger_and_nonfinite_fail(self):
        for args in [(30, 5, 10, 4), (30, -1, 20, 4), (30, 5, 30, 4), (30, 5, 20, True)]:
            with self.assertRaises(ValueError):
                bounded_envelope(*args)
        with self.assertRaises(ValueError):
            solve_frame(self.arm, [float('nan'), 0., 0.])

    def test_both_hands_boundaries_and_unaffected_joints(self):
        for hand, h in [('left', 0), ('right', 1)]:
            a = np.zeros((40, 24, 3)); ids = np.array([16, 18, 20, 22]) + h
            a[:, ids] = self.arm
            old = a.copy(); old[:, 22 + h] += [-.05, .05, .1]
            result, diagnostic = probe(a, old, hand, 5, 30, 5)
            self.assertIsNotNone(result)
            self.assertTrue(diagnostic['outside_window_unchanged'])
            self.assertTrue(diagnostic['shoulder_unchanged'])
            self.assertTrue(diagnostic['other_joints_unchanged'])
            self.assertLess(diagnostic['max_segment_length_error_m'], 1e-12)
            self.assertIsNone(diagnostic['semantic_label'])
            self.assertFalse(diagnostic['eligible_for_training'])

    def test_one_unreachable_frame_rejects_entire_probe(self):
        a = np.zeros((40, 24, 3)); a[:, [16, 18, 20, 22]] = self.arm
        old = a.copy(); old[15, 22] = [2., 0., 0.]
        result, diagnostic = probe(a, old, 'left', 5, 30, 5)
        self.assertIsNone(result)
        self.assertEqual(diagnostic['failed_frames'], [{'frame': 15, 'reason': 'target_beyond_reach'}])


if __name__ == '__main__':
    unittest.main()
