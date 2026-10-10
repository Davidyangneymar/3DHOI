import unittest
import numpy as np
from counterfactual.review_mechanics import summarize_arrays


class MechanicalEvidenceTests(unittest.TestCase):
    def test_inclusive_window_and_no_movement_from_outside_window(self):
        d=np.array([[.2,.2],[.01,.06],[.05,.05],[.1,.1]])
        t=np.array([[0,0,0],[10,0,0],[10.03,0,0],[10.03,.04,0]])
        result=summarize_arrays(d,t,1,3)
        self.assertEqual(result['frame_count'],3)
        self.assertEqual(result['both_hands_far_frames'],2)
        self.assertAlmostEqual(result['object_translation_path_m'],.07)
        self.assertAlmostEqual(result['object_max_translation_step_m'],.04)
        self.assertFalse(result['semantic_label_generated'])
        self.assertNotIn('decision',result)

    def test_single_frame_has_no_inferred_velocity(self):
        r=summarize_arrays([[.1,.1]],[[3,2,1]],0,0)
        self.assertEqual(r['object_translation_path_m'],0.)
        self.assertEqual(r['object_max_translation_step_m'],0.)

    def test_invalid_geometry_and_range_rejected(self):
        for d,t,a,b in [([[np.nan,0]],[[0,0,0]],0,0),([[-1,0]],[[0,0,0]],0,0),
                         ([[0,0]],[[0,0]],0,0),([[0,0]],[[0,0,0]],0,2)]:
            with self.assertRaises(ValueError):summarize_arrays(d,t,a,b)
