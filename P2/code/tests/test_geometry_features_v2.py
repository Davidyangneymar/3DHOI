"""Synthetic geometry only; no new human labels or model fitting."""
import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from counterfactual.critic_api import CriticScorer, validate_features as validate_v1
from counterfactual.geometry_features_v2 import CHANNELS, FEATURE_VERSION, UNITS, features, site_geometry, validate_features


def fixture(frames=4, name='box'):
    nearest = np.zeros((frames, 2, 3)); nearest[:, :, 0] = np.arange(frames)[:, None] * .02
    joints = np.zeros((frames, 24, 3)); joints[:, [22, 23]] = nearest + [0, 0, .01]
    window = dict(obj_trans=np.zeros((frames, 3)), obj_rot_mat=np.tile(np.eye(3), (frames, 1, 1)), obj_scale=np.ones(frames))
    if name in {'mop', 'vacuum'}:
        window.update(obj_bottom_trans=np.full((frames, 3), 100.),
                      obj_bottom_rot_mat=np.tile(np.eye(3), (frames, 1, 1)), obj_bottom_scale=np.ones(frames))
    return dict(window=window, joints=joints, dist=np.full((frames, 2), .01), nearest=nearest,
                part=np.zeros((frames, 2), dtype=int), name=name)


class GeometryFeaturesV2Tests(unittest.TestCase):
    def test_contract_known_metric_step_and_first_frame(self):
        out = features(**fixture())
        self.assertEqual(out.shape, (4, 17)); self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(CHANNELS), len(UNITS)); self.assertEqual(len(CHANNELS), 17)
        np.testing.assert_allclose(out[1:, 9:11], .02)
        np.testing.assert_array_equal(out[0, 4:11], 0)
        np.testing.assert_array_equal(out[0, 13:17], 0)
        np.testing.assert_array_equal(out[1:, 13:15], 1)
        np.testing.assert_array_equal(out[:, 15:17], 0)

    def test_representational_mesh_scale_does_not_change_metric_features(self):
        original = fixture(); alternate = copy.deepcopy(original)
        # Same world sites represented by a different native mesh/scale pair.
        alternate['window']['obj_scale'][:] = .025
        native_mesh_sites = original['nearest'] / .025
        alternate['nearest'] = native_mesh_sites * .025
        np.testing.assert_allclose(features(**original), features(**alternate), atol=1e-8)

    def test_time_varying_scale_retains_world_metric_change_without_clipping(self):
        item = fixture()
        scale = np.array([1., 3., 6., 10.]); item['window']['obj_scale'] = scale
        native_site = np.array([1., 0., 0.])
        item['nearest'] = np.tile(scale[:, None, None] * native_site, (1, 2, 1))
        item['joints'][:, [22, 23]] = item['nearest'] + [0, 0, .01]
        out = features(**item)
        np.testing.assert_allclose(out[1:, 9], [2, 3, 4])
        np.testing.assert_allclose(out[1:, 4], [2, 3, 4])
        self.assertGreater(out[:, 11].max(), 1)

    def test_part_origin_switch_does_not_make_fake_jump(self):
        item = fixture(name='mop'); item['part'][1:, 0] = 1
        out = features(**item)
        self.assertEqual(out[1, 15], 1); self.assertEqual(out[1, 13], 0); self.assertEqual(out[1, 9], 0)
        self.assertEqual(out[2, 15], 0); self.assertEqual(out[2, 13], 1)
        self.assertAlmostEqual(out[2, 9], .02, places=7)
        geometry = site_geometry(item['window'], item['nearest'], item['part'], item['dist'], item['name'])
        self.assertEqual(geometry['same_part_step_m'][1, 0], 0)
        self.assertGreater(abs(geometry['local_sites_m'][1, 0, 0]), 99)

    def test_contact_loss_and_reacquisition_preserve_missingness(self):
        item = fixture(frames=5); item['dist'][1:3, 0] = [.05, .06]
        out = features(**item)
        np.testing.assert_array_equal(out[:, 13], [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(out[:4, 9], 0)
        geometry = site_geometry(item['window'], item['nearest'], item['part'], item['dist'], item['name'])
        np.testing.assert_allclose(geometry['same_part_step_m'][1:, 0], .02)

    def test_part_changes_are_preserved_without_contact(self):
        item = fixture(name='vacuum'); item['part'][1:3, 0] = 1; item['dist'][:] = .07
        out = features(**item)
        np.testing.assert_array_equal(out[:, 15], [0, 1, 0, 1])
        np.testing.assert_array_equal(out[:, 13:15], 0)
        np.testing.assert_array_equal(out[:, 9:11], 0)

    def test_global_rigid_transform_preserves_features_and_inputs(self):
        item = fixture(name='vacuum'); item['part'][:, 1] = 1
        before = copy.deepcopy(item); changed = copy.deepcopy(item)
        q = Rotation.from_rotvec([.17, -.28, .41]).as_matrix(); offset = np.array([2., -3., 4.])
        for key in ('joints', 'nearest'): changed[key] = changed[key] @ q.T + offset
        for prefix in ('obj', 'obj_bottom'):
            changed['window'][prefix + '_trans'] = changed['window'][prefix + '_trans'] @ q.T + offset
            changed['window'][prefix + '_rot_mat'] = q @ changed['window'][prefix + '_rot_mat']
        np.testing.assert_allclose(features(**item), features(**changed), atol=2e-7)
        for k, v in item.items():
            if k == 'window':
                for key in v: np.testing.assert_array_equal(v[key], before[k][key])
            elif k != 'name': np.testing.assert_array_equal(v, before[k])

    def test_missing_and_malformed_transform_inputs_rejected(self):
        for name in ('box', 'mop'):
            item = fixture(name=name); del item['window']['obj_scale']
            with self.assertRaises(ValueError): features(**item)
        for key, bad in [('obj_scale', [1, 1, 1, 0]), ('obj_scale', [1, -1, 1, 1]),
                         ('obj_trans', np.zeros((3, 3))), ('obj_scale', np.ones((4, 1)))]:
            item = fixture(); item['window'][key] = bad
            with self.assertRaises(ValueError): features(**item)
        item = fixture(name='mop'); del item['window']['obj_bottom_rot_mat']
        with self.assertRaises(ValueError): features(**item)

    def test_nonproper_rotation_rejected_without_projection(self):
        for rotation in [np.diag([1., 1., -1.]), np.diag([1., 1., 1.1])]:
            item = fixture(); item['window']['obj_rot_mat'][1] = rotation
            with self.assertRaises(ValueError): features(**item)

    def test_invalid_geometry_part_and_feature_values_rejected(self):
        for key, bad in [('joints', np.zeros((4, 23, 3))), ('dist', np.full((4, 2), -.01)),
                         ('nearest', np.full((4, 2, 3), np.nan)), ('part', np.ones((4, 2), int)),
                         ('part', np.zeros((4, 2), float)), ('part', np.zeros((4, 2), bool))]:
            item = fixture(); item[key] = bad
            with self.assertRaises(ValueError): features(**item)
        valid = features(**fixture())
        for row, channel, bad in [(0, 15, 1), (1, 13, 0), (2, 2, .5), (0, 4, .01), (1, 9, np.inf), (1, 0, -1)]:
            changed = valid.copy(); changed[row, channel] = bad
            with self.assertRaises(ValueError): validate_features(changed)

    def test_unavailable_placeholder_cannot_be_confused_with_measured_step(self):
        item = fixture(); item['dist'][1, 0] = .06; out = features(**item)
        out[1, 9] = .02
        with self.assertRaises(ValueError): validate_features(out)

    def test_near_threshold_float32_rounding_keeps_original_contact_bit(self):
        item = fixture(); item['dist'][:] = .05 - 1e-11
        out = features(**item)
        np.testing.assert_array_equal(out[:, 2:4], 1)
        validate_features(out)

    def test_old_scorer_rejects_version_and_shape_without_loading_weights(self):
        old = CriticScorer.__new__(CriticScorer)
        with self.assertRaises(ValueError): old.score_many([], feature_version=FEATURE_VERSION)
        with self.assertRaises(ValueError): validate_v1(features(**fixture()))


if __name__ == '__main__': unittest.main()
