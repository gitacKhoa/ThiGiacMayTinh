"""Numerical geometry checks in mm, independent of any GUI/camera."""
import unittest

import numpy as np

from src.measurement.measure import MeasurementOptions, measure_points


def rectangle(length=80., width=30., angle=0., count=100):
    t = np.linspace(-0.49, 0.49, count)
    p = np.concatenate([np.column_stack((t * length, np.full(count, s * width / 2))) for s in (-1, 1)]
                       + [np.column_stack((np.full(count, s * length / 2), t * width)) for s in (-1, 1)])
    radians = np.radians(angle)
    R = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    return p @ R.T + [100., -50.]


class MeasurementTests(unittest.TestCase):
    def test_bounding_box_rotated_extents_without_false_fit_uncertainty(self):
        for angle in (0, 17.3, 78.2):
            result = measure_points(rectangle(angle=angle), MeasurementOptions(kind='bounding-box'))
            self.assertAlmostEqual(result.dimensions['length'].value_mm, 80, delta=0.001)
            self.assertAlmostEqual(result.dimensions['width'].value_mm, 30, delta=0.001)
            self.assertIsNone(result.fit_rmse_mm)
            self.assertEqual(result.dimensions['length'].fit_standard_uncertainty_mm, 0)
            self.assertTrue(result.inliers.all())

    def test_rotated_rectangles_subpixel_noise_and_outliers(self):
        rng = np.random.default_rng(41)
        for angle in (0, 17.3, 44.7, 78.2, 135.8):
            points = rectangle(angle=angle) + rng.normal(0, 0.02, (400, 2))
            points = np.concatenate([points, rng.uniform([40, -95], [165, 0], (15, 2))])
            rng.shuffle(points)
            with self.subTest(angle=angle):
                result = measure_points(points)
                self.assertAlmostEqual(result.dimensions['length'].value_mm, 80, delta=0.03)
                self.assertAlmostEqual(result.dimensions['width'].value_mm, 30, delta=0.03)
                self.assertLess(result.fit_rmse_mm, 0.03)
                self.assertLess(result.inliers.sum(), len(points))
                self.assertGreater(result.dimensions['length'].fit_standard_uncertainty_mm, 0)

    def test_circle_fit_rotation_translation_and_outliers(self):
        rng = np.random.default_rng(9)
        angles = np.linspace(0, 2*np.pi, 500, endpoint=False)
        center = np.array([1000., -500.])
        p = center + 12.345 * np.column_stack((np.cos(angles), np.sin(angles)))
        p += rng.normal(0, 0.015, p.shape)
        p = np.concatenate([p, center + rng.uniform(-25, 25, (20, 2))])
        result = measure_points(p, MeasurementOptions(kind='circle'))
        self.assertAlmostEqual(result.dimensions['diameter'].value_mm, 24.69, delta=0.01)
        np.testing.assert_allclose(result.center_mm, center, atol=0.01)
        self.assertLess(result.fit_rmse_mm, 0.025)

    def test_incomplete_and_wrong_shape_rejected(self):
        p = rectangle()
        with self.assertRaises(ValueError):
            measure_points(p[:300])  # Missing an entire side.
        a = np.linspace(0, np.pi / 3, 100)
        with self.assertRaises(ValueError):
            measure_points(20 * np.column_stack((np.cos(a), np.sin(a))), MeasurementOptions(kind='circle'))
        a = np.linspace(0, 2*np.pi, 200, endpoint=False)
        ellipse = np.column_stack((20*np.cos(a), 10*np.sin(a)))
        with self.assertRaises(ValueError):
            measure_points(ellipse, MeasurementOptions(kind='circle'))
        with self.assertRaises(ValueError):
            measure_points(ellipse)

    def test_invalid_inputs_and_options(self):
        for p in (np.empty((0, 2)), [[1, 2]] * 30, [[1, float('nan')]] * 30,
                  np.column_stack((np.arange(30), np.arange(30))), np.ones((40, 3))):
            with self.subTest(p=np.shape(p)), self.assertRaises(ValueError):
                measure_points(p)
        for options in ({'kind': 'auto'}, {'max_fit_rmse_mm': -1}, {'min_inlier_fraction': 0.1},
                        {'max_fit_rmse_mm': float('nan')}, {'min_points': 3}):
            with self.assertRaises(ValueError):
                MeasurementOptions(**options)
        self.assertIsNone(measure_points([], MeasurementOptions(kind='none')))


if __name__ == '__main__':
    unittest.main()
