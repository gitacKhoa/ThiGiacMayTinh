"""Regression: undistortion must not add unequal horizontal/vertical zoom."""
import unittest
import copy
import io
import tempfile
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import cv2
import numpy as np

from src.calibration.undistort import undistort_with_metadata
from src.calibration.image_geometry import save_undistorted_image, load_image_geometry, geometry_sidecar_path
from src.measurement.homography_profile import (undistortion_geometry, create_homography_profile,
                                               homography_for_image, save_homography_profile)
from src.pipeline import main
from src.verify_calibration import make_side_by_side


class UndistortionGeometryTests(unittest.TestCase):
    def test_preserves_intrinsic_aspect_with_strong_distortion(self):
        K = np.array([[980., 0, 606.], [0, 977., 325.], [0, 0, 1.]])
        distortion = np.array([-.38, .12, .003, .001, .0008])
        image = np.zeros((720, 1280, 3), np.uint8)
        for crop in (True, False):
            with self.subTest(crop=crop):
                result = undistort_with_metadata(image, K, distortion, crop=crop,
                                                  calibration_size=(1280, 720))
                sx = result.camera_matrix[0, 0] / K[0, 0]
                sy = result.camera_matrix[1, 1] / K[1, 1]
                self.assertAlmostEqual(sx, sy, places=12)

    def test_projection_crop_and_live_remap_use_same_pixel_frame(self):
        K = np.array([[300., 0, 154.], [0, 298., 116.], [0, 0, 1.]])
        d = np.array([-.3, .08, .002, -.001, 0.])
        image = np.random.default_rng(5).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        result = undistort_with_metadata(image, K, d, calibration_size=(320, 240))
        maps = cv2.initUndistortRectifyMap(K, d, None, result.camera_matrix,
                                         result.image.shape[1::-1], cv2.CV_32FC1)
        live = cv2.remap(image, *maps, cv2.INTER_LINEAR)
        # OpenCV undistort internally uses fixed-point maps; allow interpolation rounding.
        self.assertLess(np.mean(np.abs(live.astype(float) - result.image.astype(float))), 0.1)

    def prepared_scene(self):
        image = np.full((300, 540, 3), 240, np.uint8)
        for row in range(6):
            for col in range(9):
                if (row + col) % 2 == 0:
                    image[30+row*30:60+row*30, 30+col*30:60+col*30] = 10
        image[80:200, 370:470] = 10
        image = cv2.GaussianBlur(image, (5, 5), .8)
        K = np.array([[400., 0, 270.], [0, 400., 150.], [0, 0, 1.]])
        d = np.zeros(5)
        result = undistort_with_metadata(image, K, d, calibration_size=(540, 300))
        return result.image, undistortion_geometry(result, d)

    def test_sidecar_roundtrip_and_same_size_image_tampering(self):
        image, geometry = self.prepared_scene()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'image.png'
            save_undistorted_image(path, image, geometry)
            self.assertEqual(load_image_geometry(path, cv2.imread(str(path))), geometry)
            altered = image.copy()
            altered[0, 0] = 0
            cv2.imwrite(str(path), altered)
            # Even the opt-in flag must not bypass a stale checksum.
            with self.assertRaisesRegex(ValueError, 'does not match'):
                load_image_geometry(path, altered, allow_missing=True)
            geometry_sidecar_path(path).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing image geometry'):
                load_image_geometry(path, altered)
            self.assertIsNone(load_image_geometry(path, altered, allow_missing=True))

    def test_rejects_old_or_missing_h_geometry_even_with_identical_image_size(self):
        image, geometry = self.prepared_scene()
        profile = create_homography_profile(image, 15, (8, 5), preprocessing=geometry)
        homography_for_image(profile, image, preprocessing=geometry)
        legacy = copy.deepcopy(profile)
        legacy['preprocessing'].pop('model')
        legacy['preprocessing'].pop('source_camera_matrix')
        for allow in (False, True):
            with self.assertRaisesRegex(ValueError, 'model changed'):
                homography_for_image(legacy, image, preprocessing=geometry, allow_unverified=allow)
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            homography_for_image(profile, image)
        profile['preprocessing'] = None
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            homography_for_image(profile, image, preprocessing=geometry)
        with self.assertWarns(UserWarning):
            homography_for_image(profile, image, allow_unverified=True)

    def test_strict_cli_saved_image_to_mm_and_exported_geometry(self):
        image, geometry = self.prepared_scene()
        profile = create_homography_profile(image, 15, (8, 5), preprocessing=geometry)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, H = root / 'input.png', root / 'H.json'
            save_undistorted_image(path, image, geometry)
            save_homography_profile(H, profile)
            args = ['--image', str(path), '--already-undistorted', '--homography-file', str(H),
                    '--roi', '330', '40', '180', '210', '--measurement', 'rectangle',
                    '--output', str(root / 'measured')]
            with redirect_stdout(io.StringIO()):
                result = main(args)
            self.assertAlmostEqual(result.measurement.dimensions['length'].value_mm, 60, delta=.2)
            self.assertAlmostEqual(result.measurement.dimensions['width'].value_mm, 50, delta=.2)
            output = root / 'measured' / 'undistorted.png'
            self.assertEqual(load_image_geometry(output, cv2.imread(str(output))), geometry)
            geometry_sidecar_path(path).unlink()
            with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit):
                main(args)
            self.assertIn('Missing image geometry', errors.getvalue())

    def test_known_projection_does_not_turn_frontoparallel_squares_into_rectangles(self):
        K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
        d = np.array([-.38, .12, .003, .001, 0.])
        model = np.zeros((40, 3), np.float64)
        model[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 20 - [70, 40]
        distorted, _ = cv2.projectPoints(model, np.zeros(3), np.array([0., 0, 400.]), K, d)
        for crop in (True, False):
            result = undistort_with_metadata(np.zeros((480, 640, 3), np.uint8), K, d,
                                              crop=crop, calibration_size=(640, 480))
            p = cv2.undistortPoints(distorted, K, d, P=result.camera_matrix).reshape(5, 8, 2)
            sx = np.linalg.norm(np.diff(p, axis=1), axis=2).mean()
            sy = np.linalg.norm(np.diff(p, axis=0), axis=2).mean()
            self.assertAlmostEqual(sx / sy, 1., places=6)

    def test_comparison_pads_without_resizing(self):
        left = np.full((60, 100, 3), 17, np.uint8)
        right = np.full((40, 80, 3), 29, np.uint8)
        comparison = make_side_by_side(left, right)
        np.testing.assert_array_equal(comparison[30:90, :100], left)
        np.testing.assert_array_equal(comparison[30:70, 120:200], right)
        self.assertTrue(np.all(comparison[70:90, 120:200] == 255))


if __name__ == '__main__':
    unittest.main()
