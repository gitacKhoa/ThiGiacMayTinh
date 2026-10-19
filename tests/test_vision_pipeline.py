"""Numerical and integration checks; no webcam or GUI required.

Run: python -m unittest discover -s tests -p test_vision_pipeline.py -v
"""

import importlib.util
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.calibration.undistort import load_calibration, load_calibration_data, undistort_image, undistort_with_metadata
from src.edge_detection.subpixel_edge import detect_subpixel_edges
from src.measurement.homography import points_to_mm
from src.pipeline import main, process_image, save_result
from src.segmentation.segment import segment_image


ROOT = Path(__file__).resolve().parents[1]


class SegmentationTests(unittest.TestCase):
    def test_roi_coordinates_noise_and_polarity(self):
        image = np.full((130, 170), 230, np.uint8)
        image[40:90, 60:120] = 20
        image[25:27, 45:47] = 20
        for source, polarity in ((image, 'dark'), (255 - image, 'bright')):
            result = segment_image(source, roi=(30, 20, 110, 90), foreground=polarity,
                                   blur_size=1, morphology_size=1)
            self.assertEqual(result.component_count, 1)
            self.assertEqual(result.mask.shape, image.shape)
            self.assertEqual(cv2.countNonZero(result.mask), 3000)
            self.assertEqual(cv2.boundingRect(result.contours[0]), (60, 40, 60, 50))
            self.assertEqual(result.mask[25, 45], 0)

    def test_empty_and_truncated_objects(self):
        self.assertFalse(segment_image(np.full((40, 40), 100, np.uint8)).mask.any())
        image = np.full((60, 60), 255, np.uint8)
        image[15:40, :20] = 0
        self.assertFalse(segment_image(image).mask.any())
        self.assertTrue(segment_image(image, reject_border=False).mask.any())

    def test_preserve_holes_and_multiple_objects(self):
        image = np.full((120, 180), 255, np.uint8)
        cv2.circle(image, (50, 60), 30, 0, -1)
        cv2.circle(image, (50, 60), 12, 255, -1)
        cv2.circle(image, (130, 60), 18, 0, -1)
        result = segment_image(image, keep_largest=False, blur_size=1, morphology_size=1)
        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.mask[60, 50], 0)
        filled = segment_image(image, fill_holes=True)
        self.assertEqual(filled.mask[60, 50], 255)

    def test_hsv_and_hue_wrap(self):
        hsv = np.zeros((80, 120, 3), np.uint8)
        hsv[:, :, 2] = 180
        hsv[20:60, 20:50] = (178, 220, 220)
        hsv[20:60, 70:100] = (2, 220, 220)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = segment_image(image, method='hsv', hsv_lower=(170, 100, 100),
                               hsv_upper=(10, 255, 255), keep_largest=False)
        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.mask[30, 30], 255)
        self.assertEqual(result.mask[30, 80], 255)

    def test_adaptive_and_invalid_input(self):
        image = np.full((100, 100), 210, np.uint8)
        image[30:70, 30:70] = 20
        self.assertTrue(segment_image(image, method='adaptive', fill_holes=True).mask.any())
        for options in ({'roi': (-1, 0, 10, 10)}, {'roi': (0, 0, 500, 500)},
                        {'blur_size': 4}, {'method': 'unknown'}, {'min_area': -1},
                        {'method': 'hsv'}, {'method': 'adaptive', 'adaptive_block_size': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                segment_image(image, **options)
        with self.assertRaises(ValueError):
            segment_image(image.astype(float))


class SubpixelTests(unittest.TestCase):
    def test_analytic_slanted_edge_improves_pixel_localization(self):
        yy, xx = np.mgrid[:160, :160]
        # Several orientations and phases prevent an axis-aligned-only pass.
        for angle, phase in ((0.0, 70.3), (0.35, 95.2), (0.6435011088, 100.3), (1.2, 100.7)):
            normal = np.array([math.cos(angle), math.sin(angle)])
            signed = xx * normal[0] + yy * normal[1] - phase
            image = np.rint(127.5 * (1 + np.vectorize(math.erf)(signed / 1.6))).astype(np.uint8)
            result = detect_subpixel_edges(image)
            error = np.abs(result.points @ normal - phase)
            pixel_error = np.abs(result.pixel_points @ normal - phase)
            with self.subTest(angle=angle):
                self.assertGreater(len(result.points), 100)
                self.assertLess(error.mean(), 0.10)
                self.assertLess(error.mean(), pixel_error.mean())
                self.assertTrue(np.isfinite(result.points).all())
                self.assertLessEqual(np.abs(result.offsets).max(), 0.75)
                np.testing.assert_allclose(np.linalg.norm(result.normals, axis=1), 1)

    def test_curved_edge(self):
        yy, xx = np.mgrid[:180, :180]
        center = np.array([89.3, 90.6])
        radius = 45.4
        signed = np.hypot(xx - center[0], yy - center[1]) - radius
        image = np.rint(127.5 * (1 + np.vectorize(math.erf)(signed / 1.6))).astype(np.uint8)
        result = detect_subpixel_edges(image)
        error = np.abs(np.linalg.norm(result.points - center, axis=1) - radius)
        self.assertGreater(len(error), 200)
        self.assertLess(error.mean(), 0.12)

    def test_mask_gates_edges_without_creating_artificial_edges(self):
        flat = np.full((100, 100), 120, np.uint8)
        mask = np.zeros_like(flat)
        mask[20:80, 20:80] = 255
        self.assertEqual(detect_subpixel_edges(flat, mask).points.shape, (0, 2))
        image = np.full_like(flat, 230)
        image[20:80, 20:80] = 20
        image[40:60, 40:60] = 230  # Texture inside the selected solid silhouette.
        result = detect_subpixel_edges(image, mask)
        points = result.points
        self.assertGreater(len(points), 100)
        self.assertFalse(((points[:, 0] > 35) & (points[:, 0] < 65)
                          & (points[:, 1] > 35) & (points[:, 1] < 65)).any())

    def test_empty_small_and_invalid_inputs(self):
        self.assertEqual(detect_subpixel_edges(np.zeros((3, 3), np.uint8)).points.shape, (0, 2))
        image = np.zeros((30, 30), np.uint8)
        with self.assertRaises(ValueError):
            detect_subpixel_edges(image, np.zeros((10, 10), np.uint8))
        for options in ({'sigma': 0}, {'canny_low': 150, 'canny_high': 50}, {'max_offset': 2}):
            with self.assertRaises(ValueError):
                detect_subpixel_edges(image, **options)


class GeometryAndPipelineTests(unittest.TestCase):
    def test_calibration_aliases_size_mismatch_and_crop(self):
        camera = np.array([[200., 0, 79.5], [0, 201., 59.5], [0, 0, 1]])
        distortion = np.zeros(5)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for extension, aliases in (('.pkl', False), ('.npz', True), ('.npz', False)):
                payload = {'image_size': (160, 120)}
                payload.update({'mtx': camera, 'dist': distortion} if aliases else
                               {'camera_matrix': camera, 'distortion_coefficients': distortion})
                path = directory / ('calib' + extension)
                if extension == '.npz':
                    np.savez(path, **payload)
                else:
                    path.write_bytes(pickle.dumps(payload))
                data = load_calibration_data(path)
                np.testing.assert_equal(load_calibration(path)[0], camera)
                image = np.zeros((120, 160, 3), np.uint8)
                result = undistort_with_metadata(image, data['mtx'], data['dist'], calibration_size=data['image_size'])
                matrix, roi = cv2.getOptimalNewCameraMatrix(camera, distortion, (160, 120), 0)
                x, y, w, h = roi
                self.assertEqual(result.image.shape[:2], (h, w))
                np.testing.assert_allclose(result.camera_matrix[:2, 2], matrix[:2, 2] - [x, y])
                np.testing.assert_array_equal(result.image, undistort_image(image, camera, distortion, calibration_size=(160, 120)))
                with self.assertRaises(ValueError):
                    undistort_image(image, camera, distortion, calibration_size=(320, 240))

    def test_legacy_calibration_warns_when_size_unknown(self):
        with self.assertWarns(UserWarning):
            undistort_image(np.zeros((30, 30), np.uint8), np.array([[30., 0, 15], [0, 30., 15], [0, 0, 1]]), np.zeros(5))

    def test_projective_transform_preserves_subpixel_and_rejects_infinity(self):
        points = np.array([[10.125, 20.875], [32.54321, 52.45678]])
        H = np.array([[0.4, 0, 1], [0, 0.5, -2], [0, 0, 1.]])
        np.testing.assert_allclose(points_to_mm(points, H), points * [0.4, 0.5] + [1, -2], atol=1e-12)
        self.assertEqual(points_to_mm(np.empty((0, 2)), H).shape, (0, 2))
        with self.assertRaises(ValueError):
            points_to_mm(points, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            points_to_mm([[0, 1]], [[1, 0, 1], [0, 1, 0], [1, 0, 0]])

    def test_full_board_object_flow_and_export(self):
        image = np.full((300, 540, 3), 255, np.uint8)
        for row in range(6):
            for col in range(9):
                if (row + col) % 2 == 0:
                    image[30 + row * 30:60 + row * 30, 30 + col * 30:60 + col * 30] = 0
        image[80:200, 370:470] = 0
        image = cv2.GaussianBlur(image, (5, 5), 0.8)
        result = process_image(image, roi=(330, 40, 180, 210), square_mm=15)
        self.assertLess(result.homography_rmse_mm, 0.05)
        self.assertIsNotNone(result.points_mm)
        spans = np.ptp(result.points_mm, axis=0)
        # Known synthetic object spans 100x120 pixels; board scale = 0.5 mm/px.
        np.testing.assert_allclose(np.sort(spans), [50, 60], atol=0.25)
        with tempfile.TemporaryDirectory() as directory:
            save_result(directory, image, result)
            directory = Path(directory)
            metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
            self.assertTrue(metadata['has_mm'])
            with np.load(directory / 'edges.npz') as saved:
                np.testing.assert_array_equal(saved['points_px'], result.edges.points)
            csv = np.loadtxt(directory / 'edges.csv', delimiter=',', skiprows=1)
            self.assertEqual(csv.shape, (len(result.edges.points), 6))
            # Reusing an output directory in pixel mode must not retain old mm data.
            pixel_result = process_image(image, roi=(330, 40, 180, 210))
            save_result(directory, image, pixel_result)
            with np.load(directory / 'edges.npz') as saved:
                self.assertEqual(saved['points_mm'].shape, (0, 2))

    def test_missing_object_and_missing_board_are_errors(self):
        image = np.full((120, 160, 3), 200, np.uint8)
        with self.assertRaisesRegex(ValueError, 'No object'):
            process_image(image)
        image[30:90, 40:120] = 0
        with self.assertRaisesRegex(ValueError, 'Chessboard'):
            process_image(image, square_mm=29)

    def test_cli_and_legacy_calibration_resolution_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = np.full((120, 160, 3), 220, np.uint8)
            image[30:90, 40:120] = 10
            image_path = directory / 'image.png'
            cv2.imwrite(str(image_path), image)
            output = directory / 'result'
            main(['--image', str(image_path), '--already-undistorted', '--roi', '10', '10', '140', '100', '--output', str(output)])
            self.assertTrue((output / 'edges.csv').exists())
            calib = directory / 'legacy.npz'
            np.savez(calib, mtx=np.array([[200., 0, 80], [0, 200., 60], [0, 0, 1]]), dist=np.zeros(5))
            with self.assertRaises(SystemExit) as raised:
                main(['--image', str(image_path), '--calib', str(calib), '--roi', '10', '10', '140', '100'])
            self.assertEqual(raised.exception.code, 1)


class CaptureRegressionTests(unittest.TestCase):
    def test_capture_save_failure_releases_camera(self):
        path = ROOT / 'src/calibration/camera-calibration/capture_calibration_images.py'
        spec = importlib.util.spec_from_file_location('capture_failure', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from unittest.mock import MagicMock
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 120
        cap.read.return_value = (True, np.full((120, 160, 3), 120, np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, 'OUTPUT_DIRECTORY', directory), \
                    patch.object(module.cv2, 'VideoCapture', return_value=cap), \
                    patch.object(module.cv2, 'findChessboardCorners', return_value=(False, None)), \
                    patch.object(module.cv2, 'imshow'), patch.object(module.cv2, 'destroyAllWindows'), \
                    patch.object(module.cv2, 'waitKey', return_value=ord('c')), \
                    patch.object(module.cv2, 'imwrite', return_value=False):
                with self.assertRaises(OSError):
                    module.capture_calibration_images()
                cap.release.assert_called_once()

    def test_webcam_save_paths_preserve_previous_captures(self):
        spec = importlib.util.spec_from_file_location('capture_webcam', ROOT / 'tests/capture_webcam.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'capture_000.jpg'
            existing.write_bytes(b'previous image')
            result = module.build_save_path(directory, 'capture', 0)
            self.assertEqual(Path(result).name, 'capture_001.jpg')
            result = module.build_save_path(str(existing), 'unused', 0)
            self.assertNotEqual(Path(result), existing)
            self.assertEqual(existing.read_bytes(), b'previous image')
            result = module.build_save_path(str(Path(directory) / 'nested' / 'image.png'), 'unused', 0)
            self.assertTrue(Path(result).parent.exists())

    def test_calibration_capture_saves_raw_frame_without_overwrite(self):
        path = ROOT / 'src/calibration/camera-calibration/capture_calibration_images.py'
        spec = importlib.util.spec_from_file_location('capture_calibration', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frame = np.full((120, 160, 3), 120, np.uint8)
        from unittest.mock import MagicMock
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 120
        cap.read.side_effect = [(True, frame.copy()), (True, frame.copy())]
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'calibration_00.jpg'
            original.write_bytes(b'existing capture')
            with patch.object(module, 'OUTPUT_DIRECTORY', directory), \
                    patch.object(module.cv2, 'VideoCapture', return_value=cap), \
                    patch.object(module.cv2, 'findChessboardCorners', return_value=(False, None)), \
                    patch.object(module.cv2, 'imshow'), patch.object(module.cv2, 'destroyAllWindows'), \
                    patch.object(module.cv2, 'waitKey', side_effect=[ord('c'), ord('q')]), \
                    patch.object(module.cv2, 'imwrite', return_value=True) as write:
                module.capture_calibration_images()
                np.testing.assert_array_equal(write.call_args.args[1], frame)
                self.assertTrue(write.call_args.args[0].endswith('calibration_01.jpg'))
                self.assertEqual(original.read_bytes(), b'existing capture')
                cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
