"""Saved homography integration tests, using synthetic images (no GUI/webcam)."""

import copy
import importlib.util
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.measurement.homography_profile import (
    create_homography_profile, homography_for_image, load_homography_profile,
    save_homography_profile,
)
from src.pipeline import main, process_image
from src.calibration.image_geometry import load_image_geometry


def scene(board=True):
    image = np.full((300, 540, 3), 255, np.uint8)
    if board:
        for row in range(6):
            for col in range(9):
                if (row + col) % 2 == 0:
                    image[30 + row * 30:60 + row * 30, 30 + col * 30:60 + col * 30] = 0
    image[80:200, 370:470] = 0
    return cv2.GaussianBlur(image, (5, 5), 0.8)


class HomographyReuseTests(unittest.TestCase):
    def test_compute_save_then_reuse_without_board(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, target = root / 'board.png', root / 'object.png'
            cv2.imwrite(str(reference), scene())
            cv2.imwrite(str(target), scene(False))
            profile_path = root / 'homography.json'
            output = root / 'result'
            common = ['--already-undistorted', '--allow-unverified-geometry', '--roi', '330', '40', '180', '210',
                      '--homography-file', str(profile_path), '--output', str(output)]
            main(['--image', str(reference), '--homography-mode', 'compute', '--square-mm', '15', *common])
            stored_bytes = profile_path.read_bytes()
            before = np.loadtxt(output / 'edges.csv', delimiter=',', skiprows=1)
            # No mode / no square-mm means reuse, not pixel-only processing.
            with patch('src.measurement.homography_profile.compute_homography', side_effect=AssertionError('No detection in reuse')), \
                    patch('src.pipeline.compute_homography', side_effect=AssertionError('No detection in reuse')), \
                    self.assertWarnsRegex(UserWarning, 'Only image size'):
                main(['--image', str(target), *common])
            after = np.loadtxt(output / 'edges.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(after, before, atol=1e-9)
            np.testing.assert_allclose(np.sort(np.ptp(after[:, -2:], axis=0)), [50, 60], atol=0.25)
            self.assertEqual(profile_path.read_bytes(), stored_bytes)
            meta = json.loads((output / 'metadata.json').read_text())
            self.assertEqual(meta['homography_mode'], 'reuse')
            self.assertIsNone(meta['homography_rmse_mm'])  # Not a new fit on this image.
            self.assertLess(meta['homography_reference']['fit_rmse_mm'], 0.05)
            # A failed recalibration must not replace the last usable H.
            with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as raised:
                main(['--image', str(target), '--homography-mode', 'compute', '--square-mm', '15', *common])
            self.assertEqual(raised.exception.code, 1)
            self.assertIn('Khong phat hien ban co', errors.getvalue())
            self.assertEqual(profile_path.read_bytes(), stored_bytes)

    def test_profile_validation_and_geometry_mismatch(self):
        geometry = dict(source_size=[540, 300], crop_roi=[0, 0, 540, 300], alpha=0.,
                        camera_matrix=[[400., 0, 270.], [0, 400., 150.], [0, 0, 1.]],
                        distortion=[0.] * 5)
        profile = create_homography_profile(scene(), 15, (8, 5), preprocessing=geometry)
        H = homography_for_image(profile, scene(False), preprocessing=geometry)
        result = process_image(scene(False), roi=(330, 40, 180, 210), homography=H)
        np.testing.assert_allclose(np.sort(np.ptp(result.points_mm, axis=0)), [50, 60], atol=0.25)
        with self.assertRaisesRegex(ValueError, 'image size mismatch'):
            homography_for_image(profile, scene(False)[:, :-1], preprocessing=geometry)
        for key in ('source_size', 'crop_roi', 'camera_matrix', 'distortion', 'alpha'):
            altered = copy.deepcopy(geometry)
            altered[key] = (np.asarray(altered[key]) + 1).tolist()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'preprocessing mismatch'):
                homography_for_image(profile, scene(False), preprocessing=altered)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            with self.assertRaisesRegex(FileNotFoundError, 'compute first'):
                load_homography_profile(path)
            for invalid in ({}, {'H': np.zeros((3, 3)).tolist()}, {'H': [[float('nan')]*3]*3},
                            {'units': 'px'}, {'version': 99}, {'image_size': [1.5, 2]}, {'fit_rmse_mm': -1}):
                data = {} if not invalid else {**profile, **invalid}
                path.write_text(json.dumps(data), encoding='utf-8')
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_homography_profile(path)
            path.write_text('not json', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid homography JSON'):
                load_homography_profile(path)

    def test_cli_requires_valid_selection_and_existing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'object.png'
            cv2.imwrite(str(target), scene(False))
            common = ['--image', str(target), '--already-undistorted', '--roi', '330', '40', '180', '210',
                      '--homography-file', str(root / 'missing.json'), '--output', str(root / 'result')]
            for extra, expected in (([], 1), (['--homography-mode', 'compute'], 2),
                                    (['--homography-mode', 'compute', '--square-mm', 'nan'], 2),
                                    (['--homography-mode', 'reuse', '--square-mm', '15'], 2)):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main([*common, *extra])
                self.assertEqual(raised.exception.code, expected)
            self.assertFalse((root / 'result').exists())

    def test_capture_compute_rejects_missing_board_and_reuses_profile(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('capture_homography', root / 'tests/capture_webcam.py')
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        cap = MagicMock()
        calibration = dict(mtx=np.array([[400., 0, 270.], [0, 400., 150.], [0, 0, 1.]]),
                           dist=np.zeros(5), image_size=(540, 300))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'capture.png'
            profile_path = Path(directory) / 'H.json'
            args = ['--calib', 'mock-calibration.npz', '--output', str(output),
                    '--homography-file', str(profile_path)]
            with patch.object(capture, 'load_calibration_data', return_value=calibration), \
                    patch.object(capture, 'open_camera', return_value=cap), \
                    patch.object(capture.cv2, 'imshow'), patch.object(capture.cv2, 'destroyAllWindows'), \
                    patch.object(capture.cv2, 'waitKey', side_effect=[ord('c'), ord('c'), ord('q')]):
                cap.read.side_effect = [(True, scene(False)), (True, scene()), (True, scene())]
                with redirect_stdout(io.StringIO()) as messages:
                    capture.main([*args, '--homography-mode', 'compute', '--square-mm', '15'])
                self.assertIn('Khong phat hien ban co', messages.getvalue())
                self.assertTrue(output.is_file())
                saved = load_homography_profile(profile_path)
                self.assertEqual(load_image_geometry(output, cv2.imread(str(output))), saved['preprocessing'])
                self.assertLess(saved['fit_rmse_mm'], 0.05)
                # Only one capture saved: the boardless attempt was rejected.
                self.assertFalse(output.with_name('capture_001.png').exists())
                cap.release.assert_called_once()
            cap.reset_mock()
            cap.read.side_effect = [(True, scene(False)), (True, scene(False))]
            with patch.object(capture, 'load_calibration_data', return_value=calibration), \
                    patch.object(capture, 'open_camera', return_value=cap), \
                    patch.object(capture, 'create_homography_profile', side_effect=AssertionError('No board detection')), \
                    patch.object(capture.cv2, 'imshow'), patch.object(capture.cv2, 'destroyAllWindows'), \
                    patch.object(capture.cv2, 'waitKey', side_effect=[ord('c'), ord('q')]):
                capture.main([*args, '--homography-mode', 'reuse'])
                self.assertTrue(output.with_name('capture_001.png').exists())
                cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
