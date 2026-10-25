"""Headless API, perspective geometry, config, QC, CLI and export integration."""
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.configuration import config_from_dict, config_to_dict, load_config, save_config
from src.inspection import InspectionConfig, InspectionCancelled, inspect_image, prepare_image
from src.measurement.measure import MeasurementOptions
from src.measurement.homography_profile import COORDINATE_FRAME, save_homography_profile
from src.measurement.uncertainty import UncertaintyBudget
from src.pipeline import main
from src.qc.tolerance_check import ToleranceSpec
from src.reporting import inspection_to_dict, render_inspection, save_inspection


def rectangle_image():
    image = np.full((220, 340, 3), 240, np.uint8)
    image[60:160, 70:270] = 10
    return cv2.GaussianBlur(image, (5, 5), 0.8)


class InspectionTests(unittest.TestCase):
    def test_bounding_box_of_irregular_object_unknown_uncertainty(self):
        image = np.full((220, 340, 3), 240, np.uint8)
        image[60:160, 70:120] = 10
        image[130:160, 70:270] = 10
        image = cv2.GaussianBlur(image, (5, 5), 0.8)
        config = InspectionConfig(measurement=MeasurementOptions(kind='bounding-box'),
                                  tolerances={'length': ToleranceSpec(80, -0.5, 0.5)})
        result = inspect_image(image, np.diag([0.4, 0.4, 1.]), config)
        self.assertAlmostEqual(result.measurement.dimensions['length'].value_mm, 80, delta=0.2)
        self.assertAlmostEqual(result.measurement.dimensions['width'].value_mm, 40, delta=0.2)
        self.assertEqual(result.qc_status, 'NOT_VALIDATED')
        report = inspection_to_dict(result)
        self.assertFalse(report['uncertainties']['length']['available'])
        self.assertIsNone(report['uncertainties']['length']['expanded_mm'])
        with self.assertRaises(ValueError):
            inspect_image(image, np.diag([0.4, 0.4, 1.]))

    def test_cli_config_override_and_repeatable_tolerances(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image_path = folder / 'input.png'
            cv2.imwrite(str(image_path), rectangle_image())
            profile_path = folder / 'homography.json'
            save_homography_profile(profile_path, dict(version=1, units='mm',
                coordinate_frame=COORDINATE_FRAME, H=np.diag([0.4, 0.4, 1.]).tolist(),
                image_size=[340, 220], pattern_size=[8, 5], square_mm=29,
                fit_rmse_mm=0.1, preprocessing=None))
            config_path = folder / 'config.json'
            save_config(config_path, InspectionConfig(roi=(30, 20, 280, 180),
                measurement=MeasurementOptions(kind='circle')))
            effective_path = folder / 'effective.json'
            args = ['--image', str(image_path), '--already-undistorted', '--allow-unverified-geometry',
                    '--homography-mode', 'reuse', '--homography-file', str(profile_path),
                    '--config', str(config_path), '--measurement', 'rectangle',
                    '--standard-uncertainty-mm', '0.03', '--uncertainty-validated',
                    '--tolerance', 'length', '80', '-0.3', '0.3',
                    '--tolerance', 'width', '40', '-0.3', '0.3',
                    '--output', str(folder / 'output'), '--save-config', str(effective_path)]
            original_profile = profile_path.read_bytes()
            with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                result = main(args)
            self.assertEqual(result.qc_status, 'PASS')
            self.assertIn('length:', stdout.getvalue())
            self.assertEqual(load_config(effective_path).measurement.kind, 'rectangle')
            self.assertEqual(profile_path.read_bytes(), original_profile)
            self.assertTrue((folder / 'output' / 'measurements.json').is_file())

    def test_headless_service_dimensions_qc_and_no_mutation(self):
        image = rectangle_image()
        before = image.copy()
        H = np.diag([0.4, 0.4, 1.])
        config = InspectionConfig(roi=(30, 20, 280, 180),
                                  uncertainty=UncertaintyBudget(standard_mm=0.03, validated=True),
                                  tolerances={'length': ToleranceSpec(80, -0.3, 0.3),
                                              'width': ToleranceSpec(40, -0.3, 0.3)})
        events = []
        with patch('cv2.imshow', side_effect=AssertionError('No GUI')), \
                patch('cv2.selectROI', side_effect=AssertionError('No GUI')), \
                patch('cv2.imwrite', side_effect=AssertionError('No automatic disk writes')):
            result = inspect_image(image, H, config, on_progress=lambda stage, fraction: events.append((stage, fraction)))
        self.assertEqual(result.qc_status, 'PASS')
        self.assertAlmostEqual(result.measurement.dimensions['length'].value_mm, 80, delta=0.15)
        self.assertAlmostEqual(result.measurement.dimensions['width'].value_mm, 40, delta=0.15)
        np.testing.assert_array_equal(before, image)
        self.assertEqual(events[-1], ('complete', 1.0))
        rendered = render_inspection(image, result)
        self.assertEqual(rendered.shape[1], image.shape[1] + 490)
        np.testing.assert_array_equal(before, image)
        config.uncertainty = UncertaintyBudget()
        self.assertEqual(inspect_image(image, H, config).qc_status, 'NOT_VALIDATED')
        config.uncertainty = UncertaintyBudget(standard_mm=0.03, validated=True)
        config.tolerances.pop('width')
        self.assertEqual(inspect_image(image, H, config).qc_status, 'PARTIAL')

    def test_cancellation_and_bad_measurement_scope(self):
        image = rectangle_image()
        with self.assertRaises(InspectionCancelled):
            inspect_image(image, np.eye(3), is_cancelled=lambda: True)
        counter = []
        def progress(stage, fraction):
            counter.append(stage)
        with self.assertRaises(InspectionCancelled):
            inspect_image(image, np.eye(3), on_progress=progress, is_cancelled=lambda: len(counter) >= 1)
        with self.assertRaisesRegex(ValueError, 'homography'):
            inspect_image(image, None)
        image[20:40, 20:40] = 0
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            inspect_image(image, np.eye(3), InspectionConfig(segmentation={'keep_largest': False}))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            inspect_image(rectangle_image(), np.eye(3), InspectionConfig(
                roi=(120, 20, 180, 180), segmentation={'reject_border': False}))

    def test_circle_under_perspective_and_hole_exclusion(self):
        image = np.full((300, 400, 3), 240, np.uint8)
        cv2.circle(image, (200, 150), 70, (10, 10, 10), -1)
        cv2.circle(image, (200, 150), 25, (240, 240, 240), -1)
        warp = np.array([[0.85, 0.12, 10.], [-0.05, 0.9, 20.], [0.0004, -0.0002, 1.]])
        warped = cv2.warpPerspective(image, warp, (400, 300), borderValue=(240, 240, 240))
        warped = cv2.GaussianBlur(warped, (5, 5), 0.7)
        H = np.diag([0.2, 0.2, 1.]) @ np.linalg.inv(warp)
        config = InspectionConfig(measurement=MeasurementOptions(kind='circle'))
        result = inspect_image(warped, H, config)
        self.assertAlmostEqual(result.measurement.dimensions['diameter'].value_mm, 28, delta=0.3)
        self.assertLess(len(result.measurement_point_indices), len(result.processing.edges.points))

    def test_rectangle_perspective_and_anisotropic_pixels(self):
        image = rectangle_image()
        warp = np.array([[1.3, 0.08, 5.], [0.03, 0.65, 20.], [0.0002, -0.0001, 1.]])
        warped = cv2.warpPerspective(image, warp, (480, 220), borderValue=(240, 240, 240))
        H = np.diag([0.4, 0.4, 1.]) @ np.linalg.inv(warp)
        for kind in ('rectangle', 'bounding-box'):
            result = inspect_image(warped, H, InspectionConfig(measurement=MeasurementOptions(kind=kind)))
            self.assertAlmostEqual(result.measurement.dimensions['length'].value_mm, 80, delta=0.3)
            self.assertAlmostEqual(result.measurement.dimensions['width'].value_mm, 40, delta=0.3)

    def test_config_and_exports_are_serializable_and_no_stale_measurement(self):
        config = InspectionConfig(roi=(30, 20, 280, 180))
        self.assertEqual(config_from_dict(config_to_dict(config)), config)
        for invalid in ({'version': 2}, {'version': 1, 'typo': 5},
                        {'version': 1, 'segmentation': {'typo': 5}},
                        {'version': 1, 'tolerances': {'diameter': dict(nominal_mm=20, lower_deviation_mm=-1, upper_deviation_mm=1)}}):
            with self.assertRaises(ValueError):
                config_from_dict(invalid)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / 'config.json'
            save_config(path, config)
            self.assertEqual(load_config(path), config)
            image = rectangle_image()
            result = inspect_image(image, np.diag([0.4, 0.4, 1.]), config)
            json.dumps(inspection_to_dict(result), allow_nan=False)
            save_inspection(folder, image, result, config=config)
            data = json.loads((folder / 'measurements.json').read_text())
            self.assertEqual(set(data['measurement']['dimensions']), {'length', 'width'})
            self.assertEqual(len((folder / 'measurements.csv').read_text().splitlines()), 3)
            self.assertTrue((folder / 'measurement_overlay.png').is_file())
            config.measurement = MeasurementOptions(kind='none')
            result = inspect_image(image, np.eye(3), config)
            save_inspection(folder, image, result, config=config)
            self.assertIsNone(json.loads((folder / 'measurements.json').read_text())['measurement'])
            self.assertEqual(len((folder / 'measurements.csv').read_text().splitlines()), 1)

    def test_raw_preparation_uses_existing_geometry_without_mutation(self):
        image = rectangle_image()
        calibration = {'mtx': np.array([[400., 0, 170], [0, 400., 110], [0, 0, 1.]]),
                       'dist': np.zeros(5), 'image_size': (340, 220)}
        prepared = prepare_image(image, calibration=calibration)
        self.assertEqual(prepared.preprocessing['source_size'], [340, 220])
        with self.assertRaises(ValueError):
            prepare_image(image, calibration=calibration, calibration_size=(10, 20))
        np.testing.assert_array_equal(prepare_image(image).image, image)


if __name__ == '__main__':
    unittest.main()
