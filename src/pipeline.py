"""Command-line adapter for the inspection service.

Run from the repository root: python -m src.pipeline --help.
All ROI coordinates belong to the already-undistorted/cropped image.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.calibration.undistort import load_calibration_data
from src.configuration import load_config, save_config
from src.inspection import InspectionConfig, inspect_image, prepare_image
from src.measurement.homography import compute_homography  # Backward-compatible import.
from src.measurement.homography_profile import (
    add_homography_arguments, create_homography_profile, homography_for_image,
    load_homography_profile, resolve_homography_mode, save_homography_profile,
)
from src.processing import PipelineResult, process_image  # Backward-compatible public API.
from src.qc.tolerance_check import ToleranceSpec
from src.reporting import save_inspection, save_result  # Backward-compatible point export.


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument('--calib', help='Calibration PKL/NPZ for a RAW input image')
    calibration_group.add_argument('--already-undistorted', action='store_true')
    parser.add_argument('--calibration-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--config', help='Versioned JSON inspection settings; explicit CLI flags override it')
    parser.add_argument('--save-config', help='Save effective inspection settings as JSON after a successful run')
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    roi_group.add_argument('--select-roi', action='store_true')
    parser.add_argument('--method', choices=['otsu', 'adaptive', 'hsv'])
    parser.add_argument('--foreground', choices=['dark', 'bright'])
    parser.add_argument('--hsv-lower', type=int, nargs=3)
    parser.add_argument('--hsv-upper', type=int, nargs=3)
    parser.add_argument('--min-area', type=float)
    parser.add_argument('--blur-size', type=int)
    parser.add_argument('--morphology-size', type=int)
    parser.add_argument('--adaptive-block-size', type=int)
    parser.add_argument('--adaptive-c', type=float)
    parser.add_argument('--all-components', action='store_true', default=None)
    parser.add_argument('--allow-border', action='store_true', default=None)
    parser.add_argument('--fill-holes', action='store_true', default=None)
    parser.add_argument('--canny-low', type=float)
    parser.add_argument('--canny-high', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--boundary-radius', type=int)
    parser.add_argument('--min-gradient', type=float)
    parser.add_argument('--max-offset', type=float)
    parser.add_argument('--measurement', choices=['rectangle', 'circle', 'bounding-box', 'none'],
                        help='Default rectangle: fitted length/width. Circle: diameter. Bounding-box: overall extents. None: edges only.')
    parser.add_argument('--max-fit-rmse-mm', type=float)
    parser.add_argument('--min-inlier-fraction', type=float)
    parser.add_argument('--standard-uncertainty-mm', type=float,
                        help='Additional characterized standard uncertainty in mm (not board RMSE)')
    parser.add_argument('--relative-standard-uncertainty', type=float,
                        help='Relative standard scale uncertainty, e.g. 0.001 means 0.1 percent')
    parser.add_argument('--coverage-factor', type=float)
    parser.add_argument('--uncertainty-validated', action=argparse.BooleanOptionalAction, default=None,
                        help='User assertion: budget characterized on physical standards; required for QC decisions')
    parser.add_argument('--tolerance', nargs=4, action='append',
                        metavar=('DIMENSION', 'NOMINAL', 'LOWER_DEV', 'UPPER_DEV'),
                        help='Repeat per dimension, e.g. --tolerance length 80 -0.2 0.2')
    add_homography_arguments(parser)
    parser.add_argument('--output', default='outputs/edge_pipeline')
    return parser


def _configuration(args):
    config = load_config(args.config) if args.config else InspectionConfig()
    if args.roi is not None:
        config.roi = tuple(args.roi)
    for key in ('method', 'foreground', 'hsv_lower', 'hsv_upper', 'min_area', 'blur_size',
                'morphology_size', 'adaptive_block_size', 'adaptive_c', 'fill_holes'):
        value = getattr(args, key)
        if value is not None:
            config.segmentation[key] = value
    if args.all_components is not None:
        config.segmentation['keep_largest'] = not args.all_components
    if args.allow_border is not None:
        config.segmentation['reject_border'] = not args.allow_border
    for key in ('canny_low', 'canny_high', 'sigma', 'boundary_radius', 'min_gradient', 'max_offset'):
        value = getattr(args, key)
        if value is not None:
            config.edges[key] = value
    updates = {key: getattr(args, key) for key in ('max_fit_rmse_mm', 'min_inlier_fraction')
               if getattr(args, key) is not None}
    if args.measurement is not None:
        updates['kind'] = args.measurement
    config.measurement = replace(config.measurement, **updates)
    updates = {}
    for argument, field in (('standard_uncertainty_mm', 'standard_mm'),
                            ('relative_standard_uncertainty', 'relative_standard'),
                            ('coverage_factor', 'coverage_factor'), ('uncertainty_validated', 'validated')):
        if getattr(args, argument) is not None:
            updates[field] = getattr(args, argument)
    config.uncertainty = replace(config.uncertainty, **updates)
    for name, nominal, lower, upper in args.tolerance or []:
        config.tolerances[name] = ToleranceSpec(float(nominal), float(lower), float(upper))
    config.validate()
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = resolve_homography_mode(parser, args)
    try:
        config = _configuration(args)
        if config.roi is None and not args.select_roi:
            raise ValueError('Supply --roi, --select-roi, or an ROI in --config')
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f'Cannot read image: {args.image}')
        calibration = load_calibration_data(args.calib) if args.calib else None
        prepared = prepare_image(image, calibration=calibration, calibration_size=args.calibration_size)
        image = prepared.image
        metadata = {'source_image': str(Path(args.image).resolve()), 'arguments': vars(args)}
        if prepared.preprocessing is not None:
            metadata.update(prepared.preprocessing)
        if mode == 'compute':
            profile = create_homography_profile(image, args.square_mm, (args.cols, args.rows),
                                                preprocessing=prepared.preprocessing)
            save_homography_profile(args.homography_file, profile)
            H = np.asarray(profile['H'], dtype=np.float64)
            print(f'Saved homography: {args.homography_file}')
        else:
            profile = load_homography_profile(args.homography_file)
            H = homography_for_image(profile, image, preprocessing=prepared.preprocessing)
            print(f'Loaded homography: {args.homography_file} (no board detection)')
        metadata.update(homography_mode=mode, homography_file=str(Path(args.homography_file).resolve()),
                        homography_reference=profile)
        if args.select_roi:
            try:
                roi = cv2.selectROI('Select object on undistorted image', image, showCrosshair=True)
            finally:
                cv2.destroyAllWindows()
            if roi[2] == 0 or roi[3] == 0:
                raise ValueError('ROI selection cancelled')
            config.roi = tuple(map(int, roi))
        result = inspect_image(image, H, config)
        if mode == 'compute':
            result.processing.homography_rmse_mm = profile['fit_rmse_mm']
        save_inspection(args.output, image, result, config=config, metadata=metadata)
        if args.save_config:
            save_config(args.save_config, config)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        parser.exit(1, f'Pipeline error: {exc}\n')
    print(f'Saved {len(result.processing.edges.points)} subpixel points to {args.output}')
    if result.measurement is not None:
        print(f'Measurement model: {result.measurement.kind}')
        if result.measurement.fit_rmse_mm is not None:
            print(f'Geometric fit RMSE: {result.measurement.fit_rmse_mm:.4f} mm')
        for name, dimension in result.measurement.dimensions.items():
            u = result.uncertainties[name]
            label = 'validated budget' if u.validated else 'PROVISIONAL estimate'
            if result.measurement.kind == 'bounding-box' and not u.validated and u.expanded_mm == 0:
                print(f'{name}: {dimension.value_mm:.3f} mm | U unknown (external budget needed)')
            else:
                print(f'{name}: {dimension.value_mm:.3f} mm | U={u.expanded_mm:.4f} mm '
                      f'(k={u.coverage_factor:g}, {label})')
            if name in result.tolerance_results:
                t = result.tolerance_results[name]
                print(f'  nominal={t.nominal_mm:g} mm; limits=[{t.lower_limit_mm:g}, {t.upper_limit_mm:g}] mm; '
                      f'deviation={t.deviation_mm:+.4f} mm; {t.status}')
        print(f'QC: {result.qc_status}')
    for warning in result.warnings:
        print(f'Note: {warning}')
    print(f'Reference chessboard fit RMSE: {profile["fit_rmse_mm"]:.4f} mm '
          '(not object measurement uncertainty)')
    return result


if __name__ == '__main__':
    main()
