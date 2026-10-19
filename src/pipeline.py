"""Run undistortion -> segmentation -> subpixel edges -> optional plane mm.

Run from the repository root: python -m src.pipeline --help
ROI coordinates refer to the undistorted, cropped image, not the raw capture.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibration.undistort import load_calibration_data, undistort_with_metadata
from src.edge_detection.subpixel_edge import SubpixelEdgeResult, detect_subpixel_edges, draw_subpixel_edges
from src.measurement.homography import compute_homography, points_to_mm
from src.segmentation.segment import SegmentationResult, segment_image


@dataclass
class PipelineResult:
    segmentation: SegmentationResult
    edges: SubpixelEdgeResult
    homography: object
    points_mm: object
    homography_rmse_mm: object


def process_image(image, *, roi=None, segmentation_options=None, edge_options=None,
                  square_mm=None, pattern_size=(8, 5)):
    """Process an already-undistorted image without changing its geometry.

    Homography is computed from this same image; a requested but undetected
    board is an error, not an implicit fallback from millimetres to pixels.
    No object dimensions or QC decisions are inferred from an unordered cloud.
    """
    H = None
    corners = None
    rmse = None
    if square_mm is not None:
        H, corners = compute_homography(image, square_mm, pattern_size)
        if H is None:
            raise ValueError(f'Chessboard {pattern_size} not found; cannot output millimetres')
        from src.measurement.homography import chessboard_real_world_points
        residuals = points_to_mm(corners.reshape(-1, 2), H) - chessboard_real_world_points(pattern_size, square_mm)
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    segmentation = segment_image(image, roi=roi, **(segmentation_options or {}))
    if segmentation.component_count == 0:
        raise ValueError('No object found. Check ROI, polarity/HSV bounds and min_area; '
                         'leave background around the entire object inside the ROI.')
    edges = detect_subpixel_edges(image, segmentation.mask, roi=segmentation.roi, **(edge_options or {}))
    if len(edges.points) == 0:
        raise ValueError('No valid subpixel edges. Check image contrast, mask and Canny thresholds.')
    world = points_to_mm(edges.points, H) if H is not None else None
    return PipelineResult(segmentation, edges, H, world, rmse)


def save_result(output, image, result, metadata=None):
    """Save diagnostic images and lossless numeric data with coordinate metadata."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previews = {
        'undistorted.png': image,
        'mask.png': result.segmentation.mask,
        'edges.png': result.edges.edge_map,
        'overlay.png': draw_subpixel_edges(image, result.edges.points),
    }
    for name, preview in previews.items():
        if not cv2.imwrite(str(output / name), preview):
            raise OSError(f'Cannot write {output / name}')
    world = result.points_mm if result.points_mm is not None else np.empty((0, 2), dtype=np.float64)
    H = result.homography if result.homography is not None else np.empty((0, 0), dtype=np.float64)
    np.savez_compressed(output / 'edges.npz', points_px=result.edges.points,
                        pixel_points=result.edges.pixel_points, normals=result.edges.normals,
                        offsets=result.edges.offsets, strengths=result.edges.strengths,
                        points_mm=world, homography=H)
    columns = [result.edges.points, result.edges.offsets[:, None], result.edges.strengths[:, None]]
    header = 'x_px,y_px,normal_offset_px,gradient_strength'
    if result.points_mm is not None:
        columns.append(result.points_mm)
        header += ',x_mm,y_mm'
    np.savetxt(output / 'edges.csv', np.column_stack(columns), delimiter=',',
               header=header, comments='', fmt='%.10f')
    summary = dict(metadata or {})
    summary.update(
        coordinate_frame='undistorted.png; x=column, y=row; no ROI offset needed',
        image_size=[image.shape[1], image.shape[0]], roi=list(result.segmentation.roi),
        component_count=result.segmentation.component_count,
        threshold=result.segmentation.threshold,
        candidate_count=result.edges.candidate_count, point_count=len(result.edges.points),
        rejected_count=result.edges.candidate_count - len(result.edges.points),
        has_mm=result.points_mm is not None, homography_rmse_mm=result.homography_rmse_mm,
        point_order='unordered; do not use as a contour for area or perimeter',
        measurement_status='edge coordinates only; dimensions, uncertainty and QC not computed',
    )
    (output / 'metadata.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument('--calib', help='Calibration PKL/NPZ for the RAW input image')
    calibration_group.add_argument('--already-undistorted', action='store_true', help='Skip undistortion explicitly')
    parser.add_argument('--calibration-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Original calibration resolution for legacy files without image_size')
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    roi_group.add_argument('--select-roi', action='store_true', help='Select object ROI on undistorted image')
    parser.add_argument('--method', choices=['otsu', 'adaptive', 'hsv'], default='otsu')
    parser.add_argument('--foreground', choices=['dark', 'bright'], default='dark')
    parser.add_argument('--hsv-lower', type=int, nargs=3)
    parser.add_argument('--hsv-upper', type=int, nargs=3)
    parser.add_argument('--min-area', type=float, default=100)
    parser.add_argument('--blur-size', type=int, default=5)
    parser.add_argument('--morphology-size', type=int, default=3)
    parser.add_argument('--adaptive-block-size', type=int, default=31)
    parser.add_argument('--adaptive-c', type=float, default=5)
    parser.add_argument('--all-components', action='store_true')
    parser.add_argument('--allow-border', action='store_true')
    parser.add_argument('--fill-holes', action='store_true')
    parser.add_argument('--canny-low', type=float, default=50)
    parser.add_argument('--canny-high', type=float, default=150)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--boundary-radius', type=int, default=3)
    parser.add_argument('--min-gradient', type=float, default=5)
    parser.add_argument('--max-offset', type=float, default=0.75)
    parser.add_argument('--square-mm', type=float, help='Enable mm output using board in the SAME image')
    parser.add_argument('--cols', type=int, default=8)
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--output', default='outputs/edge_pipeline')
    args = parser.parse_args(argv)
    try:
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f'Cannot read image: {args.image}')
        metadata = {'source_image': str(Path(args.image).resolve()), 'arguments': vars(args)}
        if args.calib:
            calibration = load_calibration_data(args.calib)
            size = calibration.get('image_size', args.calibration_size)
            if size is None:
                raise ValueError('Legacy calibration has no image_size; supply --calibration-size WIDTH HEIGHT')
            if args.calibration_size is not None and tuple(size) != tuple(args.calibration_size):
                raise ValueError('--calibration-size conflicts with calibration metadata')
            geometry = undistort_with_metadata(image, calibration['mtx'], calibration['dist'], calibration_size=size)
            image = geometry.image
            metadata.update(source_size=list(geometry.source_size), crop_roi=list(geometry.crop_roi),
                            alpha=geometry.alpha, camera_matrix=geometry.camera_matrix.tolist())
        roi = args.roi
        if args.select_roi:
            try:
                roi = cv2.selectROI('Select object on undistorted image', image, showCrosshair=True)
            finally:
                cv2.destroyAllWindows()
            if roi[2] == 0 or roi[3] == 0:
                raise ValueError('ROI selection cancelled')
        segmentation_options = dict(
            method=args.method, foreground=args.foreground, hsv_lower=args.hsv_lower, hsv_upper=args.hsv_upper,
            min_area=args.min_area, blur_size=args.blur_size, morphology_size=args.morphology_size,
            keep_largest=not args.all_components, reject_border=not args.allow_border,
            fill_holes=args.fill_holes, adaptive_block_size=args.adaptive_block_size, adaptive_c=args.adaptive_c)
        edge_options = dict(canny_low=args.canny_low, canny_high=args.canny_high, sigma=args.sigma,
                            boundary_radius=args.boundary_radius, min_gradient=args.min_gradient, max_offset=args.max_offset)
        result = process_image(image, roi=roi, segmentation_options=segmentation_options,
                               edge_options=edge_options, square_mm=args.square_mm, pattern_size=(args.cols, args.rows))
        save_result(args.output, image, result, metadata)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        parser.exit(1, f'Pipeline error: {exc}\n')
    print(f'Saved {len(result.edges.points)} subpixel points to {args.output}')
    if result.points_mm is not None:
        print(f'Chessboard fit RMSE: {result.homography_rmse_mm:.4f} mm (not object measurement uncertainty)')


if __name__ == '__main__':
    main()
