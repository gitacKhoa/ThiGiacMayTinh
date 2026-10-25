"""GUI-independent image -> segmentation -> subpixel points -> plane mm."""

from dataclasses import dataclass

import numpy as np

from src.edge_detection.subpixel_edge import SubpixelEdgeResult, detect_subpixel_edges
from src.measurement.homography import compute_homography, points_to_mm, chessboard_real_world_points
from src.segmentation.segment import SegmentationResult, segment_image


@dataclass
class PipelineResult:
    segmentation: SegmentationResult
    edges: SubpixelEdgeResult
    homography: object
    points_mm: object
    homography_rmse_mm: object


def process_image(image, *, roi=None, segmentation_options=None, edge_options=None,
                  square_mm=None, pattern_size=(8, 5), homography=None):
    """Process an already-undistorted image, preserving full-image coordinates."""
    if homography is not None and square_mm is not None:
        raise ValueError('Choose either a supplied homography or square_mm, not both')
    H = homography
    if H is not None:
        points_to_mm(np.empty((0, 2)), H)
    rmse = None
    if square_mm is not None:
        H, corners = compute_homography(image, square_mm, pattern_size)
        if H is None:
            raise ValueError(f'Chessboard {pattern_size} not found; cannot output millimetres')
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
