"""Public service API for CLI and future desktop/web UI.

This module does not open windows, read/write files, print, or access the camera.
Call from a UI worker with a copied image; callbacks must be thread-safe. The
caller owns calibration/profile persistence and rendering/export decisions.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import cv2
import numpy as np

from src.calibration.undistort import undistort_with_metadata
from src.measurement.homography_profile import undistortion_geometry
from src.measurement.measure import MeasurementOptions, MeasurementResult, measure_points
from src.measurement.uncertainty import UncertaintyBudget, estimate_uncertainty
from src.processing import PipelineResult, process_image
from src.qc.tolerance_check import ToleranceSpec, check_tolerance
from src.segmentation.segment import as_gray


class InspectionCancelled(Exception):
    """Cooperative cancellation requested by the UI/caller."""


@dataclass
class PreparedImage:
    image: np.ndarray
    preprocessing: Optional[dict] = None


@dataclass
class InspectionConfig:
    roi: Optional[tuple] = None
    segmentation: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    measurement: MeasurementOptions = field(default_factory=MeasurementOptions)
    uncertainty: UncertaintyBudget = field(default_factory=UncertaintyBudget)
    tolerances: dict = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.measurement, MeasurementOptions) or not isinstance(self.uncertainty, UncertaintyBudget):
            raise ValueError('Invalid measurement or uncertainty configuration')
        if not isinstance(self.segmentation, dict) or not isinstance(self.edges, dict) or not isinstance(self.tolerances, dict):
            raise ValueError('Segmentation, edges and tolerances must be dictionaries')
        if 'roi' in self.segmentation or 'roi' in self.edges:
            raise ValueError('Set roi on InspectionConfig, not inside segmentation/edges')
        segmentation_keys = {'method', 'foreground', 'blur_size', 'morphology_size', 'min_area',
                             'keep_largest', 'reject_border', 'fill_holes', 'adaptive_block_size',
                             'adaptive_c', 'hsv_lower', 'hsv_upper'}
        edge_keys = {'canny_low', 'canny_high', 'sigma', 'boundary_radius', 'min_gradient', 'max_offset'}
        if set(self.segmentation) - segmentation_keys or set(self.edges) - edge_keys:
            raise ValueError('Unknown segmentation or edge configuration fields')
        if self.roi is not None and (len(self.roi) != 4 or any(
                not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in self.roi)):
            raise ValueError('roi must contain four integers')
        expected = {'rectangle': {'length', 'width'}, 'bounding-box': {'length', 'width'},
                    'circle': {'diameter'}, 'none': set()}[self.measurement.kind]
        if set(self.tolerances) - expected:
            raise ValueError(f'Tolerances for {self.measurement.kind} must use dimensions: {sorted(expected)}')
        if any(not isinstance(spec, ToleranceSpec) for spec in self.tolerances.values()):
            raise ValueError('Each tolerance must be a ToleranceSpec')


@dataclass
class InspectionResult:
    processing: PipelineResult
    measurement: Optional[MeasurementResult]
    uncertainties: dict
    tolerance_results: dict
    qc_status: str
    warnings: list
    measurement_point_indices: np.ndarray


def prepare_image(image, *, calibration=None, calibration_size=None):
    """calibration=None explicitly means input is already undistorted.

    For raw input, pass a loaded calibration dictionary (mtx/dist/image_size).
    Both the CLI and future UI use this function before ROI selection.
    """
    as_gray(image)
    if calibration is None:
        return PreparedImage(image.copy())
    size = calibration.get('image_size', calibration_size)
    if size is None:
        raise ValueError('Legacy calibration has no image_size; supply calibration_size')
    if calibration_size is not None and tuple(size) != tuple(calibration_size):
        raise ValueError('calibration_size conflicts with calibration metadata')
    result = undistort_with_metadata(image, calibration['mtx'], calibration['dist'], calibration_size=size)
    return PreparedImage(result.image, undistortion_geometry(result, calibration['dist']))


def _outer_edge_indices(processed, radius):
    # Holes/text inside the object must not enter an outside diameter/rectangle fit.
    contours, _ = cv2.findContours(processed.segmentation.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    exterior = np.zeros_like(processed.segmentation.mask)
    cv2.drawContours(exterior, contours, -1, 255, 1)
    band = cv2.dilate(exterior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    pixels = processed.edges.pixel_points.astype(int)
    return np.flatnonzero(band[pixels[:, 1], pixels[:, 0]] > 0)


def inspect_image(image, homography, config=None, *,
                  on_progress: Optional[Callable[[str, float], None]] = None,
                  is_cancelled: Optional[Callable[[], bool]] = None):
    """Inspect one object in an already-undistorted image with a compatible H.

    Returns structured results; raises ValueError for invalid/incomplete geometry.
    Cancellation is checked between stages (not inside OpenCV native calls).
    H must be validated against the prepared image by homography_for_image first.
    """
    config = config or InspectionConfig()
    config.validate()
    def checkpoint(stage, fraction):
        if is_cancelled is not None and is_cancelled():
            raise InspectionCancelled('Inspection cancelled')
        if on_progress is not None:
            on_progress(stage, fraction)

    checkpoint('processing', 0.0)
    if config.measurement.kind != 'none' and homography is None:
        raise ValueError('A pixel-to-mm homography is required for dimensional measurement')
    processed = process_image(image, roi=config.roi, segmentation_options=config.segmentation,
                              edge_options=config.edges, homography=homography)
    checkpoint('measurement', 0.55)
    warnings = []
    if config.measurement.kind == 'none':
        checkpoint('complete', 1.0)
        return InspectionResult(processed, None, {}, {}, 'NOT_CONFIGURED', warnings, np.empty(0, dtype=int))
    if processed.segmentation.component_count != 1:
        raise ValueError('Dimensional measurement requires exactly one object per ROI')
    x, y, w, h = processed.segmentation.roi
    mask = processed.segmentation.mask[y:y+h, x:x+w]
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Cannot measure a truncated object touching the ROI border')
    # A projective pole through the object invalidates the planar shape fit.
    points = processed.edges.points
    H = np.asarray(homography, dtype=float)
    H = H / np.max(np.abs(H))
    denominator = np.column_stack((points, np.ones(len(points)))) @ H[2]
    if denominator.min() <= 0 <= denominator.max():
        raise ValueError('Homography crosses infinity within the object')
    indices = _outer_edge_indices(processed, int(config.edges.get('boundary_radius', 3)))
    measurement = measure_points(processed.points_mm[indices], config.measurement)
    if measurement.kind == 'bounding-box':
        warnings.append('Bounding-box measures overall extents, not fitted datum edges. No fit uncertainty is '
                        'available: a zero fit component does NOT imply zero measurement uncertainty.')
    checkpoint('uncertainty', 0.8)
    uncertainties = {name: estimate_uncertainty(dimension.value_mm, dimension.fit_standard_uncertainty_mm,
                                               config.uncertainty)
                     for name, dimension in measurement.dimensions.items()}
    if measurement.kind == 'bounding-box':
        uncertainties = {name: replace(u, available=u.standard_mm > 0,
                                        note='No bounding-box fit covariance is available. ' + u.note)
                         for name, u in uncertainties.items()}
    if not config.uncertainty.validated:
        warnings.append('Uncertainty is provisional: fit residuals assume independent points and exact H; '
                        'characterize calibration, scale, repeatability and bias on physical standards.')
    warnings.append('All dimensions refer to the calibrated plane; keep camera/focus/plane and preprocessing unchanged.')
    checkpoint('tolerance', 0.9)
    tolerance_results = {name: check_tolerance(name, measurement.dimensions[name].value_mm, spec,
                                              uncertainties[name]) for name, spec in config.tolerances.items()}
    states = {r.status for r in tolerance_results.values()}
    missing = set(measurement.dimensions) - set(config.tolerances)
    if not states:
        qc_status = 'NOT_CONFIGURED'
    elif 'FAIL' in states:
        qc_status = 'FAIL'
    elif 'NOT_VALIDATED' in states:
        qc_status = 'NOT_VALIDATED'
    elif 'INDETERMINATE' in states:
        qc_status = 'INDETERMINATE'
    elif missing:
        qc_status = 'PARTIAL'
    else:
        qc_status = 'PASS'
    if missing:
        warnings.append('No tolerance configured for: ' + ', '.join(sorted(missing)))
    checkpoint('complete', 1.0)
    return InspectionResult(processed, measurement, uncertainties, tolerance_results, qc_status, warnings, indices)
