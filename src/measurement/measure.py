"""Fit explicit geometric models to unordered subpixel points in plane millimetres.

No file access, CLI, GUI, camera or global state. Rectangle dimensions are distances
between four fitted orthogonal lines, not an axis-aligned bounding box. Circle
diameter is a geometric least-squares fit, not a bounding-box approximation.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class MeasurementOptions:
    kind: str = 'rectangle'
    max_fit_rmse_mm: float = 0.25
    min_inlier_fraction: float = 0.8
    min_points: int = 24
    min_side_coverage: float = 0.6
    max_circle_gap_deg: float = 90.0

    def __post_init__(self):
        if self.kind not in ('rectangle', 'circle', 'bounding-box', 'none'):
            raise ValueError('measurement kind must be rectangle, circle, bounding-box or none')
        if not np.isfinite([self.max_fit_rmse_mm, self.min_inlier_fraction,
                            self.min_side_coverage, self.max_circle_gap_deg]).all():
            raise ValueError('Measurement settings must be finite')
        if self.max_fit_rmse_mm <= 0 or not 0.5 <= self.min_inlier_fraction <= 1:
            raise ValueError('Positive max_fit_rmse_mm and min_inlier_fraction in [0.5, 1] required')
        if not isinstance(self.min_points, int) or isinstance(self.min_points, bool) or self.min_points < 12:
            raise ValueError('min_points must be an integer >= 12')
        if not 0 < self.min_side_coverage <= 1 or not 0 < self.max_circle_gap_deg <= 120:
            raise ValueError('Require side coverage in (0, 1] and circle gap in (0, 120] degrees')


@dataclass
class Dimension:
    name: str
    value_mm: float
    fit_standard_uncertainty_mm: float
    endpoints_mm: list


@dataclass
class MeasurementResult:
    kind: str
    dimensions: dict
    outline_mm: np.ndarray
    center_mm: np.ndarray
    fit_rmse_mm: float | None
    inliers: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _points(points, options):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Measurement requires finite (N, 2) millimetre points')
    if len(np.unique(points, axis=0)) < options.min_points:
        raise ValueError(f'Not enough distinct edge points for measurement (need {options.min_points})')
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError('Degenerate edge points: cannot measure a line as a 2D object')
    return points


def _inliers(residuals, options):
    # Robust scale only chooses a subset; final RMS and coverage are checked too.
    scale = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    cutoff = min(3 * options.max_fit_rmse_mm, max(3 * scale, 0.05 * options.max_fit_rmse_mm, 1e-8))
    return np.abs(residuals) <= cutoff


def _quality(residuals, inliers, options):
    if inliers.sum() < options.min_points or inliers.mean() < options.min_inlier_fraction:
        raise ValueError('Too many inconsistent edges; check ROI, segmentation or measurement shape')
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    if rmse > options.max_fit_rmse_mm:
        raise ValueError(f'Geometric fit RMSE {rmse:.4f} mm exceeds {options.max_fit_rmse_mm:.4f} mm')
    return rmse


def _covariance(jacobian, residuals):
    n, parameters = jacobian.shape
    if n <= parameters or np.linalg.matrix_rank(jacobian) != parameters:
        raise ValueError('Insufficient geometry to estimate fit parameters')
    # Conditional iid residual model, NOT full metrological uncertainty.
    variance = float(residuals @ residuals / (n - parameters))
    return np.linalg.pinv(jacobian.T @ jacobian) * variance


def _rectangle_state(p, parameters):
    angle = parameters[0]
    axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = p @ axes
    distances = np.column_stack((q[:, 0] - parameters[1], q[:, 0] - parameters[2],
                                 q[:, 1] - parameters[3], q[:, 1] - parameters[4]))
    side = np.argmin(np.abs(distances), axis=1)
    rows = np.arange(len(p))
    residuals = distances[rows, side]
    jac = np.zeros((len(p), 5))
    jac[:, 0] = np.where(side < 2, q[:, 1], -q[:, 0])
    jac[rows, side + 1] = -1
    return q, axes, side, residuals, jac


def _rectangle(points, options):
    origin = np.median(points, axis=0)
    p = points - origin
    # Orientation search is robust to a few distant outliers (unlike minAreaRect).
    best = None
    count = max(options.min_points, int(len(p) * options.min_inlier_fraction))
    for angle in np.linspace(0, np.pi / 2, 180, endpoint=False):
        axes = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        q = p @ axes
        lo, hi = np.quantile(q, [0.03, 0.97], axis=0)
        candidate = np.array([angle, lo[0], hi[0], lo[1], hi[1]])
        residuals = _rectangle_state(p, candidate)[3]
        score = np.mean(np.sort(residuals ** 2)[:count])
        if best is None or score < best[0]:
            best = score, candidate
    parameters = best[1]
    for _ in range(30):
        q, axes, side, residuals, jac = _rectangle_state(p, parameters)
        keep = _inliers(residuals, options)
        if keep.sum() < options.min_points or len(np.unique(side[keep])) != 4:
            raise ValueError('Rectangle requires visible, consistent points on all four sides')
        step = np.linalg.lstsq(jac[keep], -residuals[keep], rcond=None)[0]
        parameters += step
        if np.linalg.norm(step) < 1e-9:
            break
    q, axes, side, residuals, jac = _rectangle_state(p, parameters)
    keep = _inliers(residuals, options)
    rmse = _quality(residuals, keep, options)
    left, right, bottom, top = parameters[1:]
    spans = np.array([right - left, top - bottom])
    if np.any(spans <= 0):
        raise ValueError('Nonpositive rectangle dimensions')
    coverages = []
    counts = []
    for s in range(4):
        values = q[(side == s) & keep, 1 if s < 2 else 0]
        if len(values) < 6:
            raise ValueError('Not enough points on each rectangle side (need >= 6)')
        low, high = (bottom, top) if s < 2 else (left, right)
        occupied = np.unique(np.clip(((values - low) / (high - low) * 10).astype(int), 0, 9)).size / 10
        coverage = min(float(np.ptp(values) / (high - low)), occupied)
        if coverage < options.min_side_coverage:
            raise ValueError('Incomplete rectangle edge coverage; expand ROI or improve lighting')
        coverages.append(coverage)
        counts.append(int(len(values)))
    covariance = _covariance(jac[keep], residuals[keep])
    local_center = np.array([(left + right) / 2, (bottom + top) / 2])
    outline = np.array([[left, bottom], [right, bottom], [right, top], [left, top]]) @ axes.T + origin
    dimensions = {}
    for axis, name in ((int(np.argmax(spans)), 'length'), (int(np.argmin(spans)), 'width')):
        # Square tie: the two measurements still use different axis pairs.
        if name == 'width':
            axis = 1 - int(np.argmax(spans))
        gradient = np.zeros(5)
        gradient[1 + 2 * axis:3 + 2 * axis] = [-1, 1]
        uncertainty = float(np.sqrt(max(0., gradient @ covariance @ gradient)))
        endpoints = np.tile(local_center, (2, 1))
        endpoints[:, axis] += [-spans[axis] / 2, spans[axis] / 2]
        dimensions[name] = Dimension(name, float(spans[axis]), uncertainty,
                                     (endpoints @ axes.T + origin).tolist())
    return MeasurementResult('rectangle', dimensions, outline, local_center @ axes.T + origin,
                             rmse, keep, dict(side_point_counts=counts, side_coverage=coverages,
                                             angle_deg=float(np.degrees(parameters[0]) % 180),
                                             inlier_fraction=float(keep.mean())))


def _circle_state(p, parameters):
    delta = p - parameters[:2]
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-12):
        lengths = np.maximum(lengths, 1e-12)
    residuals = lengths - parameters[2]
    jacobian = np.column_stack((-delta / lengths[:, None], -np.ones(len(p))))
    return residuals, jacobian


def _circle(points, options):
    origin = np.median(points, axis=0)
    p = points - origin
    rng = np.random.default_rng(0)
    candidates = []
    # Algebraic initialization plus deterministic RANSAC protects against spurious edges.
    for selected in [np.arange(len(p))] + [rng.choice(len(p), 3, replace=False) for _ in range(128)]:
        q = p[selected]
        A = np.column_stack((2 * q, np.ones(len(q))))
        if np.linalg.matrix_rank(A) < 3:
            continue
        a = np.linalg.lstsq(A, np.sum(q * q, axis=1), rcond=None)[0]
        radius2 = a[2] + a[:2] @ a[:2]
        if radius2 <= 0:
            continue
        params = np.r_[a[:2], np.sqrt(radius2)]
        res = _circle_state(p, params)[0]
        candidates.append((float(np.quantile(np.abs(res), options.min_inlier_fraction)), params))
    if not candidates:
        raise ValueError('Cannot initialize circle from these edges')
    parameters = min(candidates, key=lambda v: v[0])[1]
    for _ in range(30):
        residuals, jac = _circle_state(p, parameters)
        keep = _inliers(residuals, options)
        if keep.sum() < options.min_points:
            raise ValueError('Not enough consistent circle edges')
        step = np.linalg.lstsq(jac[keep], -residuals[keep], rcond=None)[0]
        parameters += step
        if np.linalg.norm(step) < 1e-9:
            break
    residuals, jac = _circle_state(p, parameters)
    keep = _inliers(residuals, options)
    rmse = _quality(residuals, keep, options)
    radius = float(parameters[2])
    if radius <= 0:
        raise ValueError('Nonpositive circle radius')
    delta = p[keep] - parameters[:2]
    angles = np.sort(np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi))
    gap = float(np.degrees(np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()))
    if gap > options.max_circle_gap_deg:
        raise ValueError('Incomplete circle coverage; a short arc cannot reliably determine diameter')
    covariance = _covariance(jac[keep], residuals[keep])
    center = parameters[:2] + origin
    angles = np.linspace(0, 2 * np.pi, 181)
    outline = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    endpoints = [center + [-radius, 0], center + [radius, 0]]
    dimension = Dimension('diameter', 2 * radius, float(2 * np.sqrt(max(0., covariance[2, 2]))),
                          np.asarray(endpoints).tolist())
    return MeasurementResult('circle', {'diameter': dimension}, outline, center, rmse, keep,
                             dict(radius_mm=radius, max_gap_deg=gap, inlier_fraction=float(keep.mean())))


def _bounding_box(points):
    """Overall dimensions of a minimum-area enclosing rectangle, NOT fitted sides.

    Keep all exterior points, including teeth/protrusions. Extreme-point noise
    affects this estimator; uncertainty must come from external characterization.
    """
    origin = points.mean(axis=0)
    box = cv2.boxPoints(cv2.minAreaRect((points - origin).astype(np.float32))).astype(np.float64) + origin
    vectors = np.array([box[1] - box[0], box[2] - box[1]])
    spans = np.linalg.norm(vectors, axis=1)
    if np.any(spans <= 0):
        raise ValueError('Degenerate bounding box')
    center = box.mean(axis=0)
    long_axis = int(np.argmax(spans))
    dimensions = {}
    for axis, name in ((long_axis, 'length'), (1 - long_axis, 'width')):
        endpoints = np.array([center - vectors[axis] / 2, center + vectors[axis] / 2])
        dimensions[name] = Dimension(name, float(spans[axis]), 0., endpoints.tolist())
    return MeasurementResult('bounding-box', dimensions, box, center, None,
                             np.ones(len(points), dtype=bool),
                             dict(definition='minimum-area enclosing rectangle in mm; overall dimensions, not datum features',
                                  uncertainty='No fit covariance exists. External characterized uncertainty is required.',
                                  inlier_fraction=1.0))


def measure_points(points_mm, options=None):
    """Measure one complete outer boundary already transformed into plane mm.

    Fit uncertainty assumes independent residuals and an exact H. Callers must
    supply a characterized external uncertainty budget before declaring QC valid.
    """
    options = options or MeasurementOptions()
    if options.kind == 'none':
        return None
    points = _points(points_mm, options)
    if options.kind == 'bounding-box':
        return _bounding_box(points)
    return _rectangle(points, options) if options.kind == 'rectangle' else _circle(points, options)
