"""Canny seeds refined by a quadratic gradient peak along the edge normal.

Refinement always uses image intensities, never a binary segmentation mask.
The returned float64 (x, y) points form an UNORDERED cloud, not a polygon.
No dimensional accuracy is implied just because coordinates are fractional.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.segmentation.segment import as_gray, validate_roi


@dataclass
class SubpixelEdgeResult:
    points: np.ndarray
    pixel_points: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    strengths: np.ndarray
    edge_map: np.ndarray
    candidate_count: int


def _bilinear(values, x, y):
    """Sample in float64 without OpenCV remap's interpolation-table rounding."""
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    return ((1 - dx) * (1 - dy) * values[y0, x0]
            + dx * (1 - dy) * values[y0, x0 + 1]
            + (1 - dx) * dy * values[y0 + 1, x0]
            + dx * dy * values[y0 + 1, x0 + 1])


def detect_subpixel_edges(image, mask=None, *, roi=None, canny_low=50,
                          canny_high=150, sigma=1.0, boundary_radius=3,
                          min_gradient=5.0, max_offset=0.75):
    """Return only valid local gradient maxima, in full input-image coordinates.

    mask gates Canny seeds to a +/- boundary_radius pixel band, including holes.
    Points with weak, flat/non-maximal profiles or unstable offsets are rejected.
    Candidates at the image/ROI edge are excluded so interpolation has support.
    strengths are Sobel gradient magnitudes on the smoothed 0..255 intensity.
    max_offset is measured along the unit gradient normal, in pixels.
    """
    gray = as_gray(image)
    x, y, w, h = validate_roi(gray.shape, roi)
    if not np.isfinite([canny_low, canny_high, sigma, min_gradient, max_offset]).all():
        raise ValueError("Edge parameters must be finite")
    if not 0 <= canny_low < canny_high or sigma <= 0 or min_gradient < 0:
        raise ValueError("Require 0 <= canny_low < canny_high, sigma > 0, min_gradient >= 0")
    if not 0 < max_offset <= 1:
        raise ValueError("max_offset must be in (0, 1]")
    if not isinstance(boundary_radius, (int, np.integer)) or boundary_radius < 1:
        raise ValueError("boundary_radius must be a positive integer")

    smoothed = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    edges = cv2.Canny(np.rint(smoothed).astype(np.uint8), canny_low, canny_high, L2gradient=True)
    allowed = np.zeros_like(gray, dtype=bool)
    # Two-pixel margin also keeps bilinear reads x0+1/y0+1 in bounds.
    if w > 4 and h > 4:
        allowed[y + 2:y + h - 2, x + 2:x + w - 2] = True
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.shape != gray.shape or mask.dtype not in (np.uint8, np.bool_):
            raise ValueError("mask must be uint8/bool with the same height and width as image")
        binary = (mask > 0).astype(np.uint8) * 255
        size = 2 * boundary_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        band = cv2.subtract(cv2.dilate(binary, kernel), cv2.erode(binary, kernel))
        allowed &= band > 0
    ys, xs = np.nonzero((edges > 0) & allowed)
    candidate_count = len(xs)
    strong = magnitude[ys, xs] > max(min_gradient, np.finfo(float).eps)
    xs, ys = xs[strong], ys[strong]
    center = magnitude[ys, xs]
    nx, ny = gx[ys, xs] / center, gy[ys, xs] / center
    minus = _bilinear(magnitude, xs - nx, ys - ny)
    plus = _bilinear(magnitude, xs + nx, ys + ny)
    curvature = minus - 2 * center + plus
    valid = (curvature < -1e-9) & (center >= minus) & (center >= plus)
    delta = np.zeros_like(center)
    # Vertex of the parabola through f(-1), f(0), f(+1).
    np.divide(0.5 * (minus - plus), curvature, out=delta, where=valid)
    valid &= np.isfinite(delta) & (np.abs(delta) <= max_offset)
    pixel_points = np.column_stack((xs[valid], ys[valid])).astype(np.float64)
    normals = np.column_stack((nx[valid], ny[valid]))
    offsets = delta[valid]
    points = pixel_points + offsets[:, None] * normals
    edge_map = np.zeros_like(gray)
    edge_map[ys[valid], xs[valid]] = 255
    return SubpixelEdgeResult(points, pixel_points, normals, offsets, center[valid], edge_map, candidate_count)


def draw_subpixel_edges(image, points, color=(0, 0, 255)):
    """Draw an overlay; the numeric coordinates remain in points/CSV/NPZ."""
    gray = as_gray(image)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N, 2) array")
    for px, py in points:
        cv2.circle(out, (int(round(px * 16)), int(round(py * 16))), 12, color,
                   thickness=-1, lineType=cv2.LINE_AA, shift=4)
    return out
