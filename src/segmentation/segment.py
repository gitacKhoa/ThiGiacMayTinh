"""Segment objects without changing the coordinate system of the input image.

Masks are uint8 (0/255). ROI is (x, y, width, height) in the input image;
returned masks and contours always use FULL image coordinates. Use an ROI
around the workpiece when a reference chessboard is visible in the same frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    contours: list
    roi: tuple
    threshold: object
    component_count: int


def as_gray(image):
    """Validate an 8-bit gray/BGR/BGRA image; never rescale intensities silently."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("image must be a nonempty uint8 array")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("image must be grayscale, BGR or BGRA")


def validate_roi(shape, roi):
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width, height
    if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("roi must contain four integers: x, y, width, height")
    x, y, w, h = map(int, roi)
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise ValueError("roi must lie entirely inside the input image")
    return x, y, w, h


def _odd_kernel(value, name):
    if not isinstance(value, (int, np.integer)) or value < 1 or value % 2 != 1:
        raise ValueError(f"{name} must be a positive odd integer (1 disables it)")


def segment_image(image, *, method="otsu", foreground="dark", roi=None,
                  blur_size=5, morphology_size=3, min_area=100,
                  keep_largest=True, reject_border=True, fill_holes=False,
                  adaptive_block_size=31, adaptive_c=5.0,
                  hsv_lower=None, hsv_upper=None):
    """Threshold, clean and select connected components of a workpiece.

    Otsu/adaptive use explicit dark/bright polarity. HSV bounds use OpenCV
    H=0..179, S/V=0..255; lower H > upper H selects a range across hue zero.
    By default, reject components touching the ROI/image border: these are
    often background or truncated objects. Expand the ROI to retain the object.
    Holes are preserved unless fill_holes=True (e.g. a printed solid object).
    min_area is foreground area in pixels, not physical square millimetres.
    """
    gray = as_gray(image)
    x, y, w, h = validate_roi(gray.shape, roi)
    if method not in ("otsu", "adaptive", "hsv"):
        raise ValueError("method must be otsu, adaptive or hsv")
    if foreground not in ("dark", "bright"):
        raise ValueError("foreground must be dark or bright")
    _odd_kernel(blur_size, "blur_size")
    _odd_kernel(morphology_size, "morphology_size")
    if not np.isfinite(min_area) or min_area <= 0:
        raise ValueError("min_area must be finite and positive")
    local_gray = gray[y:y + h, x:x + w]
    local_gray = cv2.GaussianBlur(local_gray, (blur_size, blur_size), 0)
    threshold_value = None
    polarity = cv2.THRESH_BINARY_INV if foreground == "dark" else cv2.THRESH_BINARY
    if method == "otsu":
        # A flat ROI contains no contrast from which to identify an object.
        if int(local_gray.max()) == int(local_gray.min()):
            binary = np.zeros_like(local_gray)
        else:
            threshold_value, binary = cv2.threshold(local_gray, 0, 255, polarity | cv2.THRESH_OTSU)
    elif method == "adaptive":
        _odd_kernel(adaptive_block_size, "adaptive_block_size")
        if adaptive_block_size < 3 or not np.isfinite(adaptive_c):
            raise ValueError("adaptive_block_size >= 3 and finite adaptive_c required")
        binary = cv2.adaptiveThreshold(local_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       polarity, adaptive_block_size, adaptive_c)
    else:
        if image.ndim != 3 or hsv_lower is None or hsv_upper is None:
            raise ValueError("HSV segmentation requires a color image and both HSV bounds")
        lower = np.asarray(hsv_lower, dtype=float)
        upper = np.asarray(hsv_upper, dtype=float)
        limits = np.array([179, 255, 255])
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or np.any(lower < 0) or np.any(upper < 0)
                or np.any(lower > limits) or np.any(upper > limits)
                or np.any(lower != np.floor(lower)) or np.any(upper != np.floor(upper))
                or np.any(lower[1:] > upper[1:])):
            raise ValueError("Invalid HSV bounds: H=0..179, S/V=0..255")
        hsv = cv2.cvtColor(image[y:y + h, x:x + w, :3], cv2.COLOR_BGR2HSV)
        lo, hi = lower.astype(np.uint8), upper.astype(np.uint8)
        if lo[0] <= hi[0]:
            binary = cv2.inRange(hsv, lo, hi)
        else:
            binary = cv2.bitwise_or(
                cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], dtype=np.uint8)),
                cv2.inRange(hsv, np.array([0, lo[1], lo[2]], dtype=np.uint8), hi))

    if morphology_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_size, morphology_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Pad before flood fill so holes are distinct from exterior background,
    # including when foreground touches the top-left pixel of the ROI.
    if fill_holes:
        padded = np.pad(binary, 1)
        flooded = padded.copy()
        cv2.floodFill(flooded, None, (0, 0), 255)
        binary = cv2.bitwise_or(padded, cv2.bitwise_not(flooded))[1:-1, 1:-1]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    selected = []
    for label in range(1, count):
        left, top, cw, ch, area = stats[label]
        touches_border = left == 0 or top == 0 or left + cw == w or top + ch == h
        if area >= min_area and not (reject_border and touches_border):
            selected.append(label)
    if keep_largest and selected:
        selected = [max(selected, key=lambda label: stats[label, cv2.CC_STAT_AREA])]
    local_mask = (np.isin(labels, selected).astype(np.uint8) * 255)
    mask = np.zeros_like(gray)
    mask[y:y + h, x:x + w] = local_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return SegmentationResult(mask, list(contours), (x, y, w, h), threshold_value, len(selected))
