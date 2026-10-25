"""
undistort.py
Nap thong so calibration (mtx, dist) va undistort anh.
Dung chung cho verify_calibration.py va cac module sau nay (homography, segmentation...).
"""

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

UNDISTORTION_MODEL = 'centered_uniform_scale_v2'


def load_calibration_data(path):
    """Load trusted local PKL/NPZ, normalize aliases and retain metadata."""
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
    else:
        with path.open('rb') as f:
            data = pickle.load(f)

    mtx = data.get('mtx')
    if mtx is None:
        mtx = data.get('camera_matrix')
    dist = data.get('dist')
    if dist is None:
        dist = data.get('distortion_coefficients')

    if mtx is None or dist is None:
        raise KeyError(
            f"Khong tim thay 'mtx'/'dist' trong file. Cac khoa hien co: {list(data.keys())}"
        )
    mtx, dist = _validate_parameters(mtx, dist)
    result = dict(data)
    result.update(mtx=mtx, dist=dist)
    if 'image_size' in data:
        result['image_size'] = _validate_size(data['image_size'])
    return result


def load_calibration(path):
    """Backward-compatible (camera matrix, distortion) pair."""
    data = load_calibration_data(path)
    return data['mtx'], data['dist']


def _validate_size(size):
    values = np.asarray(size, dtype=float)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(values != np.floor(values))):
        raise ValueError('image_size must be positive integer (width, height)')
    return tuple(int(v) for v in values)


def _validate_parameters(mtx, dist):
    mtx = np.asarray(mtx, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    if (mtx.shape != (3, 3) or not np.isfinite(mtx).all()
            or mtx[0, 0] <= 0 or mtx[1, 1] <= 0
            or not np.allclose(mtx[2], [0, 0, 1])):
        raise ValueError('Invalid camera matrix')
    if (dist.ndim not in (1, 2) or (dist.ndim == 2 and 1 not in dist.shape)
            or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()):
        raise ValueError('Invalid distortion coefficients')
    return mtx, dist.reshape(1, -1)


@dataclass
class UndistortionResult:
    image: np.ndarray
    camera_matrix: np.ndarray
    crop_roi: tuple
    source_size: tuple
    alpha: float
    source_camera_matrix: np.ndarray
    model: str = UNDISTORTION_MODEL


def undistort_with_metadata(img, mtx, dist, crop=True, *, calibration_size=None):
    """Use one geometry for all downstream stages; no resize after cropping.

    camera_matrix describes the returned image (principal point shifted by
    crop offset). crop_roi uses coordinates before cropping. A known size
    mismatch is rejected instead of silently applying incompatible intrinsics.
    Legacy files require callers to supply calibration_size to verify geometry.
    """
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise ValueError('img must be a nonempty image')
    mtx, dist = _validate_parameters(mtx, dist)
    h, w = img.shape[:2]
    if calibration_size is None:
        warnings.warn('Calibration image_size is unknown; confirm the original resolution '
                      'and pass calibration_size=(width, height).', UserWarning, stacklevel=2)
    elif _validate_size(calibration_size) != (w, h):
        raise ValueError(f'Image size {(w, h)} differs from calibration size {tuple(calibration_size)}')
    alpha = 0.0 if crop else 1.0
    # The default OpenCV branch optimizes fx/fy independently to fill the canvas.
    # Centering the principal point selects ONE scale for both original focal
    # lengths, preserving fx/fy. Cropping only translates the principal point.
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), alpha, (w, h), centerPrincipalPoint=True)
    if (not np.isfinite(new_mtx).all() or min(new_mtx[0, 0], new_mtx[1, 1]) <= 0
            or not np.isclose(new_mtx[0, 0] / mtx[0, 0], new_mtx[1, 1] / mtx[1, 1],
                              rtol=1e-12, atol=1e-12)):
        raise ValueError('Calibration cannot produce aspect-preserving undistortion')
    corrected = cv2.undistort(img, mtx, dist, None, new_mtx)
    crop_roi = (0, 0, w, h)
    if crop:
        x, y, rw, rh = map(int, roi)
        if rw <= 0 or rh <= 0:
            raise ValueError('Calibration produced an empty valid ROI')
        corrected = corrected[y:y + rh, x:x + rw]
        new_mtx = new_mtx.copy()
        new_mtx[0, 2] -= x
        new_mtx[1, 2] -= y
        crop_roi = (x, y, rw, rh)
    return UndistortionResult(corrected, new_mtx, crop_roi, (w, h), alpha, mtx.copy())


def undistort_image(img, mtx, dist, crop=True, *, calibration_size=None):
    """
    Sua meo anh bang mtx/dist.
    crop=True (mac dinh): cat bo vien den sinh ra sau undistort (alpha=0).
    crop=False: truong nhin rong hon (alpha=1), co the co vien den.
    Hai truc dung cung he so phong; khong resize ve kich thuoc co dinh sau crop.
    """
    return undistort_with_metadata(img, mtx, dist, crop, calibration_size=calibration_size).image
