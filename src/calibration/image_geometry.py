"""File adapters for clean undistorted images and hash-bound geometry sidecars.

Keep image.png and image.png.geometry.json together. No UI and no calibration
pickle loading here. This is provenance checking, not camera-motion detection.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibration.undistort import UNDISTORTION_MODEL


def geometry_sidecar_path(image_path):
    return Path(str(image_path) + '.geometry.json')


def validate_image_geometry(geometry, size):
    try:
        if geometry['model'] != UNDISTORTION_MODEL:
            raise ValueError('Legacy/unknown undistortion geometry; recapture or regenerate from RAW')
        source = np.asarray(geometry['source_size'], dtype=float)
        crop = np.asarray(geometry['crop_roi'], dtype=float)
        K = np.asarray(geometry['camera_matrix'], dtype=float)
        original = np.asarray(geometry['source_camera_matrix'], dtype=float)
        d = np.asarray(geometry['distortion'], dtype=float)
        if (source.shape != (2,) or crop.shape != (4,) or K.shape != (3, 3)
                or original.shape != (3, 3) or d.ndim != 1 or d.size not in (4, 5, 8, 12, 14)):
            raise ValueError('Invalid image geometry shapes')
        numbers = np.r_[source, crop, K.ravel(), original.ravel(), d, geometry['alpha']]
        if (not np.isfinite(numbers).all() or np.any(source <= 0)
                or np.any(source != np.floor(source)) or np.any(crop != np.floor(crop))
                or np.any(crop < 0) or np.any(crop[2:] <= 0)
                or np.any(crop[:2] + crop[2:] > source)
                or list(crop[2:]) != list(size) or geometry['alpha'] not in (0., 1.)
                or min(K[0, 0], K[1, 1], original[0, 0], original[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or not np.allclose(original[2], [0, 0, 1])):
            raise ValueError('Invalid image geometry values or image size')
        if not np.isclose(K[0, 0] / original[0, 0], K[1, 1] / original[1, 1], rtol=1e-12, atol=1e-12):
            raise ValueError('Image geometry contains unequal axis scaling')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Incomplete/invalid image geometry; regenerate from RAW') from exc


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_undistorted_image(path, image, geometry):
    """Save an image plus metadata bound to its encoded bytes; raises on failure."""
    path = Path(path)
    validate_image_geometry(geometry, image.shape[1::-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot save image: {path}')
    payload = dict(version=1, image_size=list(image.shape[1::-1]),
                   sha256=_digest(path), preprocessing=geometry)
    geometry_sidecar_path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def load_image_geometry(path, image, *, allow_missing=False):
    """Missing sidecars require explicit opt-in; corrupt/mismatched ones never pass."""
    sidecar = geometry_sidecar_path(path)
    if not sidecar.is_file():
        if allow_missing:
            return None
        raise ValueError(f'Missing image geometry sidecar: {sidecar}. Recapture/regenerate from RAW; '
                         'only for externally verified images use --allow-unverified-geometry.')
    try:
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
        if (payload['version'] != 1 or payload['image_size'] != list(image.shape[1::-1])
                or payload['sha256'] != _digest(path)):
            raise ValueError('Image geometry sidecar does not match image bytes/size; regenerate it from RAW')
        geometry = payload['preprocessing']
        validate_image_geometry(geometry, image.shape[1::-1])
        return geometry
    except (KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f'Invalid image geometry sidecar: {sidecar}') from exc
