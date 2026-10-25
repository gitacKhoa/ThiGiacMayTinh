"""Persist pixel-to-mm homography for an unchanged camera and measurement plane.

Input images must already be undistorted. No camera calibration file is read here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np

from src.calibration.image_geometry import validate_image_geometry
from src.measurement.homography import (
    chessboard_real_world_points, compute_homography, points_to_mm,
)


DEFAULT_HOMOGRAPHY_FILE = 'outputs/homography.json'
COORDINATE_FRAME = 'undistorted_cropped_image_pixels_to_plane_mm'


def add_homography_arguments(parser):
    parser.add_argument('--homography-mode', choices=('compute', 'reuse'),
                        help='compute: detect board and save H; reuse: load H without a board')
    parser.add_argument('--homography-file', default=DEFAULT_HOMOGRAPHY_FILE,
                        help='JSON profile to save (compute) or load (reuse)')
    parser.add_argument('--square-mm', type=float, help='Measured chessboard square side in mm (compute only)')
    parser.add_argument('--cols', type=int, default=8, help='Number of inner corner columns')
    parser.add_argument('--rows', type=int, default=5, help='Number of inner corner rows')


def resolve_homography_mode(parser, args, *, default='reuse'):
    # Preserve older commands that requested computation using --square-mm alone.
    mode = args.homography_mode or ('compute' if args.square_mm is not None else default)
    if mode == 'compute':
        if args.square_mm is None or not np.isfinite(args.square_mm) or args.square_mm <= 0:
            parser.error('compute requires --square-mm with a finite positive value')
        if args.cols < 2 or args.rows < 2:
            parser.error('--cols and --rows must be >= 2 (inner corners)')
    elif mode == 'reuse' and args.square_mm is not None:
        parser.error('--square-mm is only used with --homography-mode compute')
    return mode


def undistortion_geometry(result, distortion):
    """Describe the pixel frame, not a path to a calibration file."""
    return dict(source_size=list(result.source_size), crop_roi=list(result.crop_roi),
                alpha=result.alpha, camera_matrix=result.camera_matrix.tolist(),
                model=result.model, source_camera_matrix=result.source_camera_matrix.tolist(),
                distortion=np.asarray(distortion).reshape(-1).tolist())


def _validate_profile(profile):
    try:
        if (profile['version'] != 1 or profile['units'] != 'mm'
                or profile['coordinate_frame'] != COORDINATE_FRAME):
            raise ValueError('Unsupported homography profile version, units or coordinate frame')
        H = np.asarray(profile['H'], dtype=np.float64)
        points_to_mm(np.empty((0, 2)), H)  # Validate shape, rank and finite values.
        size = np.asarray(profile['image_size'], dtype=float)
        if (size.shape != (2,) or not np.isfinite(size).all()
                or np.any(size <= 0) or np.any(size != np.floor(size))):
            raise ValueError('Invalid homography image_size')
        chessboard_real_world_points(profile['pattern_size'], profile['square_mm'])
        rmse = profile['fit_rmse_mm']
        if not np.isfinite(rmse) or rmse < 0:
            raise ValueError('Invalid homography fit RMSE')
        geometry = profile['preprocessing']
        if geometry is not None:
            if 'model' in geometry:
                validate_image_geometry(geometry, profile['image_size'])
            for key, shape in (('source_size', (2,)), ('crop_roi', (4,)), ('camera_matrix', (3, 3))):
                value = np.asarray(geometry[key], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid preprocessing {key}')
            distortion = np.asarray(geometry['distortion'], dtype=float)
            if (distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14)
                    or not np.isfinite(distortion).all() or not np.isfinite(geometry['alpha'])):
                raise ValueError('Invalid preprocessing distortion/alpha')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid or incomplete homography profile') from exc
    return H


def create_homography_profile(image, square_mm, pattern_size, *, preprocessing=None):
    H, corners = compute_homography(image, square_mm, pattern_size)
    if H is None:
        raise ValueError(f'Khong phat hien ban co (Chessboard {pattern_size} not found); cannot compute homography')
    residuals = points_to_mm(corners.reshape(-1, 2), H) - chessboard_real_world_points(pattern_size, square_mm)
    profile = dict(version=1, units='mm', coordinate_frame=COORDINATE_FRAME,
                   created_at=datetime.now(timezone.utc).isoformat(), H=H.tolist(),
                   image_size=[image.shape[1], image.shape[0]],
                   square_mm=float(square_mm), pattern_size=list(pattern_size),
                   fit_rmse_mm=float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))),
                   preprocessing=preprocessing)
    _validate_profile(profile)
    return profile


def save_homography_profile(path, profile):
    _validate_profile(profile)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, allow_nan=False), encoding='utf-8')


def load_homography_profile(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Homography file not found: {path}. Run --homography-mode compute first.')
    try:
        profile = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid homography JSON: {path}') from exc
    _validate_profile(profile)
    return profile


def homography_for_image(profile, image, *, preprocessing=None, allow_unverified=False):
    H = _validate_profile(profile)
    if list(profile['image_size']) != [image.shape[1], image.shape[0]]:
        raise ValueError('Homography image size mismatch; keep resolution and undistort/crop unchanged')
    stored = profile['preprocessing']
    if stored is not None and preprocessing is not None:
        if stored.get('model') != preprocessing.get('model'):
            raise ValueError('Homography preprocessing mismatch: undistortion model changed; recompute H from a NEW board capture')
        for key in ('source_size', 'crop_roi', 'alpha', 'camera_matrix', 'distortion'):
            old, new = np.asarray(stored[key]), np.asarray(preprocessing[key])
            if old.shape != new.shape or not np.allclose(old, new, rtol=1e-10, atol=1e-10):
                raise ValueError(f'Homography preprocessing mismatch: {key}; recompute H')
        if 'model' in preprocessing:
            validate_image_geometry(preprocessing, image.shape[1::-1])
            if not np.allclose(stored['source_camera_matrix'], preprocessing['source_camera_matrix'], rtol=1e-10, atol=1e-10):
                raise ValueError('Homography preprocessing mismatch: source camera matrix; recompute H')
    else:
        if not allow_unverified:
            raise ValueError('Cannot verify image/H undistortion geometry. Recapture or regenerate image and H; '
                             'externally verified legacy pairs require explicit allow_unverified.')
        warnings.warn('Only image size can be checked for externally undistorted images; '
                      'keep the same undistortion, crop and pixel coordinates.', UserWarning, stacklevel=2)
    return H
