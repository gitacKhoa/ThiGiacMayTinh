"""
homography.py
Phat hien vat chuan (chessboard) trong anh da undistort, va tinh
homography de quy doi toa do pixel sang toa do thuc (mm) tren mat
phang chua vat chuan.

Gia dinh quan trong: vat can do phai nam CUNG MAT PHANG voi chessboard
(dat vat len cung 1 be mat phang voi bang, khong chong len nhau).
Neu vat co do day dang ke, ket qua se co sai so he thong tang dan
theo do day cua vat (vi diem do khong con dung tren mat phang chuan).

pattern_size = (cols, rows) la SO GOC TRONG (inner corners), KHONG PHAI
so o vuong - VD ban co chessboard co 10x7 o vuong thi pattern_size = (9, 6).
Day cung chinh la (cols, rows) da dung luc calibrateCamera() o Tuan 3.
"""

import cv2
import numpy as np

CORNER_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_chessboard(image, pattern_size):
    """
    Tim goc trong cua chessboard trong anh, da refine toi do chinh xac sub-pixel.
    Tra ve (found, corners) - corners shape (N,1,2). corners=None neu khong thay.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_SUBPIX_CRITERIA)
    return True, corners


def chessboard_real_world_points(pattern_size, square_mm):
    """
    Toa do thuc (mm) cua tung goc trong chessboard, dat goc toa do (0,0)
    tai goc dau tien - cung thu tu voi corners tra ve tu findChessboardCorners
    (quet tung hang, tu trai sang phai).
    """
    if (len(pattern_size) != 2
            or any(not isinstance(v, (int, np.integer)) or v < 2 for v in pattern_size)):
        raise ValueError('pattern_size must be two integers >= 2')
    if not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError('square_mm must be finite and positive')
    cols, rows = pattern_size
    obj_points = np.zeros((cols * rows, 2), dtype=np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return obj_points


def compute_homography(image, square_mm, pattern_size):
    """
    Detect chessboard trong anh, tinh homography pixel -> mm bang TOAN BO
    cac goc tim duoc (nhieu diem hon 1 marker ArUco nen on dinh/chinh xac hon).

    Tra ve: (H, corners) - H la ma tran 3x3 (dung cho pixel_to_mm/distance_mm),
    corners de ve lai kiem tra truc quan. Tra ve (None, None) neu khong thay
    chessboard - kiem tra lai --cols/--rows co dung so goc trong khong, hoac
    anh co du sang/net khong.
    """
    real_pts = chessboard_real_world_points(pattern_size, square_mm)
    found, corners = detect_chessboard(image, pattern_size)
    if not found:
        return None, None

    pixel_pts = corners.reshape(-1, 2).astype(np.float32)

    H, _ = cv2.findHomography(pixel_pts, real_pts)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError('Cannot compute a nondegenerate homography')
    return H, corners


def pixel_to_mm(point, H):
    """Quy doi 1 diem pixel (x, y) sang toa do mm tren mat phang vat chuan."""
    return points_to_mm(np.asarray(point, dtype=np.float64).reshape(1, 2), H)[0]


def points_to_mm(points, H):
    """Transform an (N, 2) subpixel cloud; reject points at projective infinity."""
    points = np.asarray(points, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('points must be a finite (N, 2) array')
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError('H must be a finite nonsingular 3x3 matrix')
    H = H / np.max(np.abs(H))
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ H.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError('Homography maps points to infinity')
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def distance_mm(p1, p2, H):
    """Tinh khoang cach thuc (mm) giua 2 diem pixel, dua tren homography H."""
    a = pixel_to_mm(p1, H)
    b = pixel_to_mm(p2, H)
    return float(np.linalg.norm(a - b))


def draw_chessboard(image, pattern_size, corners, found=True):
    """Ve cac goc chessboard da detect duoc len anh, dung de kiem tra truc quan."""
    out = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, pattern_size, corners, found)
    return out
