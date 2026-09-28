"""
undistort.py
Nap thong so calibration (mtx, dist) va undistort anh.
Dung chung cho verify_calibration.py va cac module sau nay (homography, segmentation...).
"""

import pickle

import cv2
import numpy as np


def load_calibration(path):
    """Doc mtx, dist tu file .pkl hoac .npz, ho tro vai kieu ten khoa pho bien."""
    if path.endswith('.npz'):
        data = np.load(path)
        return np.array(data['mtx']), np.array(data['dist'])

    with open(path, 'rb') as f:
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
    return np.array(mtx), np.array(dist)


def undistort_image(img, mtx, dist, crop=True):
    """
    Sua meo anh bang mtx/dist.
    crop=True (mac dinh): cat bo vien den sinh ra sau undistort (alpha=0).
    crop=False: giu nguyen toan bo anh goc, se co vien den cong o goc/canh.
    """
    h, w = img.shape[:2]
    alpha = 0 if crop else 1
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha)
    undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

    if crop:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y + rh, x:x + rw]

    return undistorted
