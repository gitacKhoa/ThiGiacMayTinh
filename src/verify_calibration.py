"""
verify_calibration.py
Kiem chung lai ket qua camera calibration (mtx, dist) truoc khi dung cho
cac buoc sau (homography, segmentation, edge detection...).

Cach dung:
    python src/verify_calibration.py --calib outputs/calibration_results/calibration_data.pkl --image data/calibration_test/calib_000.jpg --output outputs/undistort_check.png
    python src/verify_calibration.py --calib outputs/calibration_results/calibration_data.npz --image data/calibration_test/calib_000.jpg --grid --crop
"""

import argparse
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


def undistort_image(img, mtx, dist, crop=False):
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0 if crop else 1)
    undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

    if crop:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y + rh, x:x + rw]

    return undistorted


def draw_grid_overlay(img, spacing=40, color=(0, 0, 255)):
    """Ve luoi len anh de mat thuong de thay duong thang bi cong hay khong."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(out, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def make_side_by_side(original, undistorted, label_h=30, gap=20):
    """Ghep 2 anh canh nhau, tu resize cho bang chieu cao neu can."""
    h = max(original.shape[0], undistorted.shape[0])

    def resize_to_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))

    original = resize_to_h(original, h)
    undistorted = resize_to_h(undistorted, h)

    w = original.shape[1] + undistorted.shape[1] + gap
    canvas = np.full((h + label_h, w, 3), 255, dtype=np.uint8)

    canvas[label_h:label_h + h, :original.shape[1]] = original
    x_offset = original.shape[1] + gap
    canvas[label_h:label_h + h, x_offset:x_offset + undistorted.shape[1]] = undistorted

    cv2.putText(canvas, 'Anh goc', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, 'Sau undistort', (x_offset + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description='Kiem chung calibration bang cach undistort 1 anh test.')
    parser.add_argument('--calib', required=True, help='Duong dan file calibration (.pkl hoac .npz)')
    parser.add_argument('--image', required=True, help='Duong dan anh test de kiem chung')
    parser.add_argument('--output', default='undistort_check.png', help='Duong dan luu anh so sanh')
    parser.add_argument('--grid', action='store_true', help='Ve luoi de de quan sat duong thang bi cong hay khong')
    parser.add_argument('--crop', action='store_true', help='Cat bo vien den sau khi undistort (alpha=0)')
    args = parser.parse_args()

    mtx, dist = load_calibration(args.calib)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")

    undistorted = undistort_image(img, mtx, dist, crop=args.crop)

    if args.grid:
        img_show = draw_grid_overlay(img)
        undistorted_show = draw_grid_overlay(undistorted)
    else:
        img_show, undistorted_show = img, undistorted

    comparison = make_side_by_side(img_show, undistorted_show)
    cv2.imwrite(args.output, comparison)

    print(f"Da luu anh so sanh tai: {args.output}")
    print(f"Ma tran noi tai (mtx):\n{mtx}")
    print(f"He so meo (dist):\n{dist}")
    print("\n=> Kiem tra bang mat: neu duong thang trong 'Anh goc' bi cong ma sau")
    print("   undistort thang lai ro ret (dung --grid de thay ro hon), la calibration dat yeu cau.")


if __name__ == '__main__':
    main()
