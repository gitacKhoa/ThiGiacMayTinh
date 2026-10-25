"""
verify_calibration.py
Kiem chung lai ket qua camera calibration (mtx, dist) truoc khi dung cho
cac buoc sau (homography, segmentation, edge detection...).

Cach dung:
    python src/verify_calibration.py --calib outputs/calibration_results/calibration_data.pkl --image data/calibration_test/calib_000.jpg --output outputs/undistort_check.png
    python src/verify_calibration.py --calib outputs/calibration_results/calibration_data.npz --image data/calibration_test/calib_000.jpg --grid --crop
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration.undistort import load_calibration_data, undistort_with_metadata
from src.calibration.image_geometry import save_undistorted_image
from src.measurement.homography_profile import undistortion_geometry


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
    """Compare at native pixel scale; pad instead of resizing either image."""
    h = max(original.shape[0], undistorted.shape[0])

    w = original.shape[1] + undistorted.shape[1] + gap
    canvas = np.full((h + label_h, w, 3), 255, dtype=np.uint8)

    canvas[label_h:label_h + original.shape[0], :original.shape[1]] = original
    x_offset = original.shape[1] + gap
    canvas[label_h:label_h + undistorted.shape[0], x_offset:x_offset + undistorted.shape[1]] = undistorted

    cv2.putText(canvas, 'Anh goc', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, 'Sau undistort', (x_offset + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description='Kiem chung calibration bang cach undistort 1 anh test.')
    parser.add_argument('--calib', required=True, help='Duong dan file calibration (.pkl hoac .npz)')
    parser.add_argument('--image', required=True, help='Duong dan anh test de kiem chung')
    parser.add_argument('--output', default='undistort_check.png', help='Duong dan luu anh so sanh')
    parser.add_argument('--undistorted-output', help='Also save CLEAN undistorted image + geometry sidecar for pipeline')
    parser.add_argument('--grid', action='store_true', help='Ve luoi de de quan sat duong thang bi cong hay khong')
    parser.add_argument('--crop', action='store_true', help='Cat bo vien den sau khi undistort (alpha=0)')
    parser.add_argument('--calibration-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args()

    calibration = load_calibration_data(args.calib)
    mtx, dist = calibration['mtx'], calibration['dist']
    image_size = calibration.get('image_size', args.calibration_size)
    if image_size is None:
        parser.error('Legacy calibration requires --calibration-size WIDTH HEIGHT')
    if args.calibration_size is not None and tuple(image_size) != tuple(args.calibration_size):
        parser.error('--calibration-size conflicts with calibration metadata')
    paths = [Path(p).resolve() for p in (args.image, args.output, args.undistorted_output) if p]
    if len(set(paths)) != len(paths):
        parser.error('Input, comparison output and clean image output must be different files')

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")

    result = undistort_with_metadata(img, mtx, dist, crop=args.crop, calibration_size=image_size)
    undistorted = result.image
    if args.undistorted_output:
        save_undistorted_image(args.undistorted_output, undistorted, undistortion_geometry(result, dist))

    if args.grid:
        img_show = draw_grid_overlay(img)
        undistorted_show = draw_grid_overlay(undistorted)
    else:
        img_show, undistorted_show = img, undistorted

    comparison = make_side_by_side(img_show, undistorted_show)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.output, comparison):
        raise OSError(f'Cannot save image: {args.output}')

    print(f"Da luu anh so sanh tai: {args.output}")
    print(f"Ma tran noi tai (mtx):\n{mtx}")
    print(f"He so meo (dist):\n{dist}")
    print("\n=> Kiem tra bang mat: neu duong thang trong 'Anh goc' bi cong ma sau")
    print("   undistort thang lai ro ret (dung --grid de thay ro hon). Day chi la kiem tra dinh tinh,")
    print("   khong chung minh do chinh xac do. Phoi canh nghieng van con sau undistort.")
    print(f"Output fx/fy={result.camera_matrix[0,0]/result.camera_matrix[1,1]:.6f}; "
          f"original fx/fy={mtx[0,0]/mtx[1,1]:.6f} (preserved).")


if __name__ == '__main__':
    main()
