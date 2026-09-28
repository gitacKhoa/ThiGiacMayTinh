"""
test_homography_manual.py
Test thu cong theo dung yeu cau "Cuoi tuan" cua Tuan 4: dat 1 chessboard
(vat chuan) + 1 vat khac da biet truoc kich thuoc that trong cung 1 mat
phang, chup 1 anh, roi chay script nay de kiem tra xem homography suy ra
kich thuoc co gan dung khong.

Cach dung:
python .\tests\test_homography_manual.py `
  --calib .\outputs\calibration_results\calibration_data.pkl `
  --image .\data\calibration_test\homography_test.jpg `
  --square-mm 29 --cols 8 --rows 5 `
  --true-mm 80

--cols/--rows la SO GOC TRONG cua chessboard (giong het thong so da dung
luc calibrateCamera() o Tuan 3, khong phai so o vuong).

Sau khi cua so anh hien len:
    1. Click 2 diem dau-cuoi cua vat can kiem tra (vat co kich thuoc that
       da biet, --true-mm) tren anh.
    2. Sau khi click du 2 diem, ket qua so sanh se duoc in ra console.
    3. Nhan ESC bat cu luc nao de huy.
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.getcwd())  # cho phep chay script tu thu muc goc project
from src.calibration.undistort import load_calibration, undistort_image
from src.measurement.homography import compute_homography, distance_mm, draw_chessboard

clicked_points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Da chon diem {len(clicked_points)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(
        description='Test homography: do 1 vat da biet kich thuoc that, so sanh voi ket qua he thong tinh ra.'
    )
    parser.add_argument('--calib', required=True, help='File calibration (.pkl hoac .npz)')
    parser.add_argument('--image', required=True, help='Anh chua chessboard + vat can kiem tra')
    parser.add_argument('--square-mm', type=float, required=True, help='Kich thuoc canh that cua 1 o vuong chessboard (mm)')
    parser.add_argument('--cols', type=int, required=True, help='So goc trong theo chieu ngang (giong luc calibrateCamera)')
    parser.add_argument('--rows', type=int, required=True, help='So goc trong theo chieu doc (giong luc calibrateCamera)')
    parser.add_argument('--true-mm', type=float, required=True, help='Kich thuoc that (mm) cua vat se click de kiem tra')
    args = parser.parse_args()

    mtx, dist = load_calibration(args.calib)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {args.image}")

    undistorted = undistort_image(img, mtx, dist, crop=True)

    pattern_size = (args.cols, args.rows)
    H, corners = compute_homography(undistorted, args.square_mm, pattern_size)
    if H is None:
        print(f"Khong tim thay chessboard {pattern_size[0]}x{pattern_size[1]} goc trong trong anh.")
        print("Kiem tra lai: --cols/--rows co dung so goc trong khong (dung voi rows/cols hoan doi),")
        print("anh co du sang/net khong, chessboard co bi che khuat goc nao khong.")
        return
    print(f"Da phat hien chessboard {pattern_size[0]}x{pattern_size[1]} goc trong.")

    preview = draw_chessboard(undistorted, pattern_size, corners, True)
    cv2.putText(preview, "Click 2 diem dau-cuoi vat can do", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.namedWindow('Test homography')
    cv2.setMouseCallback('Test homography', on_mouse)

    while True:
        show = preview.copy()
        for i, p in enumerate(clicked_points):
            cv2.circle(show, p, 5, (0, 0, 255), -1)
            cv2.putText(show, str(i + 1), (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(clicked_points) == 2:
            cv2.line(show, clicked_points[0], clicked_points[1], (0, 255, 0), 2)

        cv2.imshow('Test homography', show)
        key = cv2.waitKey(20) & 0xFF
        if len(clicked_points) == 2 or key == 27:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) < 2:
        print("Chua chon du 2 diem, huy test.")
        return

    measured_mm = distance_mm(clicked_points[0], clicked_points[1], H)
    error_mm = measured_mm - args.true_mm
    error_pct = abs(error_mm) / args.true_mm * 100

    print("\n----- KET QUA TEST HOMOGRAPHY -----")
    print(f"Kich thuoc do duoc  : {measured_mm:.2f} mm")
    print(f"Kich thuoc that     : {args.true_mm:.2f} mm")
    print(f"Sai lech            : {error_mm:+.2f} mm ({error_pct:.2f}%)")
    if error_pct <= 3:
        print("=> Sai so trong nguong chap nhan duoc (<= 3%).")
    else:
        print("=> Sai so con lon, kiem tra lai: chessboard co dung mat phang voi vat khong,")
        print("   anh da undistort dung chua, hoac click diem co chinh xac khong.")


if __name__ == '__main__':
    main()
