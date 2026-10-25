"""
capture_webcam.py
Mo webcam, xem truoc, va chup anh luu lai de dung cho verify_calibration.py
(hoac de chup anh calibration/san pham cho cac buoc sau).

Cach dung:
    python tests/capture_webcam.py --output test.jpg
    python tests/capture_webcam.py --output data/calibration_test/ --prefix calib --width 1280 --height 720
    python tests/capture_webcam.py --calib outputs/calibration_results/calibration_data.pkl --calibration-size 1280 720 --width 1280 --height 720 --homography-mode compute --square-mm 29 --output undistorted.jpg

Neu dung --calib, anh hien thi va anh luu la anh da undistort, co the dua
truc tiep vao buoc homography. Khong dung --calib neu dang chup anh goc cho
buoc calibration.
Voi --calib: compute tinh/luu H khi chup anh co ban co; reuse (mac dinh)
nap H cu, khong can ban co. Camera va mat phang do phai giu nguyen.

Phim tat khi cua so preview dang mo:
    SPACE hoac C   -> chup va luu anh
    Q hoac ESC     -> thoat
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration.undistort import load_calibration_data, undistort_with_metadata
from src.measurement.homography_profile import (
    add_homography_arguments, create_homography_profile, homography_for_image,
    load_homography_profile, resolve_homography_mode, save_homography_profile,
    undistortion_geometry,
)


def open_camera(index, width=None, height=None):
    # Thu dung DSHOW tren Windows de co quyen can thiep sau vao thuoc tinh webcam
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        # Fallback neu DSHOW fail
        cap.release()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Khong mo duoc camera index {index}. Thu doi --camera sang so khac (0, 1, 2...)."
            )

    # ====================================================
    # CHEN CODE TAT AUTO-FOCUS VA CHINH TIEU CU THU CONG
    # ====================================================
    # Tat Auto-Focus (0 = Tat, 1 = Bat)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # (Tuy chon) Chinh cung tieu cu (Manual Focus).
    # Bo comment dong duoi va doi gia tri (0, 5, 10, 15, 20...) de tim khoang net phu hop
    # cap.set(cv2.CAP_PROP_FOCUS, 15)
    # ====================================================

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def build_save_path(output, prefix, index):
    """
    - Neu output la 1 thu muc (ket thuc bang / hoac \\, hoac da ton tai san):
      luu nhieu anh vao do, ten tu tang: prefix_000.jpg, prefix_001.jpg, ...
    - Neu output la 1 file cu the: lan chup dau tien dung dung ten do,
      cac lan sau tu them so thu tu de khong ghi de.
    """
    if index < 0:
        raise ValueError('index must be nonnegative')
    if output.endswith(('/', '\\')) or os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, f"{prefix}_{index:03d}.jpg")
        while os.path.exists(path):
            index += 1
            path = os.path.join(output, f"{prefix}_{index:03d}.jpg")
        return path
    root, ext = os.path.splitext(output)
    path = output if index == 0 and ext else f"{root}_{index:03d}{ext or '.jpg'}"
    while os.path.exists(path):
        index += 1
        path = f"{root}_{index:03d}{ext or '.jpg'}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mo webcam va chup anh de dung cho pipeline do luong.')
    parser.add_argument('--output', default='captured.jpg', help='Duong dan file luu, hoac thu muc neu chup nhieu anh')
    parser.add_argument('--prefix', default='capture', help='Tien to ten file khi luu nhieu anh vao 1 thu muc')
    parser.add_argument('--camera', type=int, default=0, help='Chi so camera (0, 1, 2...)')
    parser.add_argument('--width', type=int, default=None, help='Do phan giai chieu rong mong muon')
    parser.add_argument('--height', type=int, default=None, help='Do phan giai chieu cao mong muon')
    parser.add_argument('--calib', help='File calibration PKL/NPZ; bat buoc de luu anh da undistort')
    parser.add_argument('--calibration-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='Kich thuoc calibration (width height), dung cho file cu khong co image_size')
    add_homography_arguments(parser)
    args = parser.parse_args(argv)
    homography_mode = resolve_homography_mode(parser, args, default='reuse' if args.calib else None)
    if homography_mode is not None and not args.calib:
        parser.error('Webcam homography requires --calib to undistort frames first')

    calibration = None
    calibration_size = None
    profile = None
    if args.calib:
        calibration = load_calibration_data(args.calib)
        calibration_size = calibration.get('image_size', args.calibration_size)
        if calibration_size is None:
            parser.error('Calibration cu khong co image_size; hay them --calibration-size WIDTH HEIGHT')
        if args.calibration_size is not None and tuple(calibration_size) != tuple(args.calibration_size):
            parser.error('--calibration-size conflicts with calibration metadata')
    if homography_mode == 'reuse':
        try:
            profile = load_homography_profile(args.homography_file)
        except (ValueError, OSError) as exc:
            parser.exit(1, f'Homography error: {exc}\n')
        print(f'Da nap homography: {args.homography_file}; khong can ban co.')
        print('Giu nguyen camera, tieu cu, mat phang do, do phan giai va undistort/crop.')

    cap = open_camera(args.camera, args.width, args.height)
    saved_paths = []
    index = 0

    print("SPACE hoac 'c' de chup anh, 'q' hoac ESC de thoat.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Khong doc duoc frame tu camera.")
                break

            frame_for_save = frame
            preprocessing = None
            if calibration is not None:
                geometry = undistort_with_metadata(
                    frame, calibration['mtx'], calibration['dist'], calibration_size=calibration_size
                )
                frame_for_save = geometry.image
                preprocessing = undistortion_geometry(geometry, calibration['dist'])

            preview = frame_for_save.copy()
            cv2.putText(preview, "SPACE: chup | Q: thoat", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if saved_paths:
                cv2.putText(preview, f"Da chup: {len(saved_paths)}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Webcam - nhan SPACE de chup', preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(' '), ord('c'), ord('C')):
                if homography_mode == 'compute':
                    try:
                        profile = create_homography_profile(
                            frame_for_save, args.square_mm, (args.cols, args.rows), preprocessing=preprocessing)
                    except ValueError as exc:
                        print(f'Homography error: {exc}. Chua luu anh/H; hay chup lai.')
                        continue
                elif homography_mode == 'reuse':
                    homography_for_image(profile, frame_for_save, preprocessing=preprocessing)
                path = build_save_path(args.output, args.prefix, index)
                if not cv2.imwrite(path, frame_for_save):
                    raise OSError(f'Cannot save image: {path}')
                if homography_mode == 'compute':
                    save_homography_profile(args.homography_file, profile)
                    print(f'Da luu homography: {args.homography_file} (RMSE {profile["fit_rmse_mm"]:.4f} mm)')
                saved_paths.append(path)
                print(f"Da luu: {path}")
                index += 1
            elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                break
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f'Capture error: {exc}\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not saved_paths:
        print("Chua chup anh nao.")
        return

    print(f"\nTong cong da luu {len(saved_paths)} anh.")
    if calibration is None:
        print("Anh da luu la anh goc. Hay undistort truoc khi chay homography.")
        print("Vi du chay tiep buoc kiem chung calibration:")
        print(f'  python src/verify_calibration.py --calib outputs/calibration_results/calibration_data.pkl --image "{saved_paths[0]}" --grid')
    else:
        print("Anh da luu la anh da undistort; co the dung truc tiep cho homography.")
        print('Chay pipeline de tu dong doi toa do bien sang mm:')
        print(f'  python -m src.pipeline --image "{saved_paths[0]}" --already-undistorted '
              f'--homography-mode reuse --homography-file "{args.homography_file}" --select-roi')


if __name__ == '__main__':
    main()
