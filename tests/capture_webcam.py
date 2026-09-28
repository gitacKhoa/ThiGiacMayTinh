"""
capture_webcam.py
Mo webcam, xem truoc, va chup anh luu lai de dung cho verify_calibration.py
(hoac de chup anh calibration/san pham cho cac buoc sau).

Cach dung:
    python tests/capture_webcam.py --output test.jpg
    python tests/capture_webcam.py --output data/calibration_test/ --prefix calib --width 1280 --height 720

Phim tat khi cua so preview dang mo:
    SPACE hoac C   -> chup va luu anh
    Q hoac ESC     -> thoat
"""

import argparse
import os
import platform
import cv2


def open_camera(index, width=None, height=None):
    # Thu dung DSHOW tren Windows de co quyen can thiep sau vao thuoc tinh webcam
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        # Fallback neu DSHOW fail
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
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
    if output.endswith(('/', '\\')) or os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
        return os.path.join(output, f"{prefix}_{index:03d}.jpg")

    if index == 0:
        return output

    root, ext = os.path.splitext(output)
    return f"{root}_{index:03d}{ext or '.jpg'}"


def main():
    parser = argparse.ArgumentParser(description='Mo webcam va chup anh de dung cho pipeline do luong.')
    parser.add_argument('--output', default='captured.jpg', help='Duong dan file luu, hoac thu muc neu chup nhieu anh')
    parser.add_argument('--prefix', default='capture', help='Tien to ten file khi luu nhieu anh vao 1 thu muc')
    parser.add_argument('--camera', type=int, default=0, help='Chi so camera (0, 1, 2...)')
    parser.add_argument('--width', type=int, default=None, help='Do phan giai chieu rong mong muon')
    parser.add_argument('--height', type=int, default=None, help='Do phan giai chieu cao mong muon')
    args = parser.parse_args()

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

            preview = frame.copy()
            cv2.putText(preview, "SPACE: chup | Q: thoat", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if saved_paths:
                cv2.putText(preview, f"Da chup: {len(saved_paths)}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Webcam - nhan SPACE de chup', preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord(' '), ord('c'), ord('C')):
                path = build_save_path(args.output, args.prefix, index)
                cv2.imwrite(path, frame)
                saved_paths.append(path)
                print(f"Da luu: {path}")
                index += 1
            elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not saved_paths:
        print("Chua chup anh nao.")
        return

    print(f"\nTong cong da luu {len(saved_paths)} anh.")
    print("Vi du chay tiep buoc kiem chung calibration:")
    print(f"  python verify_calibration.py --calib calibration.pkl --image {saved_paths[0]} --grid")


if __name__ == '__main__':
    main()
