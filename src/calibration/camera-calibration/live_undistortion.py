import cv2
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from src.calibration.undistort import load_calibration_data, undistort_with_metadata

# Live undistortion parameters
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, 'outputs', 'calibration_results', 'calibration_data.pkl')

def live_undistortion(calibration_size=None):
    """
    Demonstrate live camera undistortion using calibration results.
    """
    # Check if calibration file exists
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
        print("Please run camera_calibration.py first to generate calibration data.")
        return

    # Load calibration data
    calibration_data = load_calibration_data(CALIBRATION_FILE)
    mtx = calibration_data['mtx']
    dist = calibration_data['dist']
    calibration_size = calibration_data.get('image_size', calibration_size)
    if calibration_size is None:
        raise ValueError('Legacy calibration: supply --calibration-size WIDTH HEIGHT')

    print("Loaded camera calibration data:")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients: {dist.ravel()}")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, calibration_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration_size[1])

    if not cap.isOpened():
        cap.release()
        print(f"Error: Could not open camera {CAMERA_ID}")
        return

    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    ok, initial_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError('Cannot read initial camera frame')
    try:
        geometry = undistort_with_metadata(initial_frame, mtx, dist, calibration_size=calibration_size)
    except Exception:
        cap.release()
        raise
    output_size = geometry.image.shape[1::-1]
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, geometry.camera_matrix, output_size, cv2.CV_32FC1)

    print("Press 'q' to quit, 'd' to toggle distortion correction")

    # Flag to toggle distortion correction
    correct_distortion = True

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image")
                break

            if frame.shape[1::-1] != tuple(calibration_size):
                raise ValueError('Camera resolution changed while streaming')
            if correct_distortion:
                # Apply undistortion
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

                # Maps already use the shared cropped coordinate system; do not resize.

                # Add text to indicate undistorted view
                cv2.putText(undistorted, "Undistorted", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the undistorted frame
                cv2.imshow('Camera Feed', undistorted)
            else:
                # Add text to indicate original view
                cv2.putText(frame, "Original", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the original frame
                cv2.imshow('Camera Feed', frame)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            # 'q' to quit
            if key == ord('q'):
                break

            # 'd' to toggle distortion correction
            elif key == ord('d'):
                correct_distortion = not correct_distortion
                print(f"Distortion correction {'ON' if correct_distortion else 'OFF'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preview shared undistortion geometry')
    parser.add_argument('--calibration-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    live_undistortion(parser.parse_args().calibration_size)
