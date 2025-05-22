import cv2
import numpy as np
import os
import time
from datetime import datetime


def create_output_dirs(left_dir="left", right_dir="right"):
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    return left_dir, right_dir


def capture_stereo_images(num_images=10, interval=5, camera_index=1, frame_width=1600, frame_height=600):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not found!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    half_width = frame_width // 2

    left_dir, right_dir = create_output_dirs()

    print(f"Starting capture of {num_images} stereo images every {interval}s...")
    for i in range(num_images):
        last_capture_time = time.time()
        countdown = interval
        while countdown > 0:
            ret, frame = cap.read()
            elapsed = time.time() - last_capture_time
            countdown = int(interval - elapsed)
            if not ret:
                print("Failed to read frame.")
                continue

            text = f"Capturing in {countdown}s..."
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Stereo Capture", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                return

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            continue

        left_img = frame[:, :half_width]
        right_img = frame[:, half_width:]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_path = os.path.join(left_dir, f"left_{timestamp}.png")
        right_path = os.path.join(right_dir, f"right_{timestamp}.png")

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)
        print(f"[{i+1}/{num_images}] Saved {left_path} and {right_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Image capture complete.")


def calibrate_stereo_cameras(left_dir="left", right_dir="right",
                              chessboard_size=(11, 8), square_size=0.012):
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    left_images = sorted([os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.endswith(".png")])
    right_images = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.endswith(".png")])

    for left_path, right_path in zip(left_images, right_images):
        imgL = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        retL, cornersL = cv2.findChessboardCorners(imgL, chessboard_size)
        retR, cornersR = cv2.findChessboardCorners(imgR, chessboard_size)

        if retL and retR:
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

    print(f"Found {len(objpoints)} valid image pairs with detected corners.")

    if len(objpoints) < 5:
        print("Not enough valid pairs for calibration.")
        return

    retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, imgL.shape[::-1], None, None)
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, imgR.shape[::-1], None, None)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6)

    retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR,
        imgL.shape[::-1],
        criteria=criteria,
        flags=flags
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR,
        imgL.shape[::-1],
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    np.savez("stereo_calibration.npz",
             mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    print("✅ Stereo calibration complete. Parameters saved to stereo_calibration.npz.")


def main():
    capture_stereo_images(num_images=10, interval=5)
    calibrate_stereo_cameras()


if __name__ == "__main__":
    main()
