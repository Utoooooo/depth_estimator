import cv2

def test_stereo_camera(camera_index=1, width=1280, height=480):
    cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("✅ Press ESC to exit preview.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Stereo View", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Try running this
test_stereo_camera(width=1280, height=480)  # Adjust to your camera's SBS resolution
