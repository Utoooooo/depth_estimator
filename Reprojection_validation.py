# Reprojection validation:
from stereo_triangulator import StereoTriangulatorMatlab
from scipy.io import savemat, loadmat
import numpy as np
import os
import glob
import cv2



# Function to handle mouse clicks on the left window
def on_mouse_click_left(event, x, y, flags, param):
    global right_image  # To update the right image with the reprojected point
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the clicked point on the left image
        left_point = [x, y]

        # Reproject the point to the right image
        reprojected_point = stereo.reprojector(cam_pos='L', point=left_point)

        # Draw the clicked point on the left image
        left_with_indicator = left_image.copy()
        cv2.circle(left_with_indicator, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Left Window', left_with_indicator)

        # Draw the reprojected point on the right image
        right_with_indicator = right_image.copy()
        reprojected_x, reprojected_y = int(reprojected_point[0]), int(reprojected_point[1])
        print(f"Reprojected Point (Left to Right): {reprojected_point}")
        cv2.circle(right_with_indicator, (reprojected_x, reprojected_y), 5, (255, 0, 0), -1)
        cv2.imshow('Right Window', right_with_indicator)

# Function to handle mouse clicks on the right window
def on_mouse_click_right(event, x, y, flags, param):
    global left_image  # To update the left image with the reprojected point
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the clicked point on the right image
        right_point = [x, y]

        # Reproject the point to the left image
        reprojected_point = stereo.reprojector(cam_pos='R', point=right_point)
        est_p = stereo.triangulator(reprojected_point, right_point)

        # Draw the clicked point on the right image
        right_with_indicator = right_image.copy()
        cv2.circle(right_with_indicator, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Right Window', right_with_indicator)

        # Draw the reprojected point on the left image
        left_with_indicator = left_image.copy()
        reprojected_x, reprojected_y = int(reprojected_point[0]), int(reprojected_point[1])
        print(f"Reprojected Point (Right to Left): {reprojected_point}")
        cv2.circle(left_with_indicator, (reprojected_x, reprojected_y), 5, (255, 0, 0), -1)
        cv2.imshow('Left Window', left_with_indicator)

def load_latest_image(folder, extensions=("*.png", "*.jpg", "*.jpeg")):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    if not files:
        print("No images found in folder.")
        return None

    latest_file = max(files, key=os.path.getmtime)
    image = cv2.imread(latest_file)
    print(f"✅ Loaded: {latest_file}")
    return image

# data_path = 'stereo_calibration.npz'
# init_z = 10000
# stereo = StereoTriangulator(data_path, init_z=init_z)


data = loadmat('stereoParam_dual.mat')
init_z = 1000
stereo = StereoTriangulatorMatlab(data, init_z=init_z)

# Load images
left_image = load_latest_image("left")
right_image = load_latest_image("right")
# Display the images
cv2.imshow('Left Window', left_image)
cv2.imshow('Right Window', right_image)

# Set the mouse callbacks for both windows
cv2.setMouseCallback('Left Window', on_mouse_click_left)
cv2.setMouseCallback('Right Window', on_mouse_click_right)

# Wait for user interaction
cv2.waitKey(0)
cv2.destroyAllWindows()