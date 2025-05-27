# Reprojection validation:
from stereo_triangulator import StereoTriangulator
from scipy.io import savemat, loadmat
import numpy as np
import os
import glob
import cv2

data_path = 'stereo_calibration.npz'
init_z = 10000
stereo = StereoTriangulator(data_path, init_z=init_z)
print(stereo.K2)