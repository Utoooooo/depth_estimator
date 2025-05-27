import numpy as np
import cv2

class StereoTriangulator:
    def __init__(self, npz_path, init_z=2000):
        # Load stereo calibration from .npz file
        calib = np.load(npz_path)

        self.K1 = calib['mtxL']
        self.K2 = calib['mtxR']

        # Distortion coefficients (assume OpenCV 5-element format)
        self.dist1 = calib['distL'].flatten()  # shape (5,)
        self.dist2 = calib['distR'].flatten()  # shape (5,)

        # Extrinsic parameters
        self.R = calib['R']
        self.T = calib['T'].reshape(3, 1)

        # Projection matrices
        self.P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.K2, np.hstack((self.R, self.T)))

        # Default Z initialization
        self.init_z = init_z


    def triangulator(self, leftPoint, rightPoint):
        leftPoint = np.asarray(leftPoint, dtype=np.float32).reshape(1, 1, 2)
        rightPoint = np.asarray(rightPoint, dtype=np.float32).reshape(1, 1, 2)

        undist_left = cv2.undistortPoints(leftPoint, self.K1, self.dist1, P=self.K1).reshape(-1, 2)
        undist_right = cv2.undistortPoints(rightPoint, self.K2, self.dist2, P=self.K2).reshape(-1, 2)

        P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K2 @ np.hstack((self.R.T, self.T))

        points4D = cv2.triangulatePoints(P1, P2, undist_left.T, undist_right.T)
        point3D = (points4D[:3] / points4D[3]).T[0]
        return point3D

    def reprojector(self, cam_pos: str, point: list):
        point = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)

        if cam_pos == 'L':
            undist = cv2.undistortPoints(point, self.K1, self.dist1, P=None)
            ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])
            pt_3d_left = ray / np.linalg.norm(ray) * self.init_z
            pt_3d_right = self.R.T @ (pt_3d_left.reshape(3, 1)+self.T)
            proj = self.K2 @ pt_3d_right
            proj /= proj[2]
            return proj[:2].flatten()

        elif cam_pos == 'R':
            undist = cv2.undistortPoints(point, self.K2, self.dist2, P=None)
            ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])
            pt_3d_right = ray / np.linalg.norm(ray) * self.init_z
            pt_3d_left = self.R @ (pt_3d_right.reshape(3, 1) - self.T)
            proj = self.K1 @ pt_3d_left
            proj /= proj[2]
            return proj[:2].flatten()

        else:
            raise ValueError("cam_pos must be 'left' or 'right'")

    def __repr__(self):
        return (f"<StereoTriangulator | Focal: ({self.K1[0,0]:.1f}, {self.K1[1,1]:.1f}), "
                f"Init Z: {self.init_z}>")
    

class StereoTriangulatorMatlab:
    def __init__(self, data, init_z=2000):
        # Load struct
        stereo_struct = data['s']

        # Extract camera parameters
        cam1 = stereo_struct['CameraParameters1'][0][0]
        cam2 = stereo_struct['CameraParameters2'][0][0]

        self.K1 = cam1['K'][0][0]
        self.K2 = cam2['K'][0][0]

        dist1 = cam1['RadialDistortion'][0][0].flatten()
        dist2 = cam2['RadialDistortion'][0][0].flatten()
        self.dist1 = np.concatenate((dist1, np.zeros(3)))  # shape (5,)
        self.dist2 = np.concatenate((dist2, np.zeros(3)))

        self.R = stereo_struct['RotationOfCamera2'][0][0]
        self.T = stereo_struct['TranslationOfCamera2'][0][0].reshape(3, 1)

        # Projection matrices
        self.P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        self.P2 = np.dot(self.K2, np.hstack((self.R, self.T)))

        # Default initial Z depth estimate
        self.init_z = init_z

    def triangulator(self, leftPoint, rightPoint):
        leftPoint = np.asarray(leftPoint, dtype=np.float32).reshape(1, 1, 2)
        rightPoint = np.asarray(rightPoint, dtype=np.float32).reshape(1, 1, 2)

        undist_left = cv2.undistortPoints(leftPoint, self.K1, self.dist1, P=self.K1).reshape(-1, 2)
        undist_right = cv2.undistortPoints(rightPoint, self.K2, self.dist2, P=self.K2).reshape(-1, 2)

        P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K2 @ np.hstack((self.R.T, self.T))

        points4D = cv2.triangulatePoints(P1, P2, undist_left.T, undist_right.T)
        point3D = (points4D[:3] / points4D[3]).T[0]
        return point3D

    def reprojector(self, cam_pos: str, point: list):
        point = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)

        if cam_pos == 'L':
            undist = cv2.undistortPoints(point, self.K1, self.dist1, P=None)
            ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])
            pt_3d_left = ray / np.linalg.norm(ray) * self.init_z
            pt_3d_right = self.R.T @ (pt_3d_left.reshape(3, 1)+self.T)
            proj = self.K2 @ pt_3d_right
            proj /= proj[2]
            return proj[:2].flatten()

        elif cam_pos == 'R':
            undist = cv2.undistortPoints(point, self.K2, self.dist2, P=None)
            ray = np.array([undist[0, 0, 0], undist[0, 0, 1], 1.0])
            pt_3d_right = ray / np.linalg.norm(ray) * self.init_z
            pt_3d_left = self.R @ (pt_3d_right.reshape(3, 1) - self.T)
            proj = self.K1 @ pt_3d_left
            proj /= proj[2]
            return proj[:2].flatten()

        else:
            raise ValueError("cam_pos must be 'left' or 'right'")

    def __repr__(self):
        return (f"<StereoTriangulator | Focal: ({self.K1[0,0]:.1f}, {self.K1[1,1]:.1f}), "
                f"Init Z: {self.init_z}>")
