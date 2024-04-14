import cv2
import depthai as dai
from cv2 import aruco


class arucoDetector:

    def __init__(
        self,
        size_of_marker=0.05,
        aruco_dict=aruco.Dictionary_get(aruco.DICT_4X4_1000),
        detector_params=aruco.DetectorParameters_create(),
    ) -> None:

        self.size_of_marker = size_of_marker
        self.aruco_dict = aruco_dict
        self.detector_params = detector_params

    def detect(self, intrinsics_matrix, distortion_coefficients, frame):
        return corners, ids, rejectedImgPoints
