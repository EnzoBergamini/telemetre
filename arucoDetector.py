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

    def detect(self, intrinsics_matrix, distortion_coefficients, frame) -> tuple:
        """detection of aruco markers in the frame
        Args:
            intrinsics_matrix (numpy.ndarray): camera intrinsics matrix
            distortion_coefficients (numpy.ndarray): distortion coefficients
            frame (numpy.ndarray): frame from the camera
        """

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            frame_gray, self.aruco_dict, parameters=self.detector_params
        )

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.size_of_marker, intrinsics_matrix, distortion_coefficients
            )

            return corners, ids, rvecs, tvecs

        return None, None, None, None

    def draw(
        self,
        frame,
        corners,
        ids,
        rvecs,
        tvecs,
        intrinsics_matrix,
        distortion_coefficients,
    ) -> None:
        """draws the detected markers on the frame
        Args:
            frame (numpy.ndarray): frame from the camera
            corners (numpy.ndarray): corners of the detected markers
            ids (numpy.ndarray): ids of the detected markers
            rvecs (numpy.ndarray): rotation vectors of the detected markers
            tvecs (numpy.ndarray): translation vectors of the detected markers
            intrinsics_matrix (numpy.ndarray): camera intrinsics matrix
            distortion_coefficients (numpy.ndarray): distortion coefficients
        """

        if ids is not None:
            frame_marker = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            for rvec, tvec in zip(rvecs, tvecs):
                if tvec is not None:
                    for i in range(len(tvecs)):
                        frame_marker = aruco.drawAxis(
                            frame_marker,
                            intrinsics_matrix,
                            distortion_coefficients,
                            rvec[i],
                            tvec[i],
                            self.size_of_marker,
                        )
            if frame_marker is not None:
                return frame_marker
            else:
                return frame
        else:
            return frame
