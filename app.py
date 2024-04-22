from camera import camera
from arucoDetector import arucoDetector
import cv2
import numpy as np
import depthai as dai


class app:
    def __init__(
        self, width_rgb=1920, height_rgb=1080, width_mono=640, height_mono=480
    ) -> None:
        print("App started")
        self.camera = camera()
        print("Camera initialized")
        self.aruco_detector = arucoDetector()
        print("Aruco detector initialized")

        self.widht_rgb = width_rgb
        self.height_rgb = height_rgb

        self.width_mono = width_mono
        self.height_mono = width_mono

        self.coordinate = []

    def match_point(self, P, intrinsic_matrix, extrinsic_matrix):
        """matches the 2d point to 3d point
        Args:
            P (numpy.ndarray): 2d point
            intrinsic_matrix (numpy.ndarray): camera intrinsics matrix
            extrinsic_matrix (numpy.ndarray): camera extrinsics matrix
        """
        translation_vector = extrinsic_matrix[:3, 3] / 100
        rotation_vector = extrinsic_matrix[:3, :3]

        P_new = np.dot(rotation_vector, P) + translation_vector

        p = np.dot(intrinsic_matrix, P_new)
        p = p / p[2]

        return p

    def run(self):
        self.camera.start()

        location_config_queue = self.camera.device.getInputQueue("locationConfig")

        video_queue = self.camera.device.getOutputQueue(
            "video", maxSize=4, blocking=False
        )
        depth_queue = self.camera.device.getOutputQueue(
            "depth", maxSize=4, blocking=False
        )
        location_data_queue = self.camera.device.getOutputQueue(
            "location", maxSize=4, blocking=False
        )

        self.camera.setup_calib_parameters()

        while True:
            in_rgb = video_queue.get()
            if in_rgb is not None:
                rgb_frame = in_rgb.getCvFrame()

                (corners, ids, rvecs, tvecs) = self.aruco_detector.detect(
                    self.camera.intrinsics_matrix_rgb,
                    self.camera.distortion_coefficients_rgb,
                    rgb_frame,
                )

                if ids is not None:
                    point = self.match_point(
                        tvecs[0][0],
                        self.camera.intrinsics_matrix_rgb,
                        self.camera.extrinsic_matrix_right,
                    )

                    top_left = dai.Point2f(
                        (point[0] - 5) / self.widht_rgb,
                        (point[1] - 5) / self.height_rgb,
                    )

                    bottom_right = dai.Point2f(
                        (point[0] + 5) / self.widht_rgb,
                        (point[1] + 5) / self.height_rgb,
                    )

                    self.camera.location_config.roi = dai.Rect(top_left, bottom_right)
                    self.camera.location_config.calculationAlgorithm = (
                        dai.SpatialLocationCalculatorAlgorithm.MEDIAN
                    )

                    cfg = dai.SpatialLocationCalculatorConfig()
                    cfg.addROI(self.camera.location_config)

                    location_config_queue.send(cfg)

            in_depth = depth_queue.get()
            if in_depth is not None:
                depth_frame = in_depth.getFrame()
                depth_frame_color = cv2.normalize(
                    depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
                )
                depth_frame_color = cv2.equalizeHist(depth_frame_color)
                depth_frame_color = cv2.applyColorMap(
                    depth_frame_color, cv2.COLORMAP_JET
                )

            in_location = location_data_queue.get()
            if in_location is not None:
                location_data = in_location.getSpatialLocations()

                for spatial_location_data in location_data:
                    roi = spatial_location_data.config.roi
                    roi = roi.denormalize(self.widht_rgb, self.height_rgb)

                    if corners is not None:

                        self.coordinate.append(
                            (
                                spatial_location_data.spatialCoordinates.x,
                                spatial_location_data.spatialCoordinates.y,
                                spatial_location_data.spatialCoordinates.z,
                            )
                        )

                        cv2.putText(
                            rgb_frame,
                            f"X: {spatial_location_data.spatialCoordinates.x:.2f} cm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                        cv2.putText(
                            rgb_frame,
                            f"Y: {spatial_location_data.spatialCoordinates.y:.2f} cm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                        cv2.putText(
                            rgb_frame,
                            f"Z: {spatial_location_data.spatialCoordinates.z:.2f} cm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
            cv2.imshow(
                "rgb",
                self.aruco_detector.draw(
                    rgb_frame,
                    corners,
                    ids,
                    rvecs,
                    tvecs,
                    self.camera.intrinsics_matrix_rgb,
                    self.camera.distortion_coefficients_rgb,
                ),
            )
            cv2.imshow("depth", depth_frame_color)

            if cv2.waitKey(1) == ord("q"):
                print(self.coordinate)
                break
