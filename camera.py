import cv2
import depthai as dai
import numpy as np


class camera:
    def __init__(self) -> None:

        self.pipeline = dai.Pipeline()
        self.setup_camera_A()
        print("Camera A setup")
        self.setup_camera_B()
        print("Camera B setup")
        self.setup_camera_C()
        print("Camera C setup")
        self.setup_stereo()
        print("Stereo setup")
        self.setup_location()
        print("Location setup")
        self.setup_links()
        print("Links setup")

    def setup_camera_A(
        self,
        fps=15,
        dai_resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    ) -> None:  # rgb camera

        self.rgb = self.pipeline.createColorCamera()

        self.rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.rgb.setResolution(dai_resolution)
        self.rgb.setInterleaved(False)
        self.rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.rgb.setFps(fps)

    def setup_camera_B(
        self, dai_resolution=dai.MonoCameraProperties.SensorResolution.THE_480_P
    ) -> None:  # right camera rgb

        self.left = self.pipeline.createMonoCamera()

        self.left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        self.left.setResolution(dai_resolution)

    def setup_camera_C(self) -> None:  # left camera rgb

        self.right = self.pipeline.createMonoCamera()

        self.right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        self.right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    def setup_stereo(self) -> None:

        self.stereo = self.pipeline.createStereoDepth()
        self.stereo.setConfidenceThreshold(200)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)

    def setup_location(self) -> None:

        self.location = self.pipeline.createSpatialLocationCalculator()
        self.location_config = dai.SpatialLocationCalculatorConfigData()
        self.location_config.depthThresholds.lowerThreshold = 100
        self.location_config.depthThresholds.upperThreshold = 10000
        self.location_algorigthm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        self.location_config.roi = dai.Rect(
            dai.Point2f(0.48, 0.48), dai.Point2f(0.52, 0.52)
        )

        self.location.inputConfig.setWaitForMessage(False)
        self.location.initialConfig.addROI(self.location_config)

    def setup_links(self) -> None:

        # link output
        self.linkOut = {
            "rgb": self.pipeline.createXLinkOut(),
            "stereo": self.pipeline.createXLinkOut(),
            "location": self.pipeline.createXLinkOut(),
        }

        self.linkOut["rgb"].setStreamName("video")
        self.linkOut["stereo"].setStreamName("depth")
        self.linkOut["location"].setStreamName("location")

        # link input
        self.linkIn = {"locationConfig": self.pipeline.createXLinkIn()}

        self.linkIn["locationConfig"].setStreamName("locationConfig")

        # link the nodes

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.location.out.link(self.linkOut["location"].input)

        self.rgb.video.link(self.linkOut["rgb"].input)

        self.stereo.depth.link(self.location.inputDepth)
        self.location.passthroughDepth.link(self.linkOut["stereo"].input)

        self.linkIn["locationConfig"].out.link(self.location.inputConfig)

    def start(self) -> None:

        self.device = dai.Device(self.pipeline)
        self.device.startPipeline()

    def setup_calib_parameters(
        self, width_A=1920, height_A=1080, width_BC=640, height_BC=480
    ) -> None:

        self.calib_data = self.device.readCalibration()

        self.intrinsics_matrix_rgb = np.array(
            self.calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, width_A, height_A
            )
        )

        self.intrinsics_matrix_right = np.array(
            self.calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_C, width_BC, height_BC
            )
        )

        self.distortion_coefficients_rgb = np.array(
            self.calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        )

        self.extrinsic_matrix_right = np.array(
            self.calib_data.getCameraExtrinsics(
                dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_C
            )
        )
