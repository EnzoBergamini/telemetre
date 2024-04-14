import cv2
import depthai as dai


class camera:
    def __init__(self) -> None:

        self.pipeline = dai.Pipeline()
        self.setup_camera_A()
        self.setup_camera_B()
        self.setup_camera_C()
        self.setup_links()

    def setup_camera_A(
        self,
        fps=15,
        dai_resolution=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    ) -> None:  # rgb camera

        self.rgb = self.pipeline.createColorCamera()

        self.rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
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

        self.right.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        self.right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    def setup_stereo(self) -> None:

        self.stereo = self.pipeline.createStereoDepth()
        self.depstereoth.setConfidenceThreshold(200)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)

    def setup_location(self) -> None:

        self.location = self.pipeline.createSpatialLocationCalculator()
        self.location_config = dai.SpatialLocationCalculatorConfigData()
        self.location_config.depthThresholds.lowerThreshold = 100
        self.location_config.depthThresholds.upperThreshold = 10000
        self.location_config.roi = dai.Rect(
            dai.Point2f(0.48, 0.48), dai.Point2f(0.52, 0.52)
        )

        self.location.setWaitForConfigInput(False)
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

        self.rgb.video.link(self.linkOut["rgb"].input)

        self.stereo.depth.link(self.linkOut["location"].input)
        self.stereo.passThrough.link(self.linkOut["stereo"].input)

        self.linkIn["locationConfig"].out.link(self.location.inputConfig)
