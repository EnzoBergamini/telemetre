import cv2
import depthai as dai
import numpy as np
from cv2 import aruco


def detect_markers(frame, dist, mtx, aruco_dict, detector_parameters, size_of_marker):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        frame_gray, aruco_dict, parameters=detector_parameters
    )

    rvecs, tvecs, trash = aruco.estimatePoseSingleMarkers(
        corners, size_of_marker, mtx, dist
    )

    if ids is not None:
        return corners, ids, rvecs, tvecs
    else:
        return None, None, None, None


def draw_markers(frame, corners, ids, rvecs, tvecs, mtx, dist):
    if ids is not None:
        frame_marker = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            if tvec is not None:
                for i in range(len(tvec)):
                    frame_marker = aruco.drawAxis(
                        frame_marker, mtx, dist, rvec[i], tvec[i], 0.1
                    )

        if frame_marker is not None:
            return frame_marker
        else:
            return frame
    else:
        return frame


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define the sources

# Create a node for the left and right mono camera
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()

# Create a node for the stereo depth
stereo = pipeline.createStereoDepth()

# Create a node for the spatial location calculator
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

# Create a node for rgb camera
rgb = pipeline.createColorCamera()

# Create a node for the XLinkOut
xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xoutRgb = pipeline.createXLinkOut()

# create a node for the XLinkIn
xinSpatialCalcConfig = pipeline.createXLinkIn()

# Set the stream names for the XLink
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xoutRgb.setStreamName("rgb")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Set the resolution and board socket for the left and right mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Set the resolution and board socket for the rgb camera
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
rgb.setInterleaved(False)
rgb.setFps(15)

# Set the initial configuration for the stereo depth
stereo.setConfidenceThreshold(200)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(False)

# configure the spatial location calculator
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Link the nodes
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

rgb.video.link(xoutRgb.input)

# Aruco Déclaration

size_of_marker = 0.0444
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
detector_parameters = aruco.DetectorParameters_create()

with dai.Device(pipeline) as device:
    # Start the pipeline
    device.startPipeline()

    # Define the queue for the XLinkIn
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    # Define the queue for the XlinkOut
    spatialDataQueue = device.getOutputQueue(
        name="spatialData", maxSize=4, blocking=False
    )
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # Camera calibration
    calibData = device.readCalibration()
    mtx = np.array(
        calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1920, 1080)
    )
    dist = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

    while True:
        inDepth = depthQueue.tryGet()
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            inSpatialData = spatialDataQueue.tryGet()
            if inSpatialData is not None:
                spatialData = inSpatialData.getSpatialLocations()
                for depthData in spatialData:
                    roi = depthData.config.roi
                    roi = roi.denormalize(
                        depthFrameColor.shape[1], depthFrameColor.shape[0]
                    )
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)
                    cv2.rectangle(
                        depthFrameColor, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2
                    )
                    cv2.putText(
                        depthFrameColor,
                        f"Distance: {depthData.spatialCoordinates.z:.2f} mm",
                        (xmin, ymin - 20),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (255, 255, 255),
                    )
            cv2.imshow("depth", depthFrameColor)

        inRgb = rgbQueue.tryGet()
        if inRgb is not None:
            rgbFrame = inRgb.getCvFrame()
            (corners, ids, rvecs, tvecs) = detect_markers(
                rgbFrame, dist, mtx, aruco_dict, detector_parameters, size_of_marker
            )

            cv2.imshow(
                "rgb", draw_markers(rgbFrame, corners, ids, rvecs, tvecs, mtx, dist)
            )

        if cv2.waitKey(1) == ord("q"):
            break
