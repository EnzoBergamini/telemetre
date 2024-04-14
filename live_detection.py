import cv2
import depthai as dai
import numpy as np
from cv2 import aruco
import math
import matplotlib.pyplot as plt


def matching_point(P, mtx_intra, mtx_extra):
    print("P: ", P)
    translation = mtx_extra[:3, 3] / 100
    rotation = mtx_extra[:3, :3]

    P_new = np.dot(rotation, P) + translation

    p_new = np.dot(mtx_intra, P_new)
    p_new = p_new / p_new[2]

    return p_new


def detect_markers(frame, dist, mtx, aruco_dict, detector_parameters, size_of_marker):

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = aruco.detectMarkers(
        frame_gray, aruco_dict, parameters=detector_parameters
    )

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
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
                        frame_marker, mtx, dist, rvec[i], tvec[i], 0.05
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
rgbOut = pipeline.createXLinkOut()
depthOut = pipeline.createXLinkOut()
spatialOut = pipeline.createXLinkOut()


# create a node for the XLinkIn
xinSpatialCalcConfig = pipeline.createXLinkIn()

# Set the stream names for the XLink
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
rgbOut.setStreamName("video")
depthOut.setStreamName("depth")
spatialOut.setStreamName("location")

# Set the resolution and board socket for the left and right mono cameras
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
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
topLeft = dai.Point2f(0.48, 0.48)
bottomRight = dai.Point2f(0.52, 0.52)

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


stereo.depth.link(
    spatialLocationCalculator.inputDepth
)  # carry out the depth calculation to the spatial location calculator
spatialLocationCalculator.passthroughDepth.link(
    depthOut.input
)  # pass the depth calculation to the sync node

spatialLocationCalculator.out.link(spatialOut.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

rgb.video.link(rgbOut.input)

# Aruco Déclaration

size_of_marker = 0.05
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
detector_parameters = aruco.DetectorParameters_create()

with dai.Device(pipeline) as device:
    newConfig = False

    # Start the pipeline
    device.startPipeline()

    # Define the queue for the XLinkIn
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    # Define the queue for the XlinkOut
    video = device.getOutputQueue("video", maxSize=4, blocking=False)
    depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    location = device.getOutputQueue("location", maxSize=4, blocking=False)

    # Camera calibration
    calibData = device.readCalibration()

    # RGB Camera calibration
    mtx = np.array(
        calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1920, 1080)
    )

    dist = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

    # right camera calibration
    mtx_extra_right = np.array(
        calibData.getCameraExtrinsics(
            dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_C
        )
    )

    mtx_intra_right = np.array(
        calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, 640, 480)
    )

    # left camera calibration
    mtx_intra_left = np.array(
        calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)
    )

    mtx_extra_left = np.array(
        calibData.getCameraExtrinsics(
            dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B
        )
    )

    p = [100, 100, 0]

    # setup scatter plot

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plot = ax.scatter([], [], [])

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    while True:
        inRgb = video.get()
        if inRgb is not None:
            rgbFrame = inRgb.getCvFrame()

            (corners, ids, rvecs, tvecs) = detect_markers(
                rgbFrame, dist, mtx, aruco_dict, detector_parameters, size_of_marker
            )

            if ids is not None:
                p = matching_point(tvecs[0][0], mtx_intra_right, mtx_extra_right)
                topLeft = dai.Point2f((p[0] - 5) / 640, (p[1] - 5) / 480)
                bottomRight = dai.Point2f((p[0] + 5) / 640, (p[1] + 5) / 480)
                config.roi = dai.Rect(topLeft, bottomRight)
                config.calculationAlgorithm = calculationAlgorithm
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)

        inDepth = depth.get()
        if inDepth is not None:
            depthFrame = inDepth.getFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            inSpatialData = location.get()
            if inSpatialData is not None:
                spatialData = inSpatialData.getSpatialLocations()
                for depthData in spatialData:
                    cv2.rectangle(
                        depthFrameColor,
                        (int(p[0] - 5), int(p[1] - 5)),
                        (int(p[0] + 5), int(p[1] + 5)),
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        depthFrameColor,
                        f"z: {depthData.spatialCoordinates.z:.2f} mm",
                        (int(p[0]), int(p[1] - 20)),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (255, 255, 255),
                    )

                    if corners is not None:

                        array = np.array(plot._offsets3d).T

                        new_array = np.vstack(
                            (
                                array,
                                [
                                    depthData.spatialCoordinates.x,
                                    depthData.spatialCoordinates.y,
                                    depthData.spatialCoordinates.z,
                                ],
                            )
                        )

                        plot._offsets3d = (
                            new_array[:, 0],
                            new_array[:, 1],
                            new_array[:, 2],
                        )

                        ax.set_xlim(
                            new_array[:, 0].min() - 100, new_array[:, 0].max() + 100
                        )
                        ax.set_ylim(
                            new_array[:, 1].min() - 100, new_array[:, 1].max() + 100
                        )
                        ax.set_zlim(
                            new_array[:, 2].min() - 100, new_array[:, 2].max() + 100
                        )

                        fig.canvas.draw()

                        cv2.putText(
                            rgbFrame,
                            f"x: {depthData.spatialCoordinates.x:.2f} mm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1])),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1,
                            (255, 255, 255),
                        )

                        cv2.putText(
                            rgbFrame,
                            f"y: {depthData.spatialCoordinates.y:.2f} mm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) + 30),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1,
                            (255, 255, 255),
                        )

                        cv2.putText(
                            rgbFrame,
                            f"z: {depthData.spatialCoordinates.z:.2f} mm",
                            (int(corners[0][0][0][0]), int(corners[0][0][0][1]) + 60),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1,
                            (255, 255, 255),
                        )

            cv2.imshow(
                "rgb", draw_markers(rgbFrame, corners, ids, rvecs, tvecs, mtx, dist)
            )

            cv2.imshow("depth", depthFrameColor)

        if cv2.waitKey(1) == ord("q"):
            break
