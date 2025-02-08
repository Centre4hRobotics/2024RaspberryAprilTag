import json
from cscore import CameraServer
import ntcore
import numpy
import cv2
import robotpy_apriltag
from wpimath.units import rotationsToRadians
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, Translation3d, CoordinateSystem

#  Flags and Team Number
IS_TABLE_HOST = True
TEAM_NUMBER = 7204

# Loading the AprilTag data
aprilTagFieldLayout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

# Load camera calibration
with open('CameraCalibration.json') as json_data:
    data = json.load(json_data)

CAMERA_PROFILE = "640x480"

cameraData = data[CAMERA_PROFILE]

Fx = cameraData["Intrinsics"]["Fx"]
Fy = cameraData["Intrinsics"]["Fy"]
Cx = cameraData["Intrinsics"]["Cx"]
Cy = cameraData["Intrinsics"]["Cy"]

xResolution = cameraData["Resolution"]["x"]
yResolution = cameraData["Resolution"]["y"]
FRAME_RATE = 30

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
	0.1651,  #tag size in meters
	Fx,
	Fy,
	Cx,
	Cy,
)

cameraDistortion = numpy.float32([
    cameraData["Distortion"]["A"],
    cameraData["Distortion"]["B"],
    cameraData["Distortion"]["C"],
    cameraData["Distortion"]["D"],
    cameraData["Distortion"]["E"] ])
cameraIntrinsics = numpy.eye(3)
cameraIntrinsics[0][0] = Fx
cameraIntrinsics[1][1] = Fy
cameraIntrinsics[0][2] = Cx
cameraIntrinsics[1][2] = Cy

# Create the PoseEstimator & adjust its settings
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)

aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

aprilTagDetectorConfig = aprilTagDetector.getConfig()
aprilTagDetectorConfig.numThreads = 4
aprilTagDetectorConfig.quadSigma = 0.5
aprilTagDetectorConfig.quadDecimate = 1
aprilTagDetector.setConfig(aprilTagDetectorConfig)

quadThresholdParameters = aprilTagDetector.getQuadThresholdParameters()
quadThresholdParameters.minClusterPixels = 5
quadThresholdParameters.criticalAngle = 0.79
aprilTagDetector.setQuadThresholdParameters(quadThresholdParameters)


# Creating the network tables
ntInstance = ntcore.NetworkTableInstance.getDefault()

# Check network tables host flag
if IS_TABLE_HOST:
    ntInstance.startServer()

else:
    ntInstance.setServerTeam(TEAM_NUMBER)
    ntInstance.startClient4("visionPi")

table = ntInstance.getTable("AprilTag Vision")

# Export robot position
localPosX = table.getDoubleTopic("Pose X").publish()
localPosY = table.getDoubleTopic("Pose Y").publish()
localPosZ = table.getDoubleTopic("Pose Z").publish()
tagRotation = table.getDoubleTopic("Tag Rotation").publish()
aprilTagPresence = table.getBooleanTopic("AprilTag Presence").publish()
widestTag = table.getIntegerTopic("Widest Tag ID").publish()
cameraChoice = table.getStringTopic("Using Camera").publish()
cameraChoice.set("LEFT")
precameraString = table.getStringTopic("Using Camera")
cameraString = precameraString.subscribe("BAD")


# Activate camera stuff

CameraServer.enableLogging()

cameraLeft = CameraServer.startAutomaticCapture(0)

cameraRight = CameraServer.startAutomaticCapture(2)

with open('CameraConfig.json') as json_data:
    cameraConfigJson = json.load(json_data)

cameraLeft.setConfigJson(cameraConfigJson)
cameraRight.setConfigJson(cameraConfigJson)

cvSinkLeft = CameraServer.getVideo(cameraLeft)
cvSinkRight = CameraServer.getVideo(cameraRight)

outputStream = CameraServer.putVideo("Vision", xResolution, yResolution)

# Images
mat = numpy.zeros(shape=(xResolution, yResolution, 3), dtype=numpy.uint8)
grayMat = numpy.zeros(shape=(xResolution, yResolution), dtype=numpy.uint8)

# Colors for drawing
lineColor = (0,255,0)

# Position of the robot relative to the camera
robotToCam = Transform3d(Translation3d(0,0,0),Rotation3d())

robotPos = Pose3d()
bestPose = Pose3d()
bestTag = -1

reefTags = [17]

# Main loop
while True:
    maxTagWidth = 0.0

    if cameraString.get() == "LEFT":
        _, mat = cvSinkLeft.grabFrame(mat)

    else:
        _, mat = cvSinkRight.grabFrame(mat)

    grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

    detections = aprilTagDetector.detect(grayMat)

    if detections != []:
        for detection in detections:
            if False:
                continue
            tagPose = aprilTagFieldLayout.getTagPose(detection.getId())

            corners = list(detection.getCorners(numpy.empty(8)))

            # Outline the tag using original corners
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(corners[2 * i]),int(corners[2 * i + 1]))
                p2 = (int(corners[2 * j]),int(corners[2 * j + 1]))
                mat = cv2.line(mat, p1, p2, lineColor, 2)

            # Manually reshape 'corners'
            distortedCorners = numpy.empty([4,2], dtype=numpy.float32)
            for i in range(4):
                distortedCorners[i][0] = corners[2 * i]
                distortedCorners[i][1] = corners[2 * i + 1]

            # run the OpenCV undistortion routine to fix the corners
            undistortedCorners = cv2.undistortImagePoints(distortedCorners, cameraIntrinsics, cameraDistortion)
            for i in range(4):
                corners[2 * i] = undistortedCorners[i][0][0]
                corners[2 * i + 1] = undistortedCorners[i][0][1]

            # find the widest tag
            if corners[2]-corners[0] > maxTagWidth:
                maxTagWidth = corners[2]-corners[0]
                bestTag = detection.getId()

            # run the pose estimator using the fixed corners
            cameraToTag = poseEstimator.estimate(
                homography = detection.getHomography(),
                corners = tuple(corners))
            tagID = detection.getId()

            # first we need to flip the Camera To Tag transform's angle 180 degrees around the y axis since the tag is oriented into the field
            flipTagRotation = Rotation3d(axis = (0, 1, 0), angle = rotationsToRadians(0.5))
            cameraToTag = Transform3d(cameraToTag.translation(), cameraToTag.rotation().rotateBy(flipTagRotation))

            # The Camera To Tag transform is in a East/Down/North coordinate system, but we want it in the WPILib standard North/West/Up
            cameraToTag = CoordinateSystem.convert(cameraToTag, CoordinateSystem.EDN(), CoordinateSystem.NWU())

            robotToTag = robotToCam + cameraToTag

            if detection.getId() == bestTag:
                bestPose = robotToTag

    # Publish everything
    outputStream.putFrame(mat)
    tagRotation.set(bestPose.rotation().z)
    localPosX.set(bestPose.x)
    localPosY.set(bestPose.y)
    localPosZ.set(bestPose.z)

    aprilTagPresence.set(detections != [])
    widestTag.set(bestTag)
