import math
import random
from cscore import CameraServer
import ntcore
import numpy
import cv2
import robotpy_apriltag
from wpimath.units import rotationsToRadians
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, Translation3d, CoordinateSystem

# Flags and Team Number
isTableHost = False
teamNumber = 7204

# Loading the AprilTag data
aprilTagFieldLayout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

# Load camera data
Fx = 560.75
Fy = 566.44
Cx = 299.65
Cy = 262.37

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
	0.1651,  #tag size in meters
	Fx,
	Fy,
	Cx,
	Cy,
)

cameraDistortion = numpy.float32([ 0.04, -0.101, 0.005, 0.001, 0.069 ])
cameraIntrinsics = numpy.eye(3)
cameraIntrinsics[0][0] = Fx
cameraIntrinsics[1][1] = Fy
cameraIntrinsics[0][2] = Cx
cameraIntrinsics[1][2] = Cy

# Create the PoseEstimator
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)

aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

# Creating the network tables
ntInstance = ntcore.NetworkTableInstance.getDefault()

# Check network tables host flag
if isTableHost:
    ntInstance.startServer()
else:
    ntInstance.setServerTeam(teamNumber)
    ntInstance.startClient4("visionPi")

table = ntInstance.getTable("AprilTag Vision")

# Export robot position
#robotCenter = table.getDoubleArrayTopic("Robot Global Position").publish()
#localPos = table.getDoubleArrayTopic("Localized Pose").publish()
localPosX = table.getDoubleTopic("Pose X").publish()
localPosY = table.getDoubleTopic("Pose Y").publish()
localPosZ = table.getDoubleTopic("Pose Z").publish()
tagRotation = table.getDoubleTopic("Tag Rotation").publish()
aprilTagPresence = table.getBooleanTopic("AprilTag Presence").publish()
widestTag = table.getIntegerTopic("Widest Tag ID").publish()
#allTags = table.getIntegerArrayTopic("All detected tags").publish()

# Camera constants
xResolution = 640
yResolution = 480
frameRate = 30

# Activate camera stuff
camera = CameraServer.startAutomaticCapture()
CameraServer.enableLogging()

camera.setResolution(xResolution, yResolution)
camera.setFPS(frameRate)
cvSink = CameraServer.getVideo()
outputStream = CameraServer.putVideo("Vision", xResolution, yResolution)

# Images
mat = numpy.zeros(shape=(xResolution, yResolution, 3), dtype=numpy.uint8)
grayMat = numpy.zeros(shape=(xResolution, yResolution), dtype=numpy.uint8)

# Colors for drawing
lineColor = (0,255,0)

# Position of the robot relative to the camera
robotToCam = Transform3d(Translation3d(-0.255,0,0),Rotation3d())

robotPos = Pose3d()
bestPose = Pose3d()
bestTag = -1


# Main loop
while True:
    maxTagWidth = 0.0
    robotPose = []
    avgPose = (0,0,0)

    _, mat = cvSink.grabFrame(mat)
    grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
    detections = aprilTagDetector.detect(grayMat)

    if detections != []:
        for detection in detections:

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

            if tagPose is not None:
                 # We now have a corrected transform from the camera to the tag. Apply the inverse transform to the tag pose to get the camera's pose
                cameraPose = tagPose.transformBy(robotToTag.inverse())

                # compute robot pose from robot to camera transform
                robotPose.append(cameraPose)


    # Set robotPos to the average position of all detections
    if robotPose != []:
        for pose in robotPose:
            avgPose = (avgPose[0] + pose.x, avgPose[1] + pose.y,avgPose[2] + pose.z)
        avgPose = (avgPose[0] / len(robotPose), avgPose[1] / len(robotPose), avgPose[2] / len(robotPose))
        robotPos = Pose3d(Translation3d(avgPose[0],avgPose[1],avgPose[2]),Rotation3d())

    # Publish everything
    outputStream.putFrame(mat)
    #robotCenter.set(list((round(robotPos.x,6),round(robotPos.y,6),-1 * round(robotPos.z,6))))
    #localPos.set(list((round(bestPose.x,6),round(bestPose.y,6),-1 * round(bestPose.z,6))))
    tagRotation.set(bestPose.rotation().z)
    localPosX.set(bestPose.x)
    localPosY.set(bestPose.y)
    localPosZ.set(bestPose.z)

    aprilTagPresence.set(detections != [])
    widestTag.set(bestTag)
