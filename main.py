import math
import random
from cscore import CameraServer
import ntcore
import numpy
import cv2
import robotpy_apriltag
from wpimath.units import rotationsToRadians
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, Translation3d, CoordinateSystem

# Loading the AprilTag data
aprilTagFieldLayout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

# Load camera data
Fx = 458.80  #Fx: x focal length
Fy = 461.62  #Fy: y focal length
Cx = 330.26  #Cx: x focal center (based on 640x480 resolution)
Cy = 258.17  #Cy: y focal center (based on 640x480 resolution)

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
	0.1651,  #tag size in meters
	Fx,
	Fy,
	Cx,
	Cy,
)

cameraDistortion = numpy.float32([ -0.077, 0.122, 0.002, 0.003, -0.121 ])

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
ntInstance.startServer()


table = ntInstance.getTable("AprilTag Vision")

# Export robot position
robotCenter = table.getDoubleArrayTopic("Robot Position").publish()
aprilTagPresence = table.getBooleanTopic("AprilTag Presence").publish()

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
lineColor = (0, 255, 0)

# Position of the robot relative to the camera
robotToCam = Transform3d(Translation3d(0, 0, 0),Rotation3d())

robotPos = Pose3d()

# Main loop
while True:
    robotPose = []
    avgPose = (0, 0, 0)

    _, mat = cvSink.grabFrame(mat)
    grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
    detections = aprilTagDetector.detect(grayMat)

    if detections != []:
        for detection in detections:
            
            tagPose = aprilTagFieldLayout.getTagPose(detection.getId())

            if tagPose is not None:

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

                # Run the OpenCV undistortion routine to fix the corners
                undistortedCorners = cv2.undistortImagePoints(distortedCorners, cameraIntrinsics, cameraDistortion)
                for i in range(4):
                    corners[2 * i] = undistortedCorners[i][0][0]
                    corners[2 * i + 1] = undistortedCorners[i][0][1]

                # Run the pose estimator using the fixed corners
                cameraToTag = poseEstimator.estimate(
                    homography = detection.getHomography(),
                    corners = tuple(corners))
                tagID = detection.getId()
                
                if tagPose is not None:
                    # First, we need to flip the Camera To Tag transform's angle 180 degrees around the y axis since the tag is oriented into the field
                    flipTagRotation = Rotation3d(axis = (0, 1, 0), angle = rotationsToRadians(0.5))
                    cameraToTag = Transform3d(cameraToTag.translation(), cameraToTag.rotation().rotateBy(flipTagRotation))

                    # The Camera To Tag transform is in a East/Down/North coordinate system, but we want it in the WPILib standard North/West/Up
                    cameraToTag = CoordinateSystem.convert(cameraToTag, CoordinateSystem.EDN(), CoordinateSystem.NWU())

                    # We now have a corrected transform from the camera to the tag. Apply the inverse transform to the tag pose to get the camera's pose
                    cameraPose = tagPose.transformBy(cameraToTag.inverse())

                    # Compute robot pose from robot to camera transform
                    robotPose.append(cameraPose.transformBy(robotToCam.inverse()))

    # Set robotPos to the average position of all detections
    if robotPose != []:
        for pose in robotPose:
            avgPose = (avgPose[0] + pose.x, avgPose[1] + pose.y,avgPose[2] + pose.z)
        avgPose = (avgPose[0] / len(robotPose), avgPose[1] / len(robotPose), avgPose[2] / len(robotPose))
        robotPos = Pose3d(Translation3d(avgPose[0], avgPose[1], avgPose[2]),Rotation3d())

    # Publish everything
    outputStream.putFrame(mat)
    robotCenter.set(list((round(robotPos.x, 6), round(robotPos.y, 6), -1 * round(robotPos.z, 6))))
    aprilTagPresence.set(detections != [])
