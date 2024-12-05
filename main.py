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

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
	0.1651,  #tag size in meters
	543.93,  #Fx: x focal length
	544.98,  #Fy: y focal length
	316.29,  #Cx: x focal center (based on 640x480 resolution)
	250.55,  #Cy: y focal center (based on 640x480 resolution)
)

# Create the PoseEstimator
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)

aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

# Camera constants
xResolution = 640
yResolution = 480
frameRate = 30

# Creating the network tables
ntInstance = ntcore.NetworkTableInstance.getDefault()
ntInstance.startServer()

camera = CameraServer.startAutomaticCapture()

CameraServer.enableLogging()

table = ntInstance.getTable("AprilTag Vision")

# Export robot position
robotCenter = table.getDoubleArrayTopic("Robot Position").publish()
aprilTagPresence = table.getBooleanTopic("AprilTag Presence").publish()

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
    time, mat = cvSink.grabFrame(mat)
    grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
    detections = aprilTagDetector.detect(grayMat)


    if detections != []:
        for detection in detections:
            tagPose = aprilTagFieldLayout.getTagPose(detection.getId())
            if tagPose is not None:
                # Get field position, add to robotPose
                cameraToTag = poseEstimator.estimate(detection)
                flipTagRotation = Rotation3d(axis=(0, 1, 0), angle=rotationsToRadians(0.5))
                cameraToTag = Transform3d(cameraToTag.translation(), cameraToTag.rotation().rotateBy(flipTagRotation))
                cameraToTag = CoordinateSystem.convert(cameraToTag, CoordinateSystem.EDN(), CoordinateSystem.NWU())
                tagToRobot = cameraToTag.inverse()

                robotPose.append(tagPose.transformBy(tagToRobot))

            # Draw box around all AprilTags
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
                p2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
                mat = cv2.line(mat, p1, p2, lineColor, 2)

    # Set robotPos to the average position of all detections
    if robotPose != []:
        for pose in robotPose:
            avgPose = (avgPose[0] + pose.x, avgPose[1] + pose.y,avgPose[2] + pose.z)
        avgPose = (avgPose[0] / len(robotPose), avgPose[1] / len(robotPose), avgPose[2] / len(robotPose))
        robotPos = Pose3d(Translation3d(avgPose[0], avgPose[1], avgPose[2]), Rotation3d())

    # Set robotPos to the closest to the nearest pose to the last one
    """if robotPos is not Pose3d():
        robotPos = robotPos.nearest(robotPose)
    else:
        robotPos = random.choice(robotPose)
    """
    # Publish everything
    outputStream.putFrame(mat)
    robotCenter.set(list((round(robotPos.x, 6), round(robotPos.y, 6), round(robotPos.z, 6))))
    aprilTagPresence.set(detections != [])
