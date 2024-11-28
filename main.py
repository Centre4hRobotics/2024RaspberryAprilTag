import math
import random
from cscore import CameraServer # type: ignore
import ntcore # type: ignore
import numpy # type: ignore
import cv2 # type: ignore
import robotpy_apriltag # type: ignore
import wpimath # type: ignore

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
lineColor = (0,255,0)

# Position of the robot relative to the camera
robotCam = wpimath.geometry.Transform3d(wpimath.geometry.Translation3d(0,0,1),wpimath.geometry.Rotation3d())

# Robot position
robotPos = wpimath.geometry.Pose3d()

# Main loop
while True:
    cameraPose = []

    time, mat = cvSink.grabFrame(mat)
    grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)
    detections = aprilTagDetector.detect(grayMat)

    if (detections != []):
        for detection in detections:
            tagPose = aprilTagFieldLayout.getTagPose(detection.getId())
            if tagPose is not None:
                # Get field position, add to cameraPose
                transform = poseEstimator.estimate(detection)
                transformPose = wpimath.geometry.Pose3d(transform.translation(), transform.rotation())
                temp = transformPose.transformBy(robotCam)
                cameraPose.append(tagPose.transformBy(wpimath.geometry.Transform3d(temp.translation(), temp.rotation())))

            # Draw box around all AprilTags
            for i in range(4):
                j = (i + 1) % 4
                p1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
                p2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
                mat = cv2.line(mat, p1, p2, lineColor, 2)

    # Set position to a random pose from cameraPose, if cameraPose is empty don't
    if cameraPose != []:
        robotPos = random.choice(cameraPose)

    # Publish everything
    outputStream.putFrame(mat)
    robotCenter.set(list((round(robotPos.x, 4), round(robotPos.y, 4), round(robotPos.z, 4))))
    aprilTagPresence.set(detections != [])
