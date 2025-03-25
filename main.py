""" This is the main file for FRC Team 4027's 2024 AprilTag Vision. """

import subprocess
import json
import math
import ntcore
import numpy
import cv2
import robotpy_apriltag
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, CoordinateSystem
from cscore import CameraServer


#  All settings

IS_TABLE_HOST = False
print("Is table host: " + IS_TABLE_HOST)

TEAM_NUMBER = 4027
print("Team number " + TEAM_NUMBER)

FRAME_RATE = 30
CAMERA_PROFILE = "640x480"
print("Using camera profile " + CAMERA_PROFILE + " at " + FRAME_RATE + "FPS")


# Loading the AprilTag data
april_tag_field_layout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")
print("Loaded field layout")


# Load camera calibration
with open('CameraCalibration.json', encoding="utf-8") as json_data:
    calibration_data = json.load(json_data)
    print("Opened CameraCalibration.json")

camera_data = calibration_data[CAMERA_PROFILE]

Fx = camera_data["Intrinsics"]["Fx"]
Fy = camera_data["Intrinsics"]["Fy"]
Cx = camera_data["Intrinsics"]["Cx"]
Cy = camera_data["Intrinsics"]["Cy"]

x_resolution = camera_data["Resolution"]["x"]
y_resolution = camera_data["Resolution"]["y"]


pose_estimator_config = robotpy_apriltag.AprilTagPoseEstimator.Config(
	0.1651,  #tag size in meters
	Fx,
	Fy,
	Cx,
	Cy,
)

camera_distortion = numpy.float32([
    camera_data["Distortion"]["A"],
    camera_data["Distortion"]["B"],
    camera_data["Distortion"]["C"],
    camera_data["Distortion"]["D"],
    camera_data["Distortion"]["E"] ])
camera_intrinsics = numpy.eye(3)
camera_intrinsics[0][0] = Fx
camera_intrinsics[1][1] = Fy
camera_intrinsics[0][2] = Cx
camera_intrinsics[1][2] = Cy

print("Set camera calibration data")


# Create the april_tag_detector & adjust its settings
pose_estimator = robotpy_apriltag.AprilTagPoseEstimator(pose_estimator_config)

april_tag_detector = robotpy_apriltag.AprilTagDetector()
april_tag_detector.addFamily("tag36h11", 3)

april_tag_detector_config = april_tag_detector.getConfig()
april_tag_detector_config.numThreads = 4
april_tag_detector_config.quadSigma = 0.5
april_tag_detector_config.quadDecimate = 1
april_tag_detector.setConfig(april_tag_detector_config)

quad_threshold_parameters = april_tag_detector.getQuadThresholdParameters()
quad_threshold_parameters.minClusterPixels = 5
quad_threshold_parameters.criticalAngle = 0.79
april_tag_detector.setQuadThresholdParameters(quad_threshold_parameters)

print("Created pose_estimator and april_tag_detector")


# Creating the network tables
ntInstance = ntcore.NetworkTableInstance.getDefault()

# Check network tables host flag
if IS_TABLE_HOST:
    ntInstance.startServer()
    print("Started network table server")

else:
    ntInstance.setServerTeam(TEAM_NUMBER)
    ntInstance.startClient4("visionPi")
    print("Started network table client")

table = ntInstance.getTable("AprilTag Vision")


# Export robot position

# Global position of the robot
robot_x = table.getDoubleTopic("Global X").publish()
robot_y = table.getDoubleTopic("Global Y").publish()
#robot_z = table.getDoubleTopic("Global Z").publish()

# Tag to camera transform (this is more useful than the raw pose)
tag_to_camera_x = table.getDoubleTopic("tag_to_camera X").publish()
tag_to_camera_y = table.getDoubleTopic("tag_to_camera Y").publish()
#tag_to_camera_z = table.getDoubleTopic("tag_to_camera Z").publish()
tag_to_camera_theta = table.getDoubleTopic("tag_to_camera Theta").publish()

# Location of the tag on video feed (No pose estimation)
# Returns values between -1 and 1
tag_center_x = table.getDoubleTopic("Tag Center X").publish()
#tag_center_Y = table.getDoubleTopic("Tag Center Y").publish()

# Returns all visible Tags
all_tags_topic = table.getIntegerArrayTopic("All Tags").publish()

# Returns whether we have a tag
apriltag_presence = table.getBooleanTopic("AprilTag Presence").publish()

# Returns the best tag visible/which tag is being reported on
best_tag_id_topic = table.getIntegerTopic("Best Tag ID").publish()

# Tells which camera is being used. Can also be changed by others
camera_choice = table.getStringTopic("Using Camera").publish()
camera_choice.set("LEFT")

camera_string = table.getStringTopic("Using Camera").subscribe("NO TABLE FOUND")

tag_choice_topic = table.getIntegerTopic("Tag Choice").publish()
tag_choice_topic.set(0)

tag_choice = table.getIntegerTopic("Tag Choice").subscribe(0)

# Activate camera stuff

CameraServer.enableLogging()

left_camera = CameraServer.startAutomaticCapture(2)

right_camera = CameraServer.startAutomaticCapture(0)

left_camera.setResolution(x_resolution, y_resolution)
right_camera.setResolution(x_resolution, y_resolution)

cv_sink_left = CameraServer.getVideo(left_camera)
cv_sink_right = CameraServer.getVideo(right_camera)

outputStream = CameraServer.putVideo("Vision", x_resolution, y_resolution)

rc = subprocess.call("chmod u+rx set_camera_settings.sh "
+ "&& /home/pi/2024RaspberryAprilTag/set_camera_settings.sh", shell = True)
print("set_camera_settings.sh returned: ", rc)

# Images
mat = numpy.zeros(shape=(x_resolution, y_resolution, 3), dtype=numpy.uint8)
gray_mat = numpy.zeros(shape=(x_resolution, y_resolution), dtype=numpy.uint8)

# Colors for drawing
line_color = (0,255,0)
best_color = (255,255,0)

# Etc.
robot_pose = Pose3d()
#robot_to_cam = [Transform3d(), Transform3d()]
best_tag_to_camera = Transform3d()
best_tag_center_x = 0
best_tag = -1
theta = 0
reef_tags = [6,7,8,9,10,11,17,18,19,20,21,22]

# Main loop
while True:
    all_tags = []
    min_tag_x = x_resolution + 1
    has_tag = False

    if camera_string.get() == "LEFT":
        # grabFrame returns two values, the first of which we don't care about
        _, mat = cv_sink_left.grabFrame(mat)

    else:
        _, mat = cv_sink_right.grabFrame(mat)

    # Rotate video to not be upside down/on it's side
    mat = cv2.rotate(mat, cv2.ROTATE_180)

    # Convert the video to grayscale
    gray_mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

    # Detect AprilTags
    detections = april_tag_detector.detect(gray_mat)

    for detection in detections:

        all_tags.append(detection.getID)

        if tag_choice.get() > 0 and not detection.getId() is tag_choice.get():
            continue

        # Ignore tags not in reef_tags
        if detection.getId() not in reef_tags:
            continue # Move on to the next detection or exit the for loop

        corners = list(detection.getCorners(numpy.empty(8)))

        # Outline the tag using original corners
        for i in range(4):
            j = (i + 1) % 4
            p1 = (int(corners[2 * i]),int(corners[2 * i + 1]))
            p2 = (int(corners[2 * j]),int(corners[2 * j + 1]))
            mat = cv2.line(mat, p1, p2, line_color, 2)

        # Manually reshape 'corners'
        distorted_corners = numpy.empty([4,2], dtype=numpy.float32)
        for i in range(4):
            distorted_corners[i][0] = corners[2 * i]
            distorted_corners[i][1] = corners[2 * i + 1]

        # run the OpenCV undistortion routine to fix the corners
        undistorted_corners = cv2.undistortImagePoints(distorted_corners,
                                                       camera_intrinsics,
                                                       camera_distortion)
        
        for i in range(4):
            corners[2 * i] = undistorted_corners[i][0][0]
            corners[2 * i + 1] = undistorted_corners[i][0][1]

        if numpy.abs((2*detection.getCenter().x - x_resolution)/x_resolution) < min_tag_x:
            min_tag_x = numpy.abs((2*detection.getCenter().x - x_resolution)/x_resolution)
            best_detection = detection
            best_Corners = corners

        has_tag = True

    if has_tag:
        # run the pose estimator using the fixed corners
        camera_to_tag = pose_estimator.estimate(
        homography = best_detection.getHomography(),
            corners = tuple(best_Corners)
        )

        best_tag = best_detection.getId()

        # First, we flip the camera_to_tag transform's angle 180 degrees around the y axis
        # since the tag is oriented into the field
        flip_tag_rotation = Rotation3d(axis = (0, 1, 0), angle = math.pi)
        camera_to_tag = Transform3d(camera_to_tag.translation(),
                                    camera_to_tag.rotation().rotateBy(flip_tag_rotation))

        # The Camera To Tag transform is in a East/Down/North coordinate system,
        # but we want it in the WPILib standard North/West/Up
        camera_to_tag = CoordinateSystem.convert(camera_to_tag,
                                                 CoordinateSystem.EDN(),
                                                 CoordinateSystem.NWU())

        tag_to_camera = camera_to_tag.inverse()

        # Check if this tag is both the current best, and is in reef_tags
        theta = tag_to_camera.rotation().z
        theta -= numpy.sign(theta) * math.pi

        best_tag_center_x = (2 * best_detection.getCenter().x - x_resolution) / x_resolution
        best_tag_to_camera = tag_to_camera

        for i in range(4):
            j = (i + 1) % 4
            p1 = (int(best_Corners[2 * i]),int(best_Corners[2 * i + 1]))
            p2 = (int(best_Corners[2 * j]),int(best_Corners[2 * j + 1]))
            mat = cv2.line(mat, p1, p2, best_color, 2)

    # Publish everything

    outputStream.putFrame(mat)

    # Publish global position
    robot_x.set(robot_pose.x)
    robot_y.set(robot_pose.y)
    #robot_z.set(robot_pose.z)

    # Publish local position & rotation
    tag_to_camera_x.set(best_tag_to_camera.x)
    tag_to_camera_y.set(best_tag_to_camera.y)
    #tag_to_camera_z.set(best_tag_to_camera.z)
    tag_to_camera_theta.set(theta)

    # Other
    apriltag_presence.set(has_tag)
    best_tag_id_topic.set(best_tag)
    tag_center_x.set(best_tag_center_x)
    all_tags_topic.set(all_tags)
