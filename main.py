""" This is the main file for FRC Team 4027's 2024 AprilTag Vision. """

import subprocess
import math
import numpy
import cv2
import robotpy_apriltag
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, CoordinateSystem
from cscore import CameraServer
from camera_calibration import Camera_Calibration
from apriltag_detector import AprilTag_Detector
from nt_tables import Tables

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

camera_calibration = Camera_Calibration(CAMERA_PROFILE)

# Create the april_tag_detector & adjust its settings

apriltag_detector = AprilTag_Detector(camera_calibration)

# Creating the network tables

# Check network tables host flag

network_tables = Tables( TEAM_NUMBER if not IS_TABLE_HOST else ())

# Activate camera stuff

CameraServer.enableLogging()

left_camera = CameraServer.startAutomaticCapture(2)

right_camera = CameraServer.startAutomaticCapture(0)

left_camera.setResolution(camera_calibration.x_resolution, camera_calibration.y_resolution)
right_camera.setResolution(camera_calibration.x_resolution, camera_calibration.y_resolution)

cv_sink_left = CameraServer.getVideo(left_camera)
cv_sink_right = CameraServer.getVideo(right_camera)

outputStream = CameraServer.putVideo("Vision", camera_calibration.x_resolution, camera_calibration.y_resolution)

rc = subprocess.call("chmod u+rx set_camera_settings.sh "
+ "&& /home/pi/2024RaspberryAprilTag/set_camera_settings.sh", shell = True)
print("set_camera_settings.sh returned: ", rc)

# Images
mat = numpy.zeros(shape=(camera_calibration.x_resolution, camera_calibration.y_resolution, 3), dtype=numpy.uint8)
gray_mat = numpy.zeros(shape=(camera_calibration.x_resolution, camera_calibration.y_resolution), dtype=numpy.uint8)

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
    min_tag_x = camera_calibration.x_resolution + 1
    has_tag = False

    if network_tables.camera_string.get() == "LEFT":
        # grabFrame returns two values, the first of which we don't care about
        _, mat = cv_sink_left.grabFrame(mat)

    else:
        _, mat = cv_sink_right.grabFrame(mat)

    # Rotate video to not be upside down/on it's side
    mat = cv2.rotate(mat, cv2.ROTATE_180)

    # Convert the video to grayscale
    gray_mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

    # Detect AprilTags
    detections = apriltag_detector.april_tag_detector.detect(gray_mat)

    for detection in detections:

        all_tags.append(detection.getID)

        if network_tables.tag_choice.get() > 0 and not detection.getId() is network_tables.tag_choice.get():
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
                                                       camera_calibration.camera_intrinsics,
                                                       camera_calibration.camera_distortion)
        
        for i in range(4):
            corners[2 * i] = undistorted_corners[i][0][0]
            corners[2 * i + 1] = undistorted_corners[i][0][1]

        if numpy.abs((2*detection.getCenter().x - camera_calibration.x_resolution)/camera_calibration.x_resolution) < min_tag_x:
            min_tag_x = numpy.abs((2*detection.getCenter().x - camera_calibration.x_resolution)/camera_calibration.x_resolution)
            best_detection = detection
            best_Corners = corners

        has_tag = True

    if has_tag:
        # run the pose estimator using the fixed corners
        camera_to_tag = apriltag_detector.pose_estimator.estimate(
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

        best_tag_center_x = (2 * best_detection.getCenter().x - camera_calibration.x_resolution) / camera_calibration.x_resolution
        best_tag_to_camera = tag_to_camera

        for i in range(4):
            j = (i + 1) % 4
            p1 = (int(best_Corners[2 * i]),int(best_Corners[2 * i + 1]))
            p2 = (int(best_Corners[2 * j]),int(best_Corners[2 * j + 1]))
            mat = cv2.line(mat, p1, p2, best_color, 2)

    # Publish everything

    outputStream.putFrame(mat)

    network_tables.Set_Values(robot_pose, best_tag_to_camera, theta, best_tag_center_x, all_tags, has_tag, best_tag)