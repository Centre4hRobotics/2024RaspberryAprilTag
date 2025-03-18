import subprocess
import json
import math
import ntcore
import numpy
import cv2
import robotpy_apriltag
from wpimath.geometry import Transform3d, Rotation3d, Pose3d, Translation3d, CoordinateSystem
from cscore import CameraServer

#  All settings
IS_TABLE_HOST = False
TEAM_NUMBER = 4027
FRAME_RATE = 30
CAMERA_PROFILE = "640x480"

# Loading the AprilTag data
aprilTag_field_layout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

# Load camera calibration
with open('CameraCalibration.json') as json_data:
    calibration_data = json.load(json_data)

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

# Create the PoseEstimator & adjust its settings
pose_estimator = robotpy_apriltag.AprilTagPoseEstimator(pose_estimator_config)

aprilTag_detector = robotpy_apriltag.AprilTagDetector()
aprilTag_detector.addFamily("tag36h11", 3)

aprilTag_detector_config = aprilTag_detector.getConfig()
aprilTag_detector_config.numThreads = 4
aprilTag_detector_config.quadSigma = 0.5
aprilTag_detector_config.quadDecimate = 1
aprilTag_detector.setConfig(aprilTag_detector_config)

quad_threshold_parameters = aprilTag_detector.getQuadThresholdParameters()
quad_threshold_parameters.minClusterPixels = 5
quad_threshold_parameters.criticalAngle = 0.79
aprilTag_detector.setQuadThresholdParameters(quad_threshold_parameters)


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

# Global position of the robot
robot_x = table.getDoubleTopic("Global X").publish()
robot_y = table.getDoubleTopic("Global Y").publish()
#robot_z = table.getDoubleTopic("Global Z").publish()

# Tag to camera transform (this is more useful than the raw pose)
tag_to_camera_x = table.getDoubleTopic("tag_to_camera X").publish()
tag_to_camera_y = table.getDoubleTopic("tag_to_camera Y").publish()
tag_to_camera_z = table.getDoubleTopic("tag_to_camera Z").publish()
tag_to_camera_theta = table.getDoubleTopic("tag_to_camera Theta").publish()

# Raw tag center (just the raw center of the tag with no pose estimation.
# Should be more stable when we're fine tuning our pose)
# Is a value from -1 to 1
tag_center_x = table.getDoubleTopic("Tag Center X").publish()

# Returns whether we have a tag
apriltag_presence = table.getBooleanTopic("AprilTag Presence").publish()

best_tag_id_topic = table.getIntegerTopic("Best Tag ID").publish()

camera_choice = table.getStringTopic("Using Camera").publish()
camera_choice.set("LEFT")

camera_string = table.getStringTopic("Using Camera").subscribe("NO TABLE FOUND")


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

# Etc.
robot_pose = Pose3d()
best_tag_to_camera = Transform3d()
best_tag_center_x = 0
best_tag = -1
theta = 0
reef_tags = [6,7,8,9,10,11,17,18,19,20,21,22]

# Main loop
while True:
    has_tag = False

    if camera_string.get() == "LEFT":
        _, mat = cv_sink_left.grabFrame(mat)

    else:
        _, mat = cv_sink_right.grabFrame(mat)

    # Bill: both cameras were upside down so I rotated them.
    mat = cv2.rotate(mat, cv2.ROTATE_180)
    gray_mat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

    detections = aprilTag_detector.detect(gray_mat)

    min_tag_x = 100000

    for detection in detections:

        # Remove detection if it is not a reef tag.
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
        best_color = (255,255,0)
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

    tag_to_camera_theta.set(theta)

    # Other
    apriltag_presence.set(has_tag)
    best_tag_id_topic.set(best_tag)
    tag_center_x.set(best_tag_center_x)
