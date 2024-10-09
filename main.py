from cscore import CameraServer # type: ignore
import ntcore # type: ignore
import numpy # type: ignore
import cv2 # type: ignore
import robotpy_apriltag # type: ignore
import wpimath # type: ignore

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
    0.1651,  #tag size in meters
    543.93,  #Fx: x focal length
    544.98,  #Fy: y focal length
    316.29,  #Cx: x focal center (based on 640x480 resolution)
    250.55,  #Cy: y focal center (based on 640x480 resolution)
)
# create the PoseEstimator
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)


aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

#Camera constants
xResolution = 640
yResolution = 480
frameRate = 30

ntInstance = ntcore.NetworkTableInstance.getDefault()
ntInstance.startServer()

camera = CameraServer.startAutomaticCapture()

CameraServer.enableLogging()

table = ntInstance.getTable("Vision")

xPublisher = table.getDoubleTopic("X").publish()
yPublisher = table.getDoubleTopic("Y").publish()
zPublisher = table.getDoubleTopic("Distance").publish()
tagIdPublisher = table.getIntegerTopic("Tag ID").publish()
tagQuestionMarkQuestionMark = table.getBooleanTopic("Is there an AprilTag?").publish()


camera.setResolution(xResolution, yResolution)
camera.setFPS(frameRate)

cvSink = CameraServer.getVideo()

outputStream = CameraServer.putVideo("Vision", xResolution, yResolution)

#Images
mat = numpy.zeros(shape=(xResolution, yResolution, 3), dtype=numpy.uint8)
grayMat = numpy.zeros(shape=(xResolution, yResolution), dtype=numpy.uint8)

#Drawing stuff
lineColor = (0,0,255)
crosshairColor = (0, 255, 0)
crosshairSize = 20

#main loop
while True:
	time, mat = cvSink.grabFrame(mat)

	grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

	detections = aprilTagDetector.detect(grayMat)

	#Crosshair
	point1 = (int(xResolution / 2 - crosshairSize), int(yResolution / 2))
	point2 = (int(xResolution / 2 + crosshairSize), int(yResolution / 2))
	mat = cv2.line(mat, point1, point2, crosshairColor, 2)
	point1 = (int(xResolution / 2), int(yResolution / 2 - crosshairSize))
	point2 = (int(xResolution / 2), int(yResolution / 2 + crosshairSize))
	mat = cv2.line(mat, point1, point2, crosshairColor, 2)

	#mat = cv2.circle(mat, (int(xResolution / 2), int(yResolution / 2)), 30, (0,255,255), 5)

	if (detections != []):
        # estimate the pose from the first tag
		pose = poseEstimator.estimate(detections[0])
		tagId = detections[0].getId()
	else:
        # no tags found, so just store an empty transform
		pose = wpimath.geometry.Transform3d()
		tagId = 0

	#Outline
	for detection in detections:
		for i in range(4):
			j = (i + 1) % 4
			p1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
			p2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
			mat = cv2.line(mat, p1, p2, lineColor, 2)

	#Output
	outputStream.putFrame(mat)
	xPublisher.set(pose.x)
	yPublisher.set(pose.y)
	zPublisher.set(pose.z)
	tagIdPublisher.set(tagId)
	tagQuestionMarkQuestionMark.set(detections != [])
