import robotpy_apriltag

class AprilTag_Detector:
    def __init__(self, camera_calibration):

        pose_estimator_config = robotpy_apriltag.AprilTagPoseEstimator.Config(
            0.1651,  #tag size in meters
            camera_calibration.Fx,
            camera_calibration.Fy,
            camera_calibration.Cx,
            camera_calibration.Cy,
        )

        self.pose_estimator = robotpy_apriltag.AprilTagPoseEstimator(pose_estimator_config)

        self.april_tag_detector = robotpy_apriltag.AprilTagDetector()
        self.april_tag_detector.addFamily("tag36h11", 3)

        april_tag_detector_config = self.april_tag_detector.getConfig()
        april_tag_detector_config.numThreads = 4
        april_tag_detector_config.quadSigma = 0.5
        april_tag_detector_config.quadDecimate = 1
        self.april_tag_detector.setConfig(april_tag_detector_config)

        quad_threshold_parameters = self.april_tag_detector.getQuadThresholdParameters()
        quad_threshold_parameters.minClusterPixels = 5
        quad_threshold_parameters.criticalAngle = 0.79
        self.april_tag_detector.setQuadThresholdParameters(quad_threshold_parameters)

        print("Created pose_estimator and april_tag_detector")