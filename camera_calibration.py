import json
import numpy

class Camera_Calibration:
    def __init__(self, profile):
        with open('CameraCalibration.json', encoding="utf-8") as json_data:
            calibration_data = json.load(json_data)

        print("Opened CameraCalibration.json")

        camera_data = calibration_data[profile]

        self.Fx = camera_data["Intrinsics"]["Fx"]
        self.Fy = camera_data["Intrinsics"]["Fy"]
        self.Cx = camera_data["Intrinsics"]["Cx"]
        self.Cy = camera_data["Intrinsics"]["Cy"]

        self.x_resolution = camera_data["Resolution"]["x"]
        self.y_resolution = camera_data["Resolution"]["y"]


        self.camera_distortion = numpy.float32([
            camera_data["Distortion"]["A"],
            camera_data["Distortion"]["B"],
            camera_data["Distortion"]["C"],
            camera_data["Distortion"]["D"],
            camera_data["Distortion"]["E"] ])
        self.camera_intrinsics = numpy.eye(3)
        self.camera_intrinsics[0][0] = self.Fx
        self.camera_intrinsics[1][1] = self.Fy
        self.camera_intrinsics[0][2] = self.Cx
        self.camera_intrinsics[1][2] = self.Cy

        print("Set camera calibration data")