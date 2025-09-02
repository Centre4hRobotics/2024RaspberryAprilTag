import ntcore

class Tables:
    def __init__(self, team_number):
        ntInstance = ntcore.NetworkTableInstance.getDefault()

        # Check network tables host flag
        if team_number:
            ntInstance.startServer()
            print("Started network table server")

        else:
            ntInstance.setServerTeam(team_number)
            ntInstance.startClient4("visionPi")
            print("Started network table client")

        table = ntInstance.getTable("AprilTag Vision")


        # Export robot position

        # Global position of the robot
        self.robot_x = table.getDoubleTopic("Global X").publish()
        self.robot_y = table.getDoubleTopic("Global Y").publish()
        #robot_z = table.getDoubleTopic("Global Z").publish()

        # Tag to camera transform (this is more useful than the raw pose)
        self.tag_to_camera_x = table.getDoubleTopic("tag_to_camera X").publish()
        self.tag_to_camera_y = table.getDoubleTopic("tag_to_camera Y").publish()
        #tag_to_camera_z = table.getDoubleTopic("tag_to_camera Z").publish()
        self.tag_to_camera_theta = table.getDoubleTopic("tag_to_camera Theta").publish()

        # Location of the tag on video feed (No pose estimation)
        # Returns values between -1 and 1
        self.tag_center_x = table.getDoubleTopic("Tag Center X").publish()
        #tag_center_Y = table.getDoubleTopic("Tag Center Y").publish()

        # Returns all visible Tags
        self.all_tags = table.getIntegerArrayTopic("All Tags").publish()

        # Returns whether we have a tag
        self.apriltag_presence = table.getBooleanTopic("AprilTag Presence").publish()

        # Returns the best tag visible/which tag is being reported on
        self.best_tag = table.getIntegerTopic("Best Tag ID").publish()

        # Tells which camera is being used. Can also be changed by others
        self.camera_choice = table.getStringTopic("Using Camera").publish()
        self.camera_choice.set("LEFT")

        self.camera_string = table.getStringTopic("Using Camera").subscribe("NO TABLE FOUND")

        self.tag_choice_topic = table.getIntegerTopic("Tag Choice").publish()
        self.tag_choice_topic.set(0)

        self.tag_choice = table.getIntegerTopic("Tag Choice").subscribe(0)

    def Set_Values(self, robot_pos, tag_to_camera_pos, tag_to_camera_theta, tag_center_pos, all_tags, has_tag, best_tag):
        self.robot_x.set(robot_pos.x)
        self.robot_y.set(robot_pos.y)
        self.tag_to_camera_x.set(tag_to_camera_pos.x)
        self.tag_to_camera_y.set(tag_to_camera_pos.y)
        self.tag_to_camera_theta.set(tag_to_camera_theta)
        self.tag_center_x.set(tag_center_pos.x)
        self.all_tags.set(all_tags)
        self.apriltag_presence.set(has_tag)
        self.best_tag.set(best_tag)