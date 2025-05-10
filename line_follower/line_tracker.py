# Third-Party Libraries
import os, cv2
import numpy as np

from ultralytics import YOLO

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
)  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import (
    get_corners,
    to_surface_coordinates,
    read_transform_config,
    draw_box,
    parse_predictions,
    get_base,
    detect_bbox_center,
)

from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_compressedimage


class LineFollower(Node):
    def __init__(self, model_path, config_path, debug=False):
        super().__init__("line_tracker")

        # Define a message to send when the line tracker has lost track
        self.lost_msg = PoseStamped()
        self.lost_msg.pose.position.x = self.lost_msg.pose.position.y = (
            self.lost_msg.pose.position.z
        ) = float("nan")

        # Plot the result if debug is True
        self.debug = debug

        # Check if the model and config files exist
        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")

        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(config_path)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile,
        )

        # Publisher to send calculated waypoints
        self.waypoint_publisher = self.create_publisher(
            PoseStamped, "/waypoint", qos_profile
        )

        # Publisher to send 3d object positions
        self.object_publisher = self.create_publisher(
            PoseStamped, "/object", qos_profile
        )

        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(
            CompressedImage, "/result", qos_profile
        )

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = YOLO(model_path)

        # Map class IDs to labels and labels to IDs
        self.id2label = self.model.names
        targets = ["stop", "speed_3mph", "speed_2mph"]
        self.id2target = {
            id: lbl for id, lbl in self.id2label.items() if lbl in targets
        }

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info(
            "Line Tracker Node started. Custom YOLO model loaded successfully."
        )

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Run YOLO inference
        predictions = self.model(image, verbose=False)

        # Draw results on the image
        plot = predictions[0].plot()

        # Convert back to ROS2 Image and publish
        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

        # Publish predictions
        self.im_publisher.publish(im_msg)

        success, mask = parse_predictions(predictions)

        if success:
            cx, cy = get_base(mask)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            # Extract the timestamp from the incoming image message
            timestamp = msg.header.stamp

            # Publish waypoint as Pose message
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.waypoint_publisher.publish(pose_msg)
        else:
            self.waypoint_publisher.publish(self.lost_msg)
            self.get_logger().info("Lost track!")

        # Loop through each label and publish detections
        for id, lbl in self.id2target.items():
            detected, u, v = detect_bbox_center(predictions, id)

            if detected:
                # Transform from pixel to world coordinates
                x, y = self.to_surface_coordinates(u, v)

                # Publish object as Pose message
                pose_msg = np_to_pose(np.array([x, y, id]), 0.0, timestamp=timestamp)

            else:
                pose_msg = (
                    self.lost_msg
                )  # Attention: this creates a reference — use deepcopy() if you want self.stop_msg to remain unchanged
                pose_msg.pose.position.z = float(id)
                self.get_logger().info(f"Lost track of {self.id2target[id]}!")
            self.object_publisher.publish(pose_msg)

    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace="",
            parameters=[
                ("win_h", 20),
                ("win_w", 90),
                ("win_x", 310),
                ("win_y", 280),
                ("image_w", 640),
                ("image_h", 360),
                ("canny_min", 80),
                ("canny_max", 180),
                ("k", 3),
            ],
        )

    def load_params(self):
        try:
            self.win_h = self.get_parameter("win_h").get_parameter_value().integer_value
            self.win_w = self.get_parameter("win_w").get_parameter_value().integer_value
            self.win_x = self.get_parameter("win_x").get_parameter_value().integer_value
            self.win_y = self.get_parameter("win_y").get_parameter_value().integer_value
            self.image_w = (
                self.get_parameter("image_w").get_parameter_value().integer_value
            )
            self.image_h = (
                self.get_parameter("image_h").get_parameter_value().integer_value
            )
            self.canny_min = (
                self.get_parameter("canny_min").get_parameter_value().integer_value
            )
            self.canny_max = (
                self.get_parameter("canny_max").get_parameter_value().integer_value
            )
            self.k = self.get_parameter("k").get_parameter_value().integer_value

            # Ensure kernel is at least 3 and an odd number
            self.k = max(3, self.k + (self.k % 2 == 0))
            self.kernel = (self.k, self.k)

            # Returns a function that calculates corner points with fixed window and image parameters
            self.get_corners = lambda win_x: get_corners(
                win_x, self.win_y, self.win_w, self.win_h, self.image_w, self.image_h
            )

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")


def main(args=None):

    # Get the package share directory
    package_share_dir = get_package_share_directory("line_follower")

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = package_share_dir + "/config/transform_config_640x360.yaml"

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix("line_follower").replace("install", "src")
    model_path = pkg_path + "/models/best.pt"

    rclpy.init(args=args)
    node = LineFollower(model_path=model_path, config_path=config_path, debug=True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
