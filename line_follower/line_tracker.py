# Third-Party Libraries
import os
import numpy as np

from yolo_onnx_runner import YOLO

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped 
from ament_index_python.packages import get_package_share_directory

# Project-Specific Imports
from line_follower.utils import (
    get_corners, 
    to_surface_coordinates, 
    read_transform_config, 
    draw_box,
    parse_predictions,
    get_base,
)
from ros2_numpy import image_to_np, np_to_image, np_to_pose

class LineFollower(Node):
    def __init__(self, filepath, debug = False):
        super().__init__('line_tracker')

        # Define a model
        self.model = YOLO('../models/best.onnx')

        # Define a message to send when the line tracker has lost track
        self.lost_msg = PoseStamped()
        self.lost_msg.pose.position.x = self.lost_msg.pose.position.y = self.lost_msg.pose.position.z = float('nan')

        # Plot the result if debug is True
        self.debug = debug

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at '{filepath}' was not found.")

        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(filepath)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Publisher to send calculated waypoints
        self.publisher = self.create_publisher(PoseStamped, '/waypoint', qos_profile)
        
        # Publisher to send processed result images for visualization
        if self.debug:
            self.im_publisher = self.create_publisher(Image, '/result', qos_profile)


        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        self.get_logger().info("Line Tracker Node started.")

    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Get the line center from the model
        results = self.model(image)
        img_mask = parse_predictions(results)

        # Plot and publish result if debug is Trze
        if self.debug:
            pass
            # corners = self.get_corners(self.win_x)
            # result = draw_box(image, im_canny, corners)
            # im_msg = np_to_image(result)
            # self.im_publisher.publish(im_msg)


        if not img_mask is None:
            # Get center coordinates from mask
            cx, cy = get_base(img_mask)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            # Extract the timestamp from the incoming image message
            timestamp = msg.header.stamp

            # Publish waypoint as Pose message
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)

        else:
            self.publisher.publish(self.lost_msg)
            self.get_logger().info("Lost track!")

    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('win_h', 20),
                ('win_w', 90),
                ('win_x', 310),
                ('win_y', 280),
                ('image_w', 640),
                ('image_h', 360),
                ('canny_min', 80),
                ('canny_max', 180),
                ('k', 3)
            ]
        )

    def load_params(self):
        try:
            self.win_h = self.get_parameter('win_h').get_parameter_value().integer_value
            self.win_w = self.get_parameter('win_w').get_parameter_value().integer_value
            self.win_x = self.get_parameter('win_x').get_parameter_value().integer_value
            self.win_y = self.get_parameter('win_y').get_parameter_value().integer_value
            self.image_w = self.get_parameter('image_w').get_parameter_value().integer_value
            self.image_h = self.get_parameter('image_h').get_parameter_value().integer_value
            self.canny_min = self.get_parameter('canny_min').get_parameter_value().integer_value
            self.canny_max = self.get_parameter('canny_max').get_parameter_value().integer_value
            self.k = self.get_parameter('k').get_parameter_value().integer_value

            # Ensure kernel is at least 3 and an odd number
            self.k = max(3, self.k + (self.k % 2 == 0))
            self.kernel = (self.k, self.k)

            # Returns a function that calculates corner points with fixed window and image parameters
            self.get_corners = lambda win_x: get_corners(win_x, self.win_y, self.win_w, self.win_h, self.image_w, self.image_h)

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")

def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    filepath = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'
            
    rclpy.init(args=args)
    node = LineFollower(filepath, debug = True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()