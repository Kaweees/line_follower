# Third-Party Libraries
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import to_surface_coordinates, read_transform_config, parse_predictions, get_base, detect_bbox_center
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_compressedimage


class LineFollower(Node):
    def __init__(self, model_path, config_path):
        '''
        Initializes a line follower ROS node. Follows a line and observes traffic signs using a YOLO segmentation model.
        Subscribes to raw image data (image frames from camera).
        Publishes:
            /waypoint   = center of the line to follow in (real world coordinates).
            /object     = object id and position (real world coordinates).
            /result     = visualization of results (segmentation info plotted on raw image data).
        '''
        super().__init__('line_tracker')

        # Indicate starting line follower node
        self.get_logger().info("Line Tracker Node starting initialization...")

        # Ensure input paths exist
        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")

        # Define a message to send when the line tracker has lost track
        self.not_found_msg = PoseStamped()
        self.not_found_msg.pose.position.x = self.not_found_msg.pose.position.y = self.not_found_msg.pose.position.z = float('nan')

        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(config_path)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Subscriber to receive camera images from sensor
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Publisher to send calculated waypoints
        self.way_publisher = self.create_publisher(PoseStamped, '/waypoint', qos_profile)

        # Publisher to send object positions (in real world (3D) coordinates)
        self.obj_publisher = self.create_publisher(PoseStamped, '/object', qos_profile)
        
        # Publisher to send processed result images for visualization and debugging 
        self.im_publisher = self.create_publisher(CompressedImage, '/result', qos_profile)

        # Returns a function that converts pixel coordinates to surface (real world) coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        id2label = self.model.names
        targets = ['stop', 'speed_3mph', 'speed_2mph']
        self.center_line_id: list[int] = [id_ for id_, lbl in id2label.items() if lbl == 'center']
        self.id2target = {id: lbl for id, lbl in id2label.items() if lbl in targets}

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO model loaded successfully.")
        self.get_logger().info(f"Detecting id:objects listed here {self.id2target} and {self.center_line_id}:center")


    def load_model(self, filepath):
        '''
        Returns the YOLO model at filepath.
        '''
        # Get the model
        model = YOLO(filepath)

        # Get the image size (imgsz) the loaded model was trained on.
        self.imgsz = model.args['imgsz']  # type: ignore

        # Initialize model
        print("line_tracker.py: Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im)  
        print("line_tracker.py: Model initialization complete.")

        # Return the model
        return model


    def image_callback(self, msg):
        '''
        Callback for each image message received via subscription. 
        Passes image through YOLO model and publishes information about objects detected.
        '''
        # Extract the timestamp from the incoming image message
        timestamp = msg.header.stamp
        # Convert ROS image to numpy format
        image, timestamp_unix = image_to_np(msg) # timestamp_unix is the image timestamp in seconds (Unix time)

        # Run YOLO inference
        predictions = self.model(image, verbose=False)

        # Draw results on the image
        plot = predictions[0].plot()
        # Convert back to ROS2 Image for publishing
        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
        # Publish predictions for debugging purposes
        self.im_publisher.publish(im_msg)

        # Try to create a mask for pixels in image that are center line pixels
        success, line_mask = parse_predictions(predictions, class_ids=self.center_line_id)

        # Center line found in the image
        if success:
            # Get the average coordinate of a line pixel  
            cx, cy = get_base(line_mask)
            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            # Publish waypoint as a pose message
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.way_publisher.publish(pose_msg)

        # Center line not found in image
        else:
            # Publish line not found message
            self.way_publisher.publish(self.not_found_msg)
            # Indicate lost track of line in log
            self.get_logger().info("Lost track of line!")

        # For each of the pre-defined important objects 
        for id, lbl in self.id2target.items():
            # Try to find the center of the object's bounding box
            detected, u, v = detect_bbox_center(predictions, id)

            # Object with the current id was detected
            if detected and u is not None and v is not None:
                # Transform from pixel to world coordinates
                x, y = self.to_surface_coordinates(u.item(), v.item())

                # Create pose message to publish location of object
                pose_msg = np_to_pose(np.array([x, y, id]), 0.0, timestamp=timestamp)
            else:
                # Create pose message to indicate object with this id not found
                # Attention: this creates a reference — use deepcopy() if you want self.not_found_msg to remain unchanged
                pose_msg = self.not_found_msg 
                pose_msg.pose.position.z = float(id)

            # Publish location of object as a pose message (with object id in z coordinate)
            self.obj_publisher.publish(pose_msg)


def main(args=None):
    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') 
    model_path = pkg_path + '/models/best.pt'

    rclpy.init(args=args)
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()