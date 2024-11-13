import os
import launch
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection
from launch_ros.actions import Node


def generate_launch_description():

    lidar_object_detection = Node(
        package="lidar_object_detection_ros2",
        executable="lidar_object_detection.py",
        parameters=[
            {"dbscan_eps": 0.2},
            {"dbscan_min_samples": 5},
            {"lidar_angular_resolution": 0.5},
            {"frame_id": "map"},
            {"lidar_frame_id": "lidar"},
            {"flip_x_axis": True},
            {"flip_y_axis": True},
            {"update_rate": 0.2},
            {"min_l": 0.05},
            {"max_l": 1.0}
        ],
        output="screen",
    )

    lod_visualization = Node(
        package="lidar_object_detection_ros2",
        executable="lod_visualization.py",
        parameters=[
            {"x_lims": [-4.0,4.0]},
            {"y_lims": [-3.0,3.0]},
            {"frame_id": "map"},
            {"lidar_frame_id": "lidar"},
        ]
    )

    return LaunchDescription([
        lidar_object_detection,
        lod_visualization
    ])