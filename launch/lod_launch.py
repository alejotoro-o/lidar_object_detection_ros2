from launch import LaunchDescription
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

    return LaunchDescription([
        lidar_object_detection,
    ])