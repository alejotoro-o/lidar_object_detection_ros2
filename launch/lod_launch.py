from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    lidar_object_detection = Node(
        package="lidar_object_detection_ros2",
        executable="lidar_object_detection.py",
        parameters=[
            # DBSCAN Parameters
            {"dbscan_eps": 0.2},
            {"dbscan_min_samples": 5},
            # Lidar and Frame Parameters
            {"lidar_angular_resolution": 0.5},
            {"frame_id": "map"},
            {"lidar_frame_id": "lidar"},
            {"flip_x_axis": True},
            {"flip_y_axis": True},
            {"update_rate": 0.2},
            # L-shape Detection Parameters
            {"min_l": 0.05},
            {"max_l": 1.0},
            # Lidar Clipping Parameters
            {"min_range": 0.1},
            {"max_range": 8.0},
            # Kalman Filter Parameters
            {"use_kalman_filter": True},
            {"kf_q_std": 0.05},
            {"kf_r_std": 0.1},
            # Visualization Parameters
            {"publish_markers": True},
            {"color_bbox": [0.0, 1.0, 0.0, 0.5]},  # Green
            {"color_text": [1.0, 1.0, 1.0, 1.0]},  # White
            {"color_arrow": [1.0, 1.0, 0.0, 1.0]}, # Yellow
            {"min_velocity_show": 0.01},
            # Tracking/Association Parameters
            {"max_disappeared": 6},
            {"max_association_distance": 0.8}
        ],
        output="screen",
    )

    return LaunchDescription([
        lidar_object_detection,
    ])