#!/usr/bin/env python3
import rclpy
import rclpy.duration
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from lidar_object_detection_ros2.msg import Pose2D, Object, ObjectsArray, ScanClusters

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment

class LidarObjectDetectionNode(Node):

    def __init__(self):

        super().__init__('lidar_object_detection')

        ## Parameters
        self.declare_parameter("dbscan_eps", 0.1)
        dbscan_esp = self.get_parameter("dbscan_eps").get_parameter_value().double_value

        self.declare_parameter("dbscan_min_samples", 5)
        dbscan_min_samples = self.get_parameter("dbscan_min_samples").get_parameter_value().integer_value

        self.declare_parameter("frame_id", "")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        self.declare_parameter("lidar_frame_id", "laser_link")
        self.lidar_frame_id = self.get_parameter("lidar_frame_id").get_parameter_value().string_value
        if self.frame_id == "":
            self.frame_id = self.lidar_frame_id

        self.declare_parameter("lidar_angular_resolution", 0.5)
        self.lidar_ang_res = np.deg2rad(self.get_parameter('lidar_angular_resolution').get_parameter_value().double_value)

        self.declare_parameter("update_rate", 0.1)
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value

        self.declare_parameter("flip_x_axis", False)
        flip_x_axis = self.get_parameter("flip_x_axis").get_parameter_value().bool_value
        self.flip_x_axis = -1 if flip_x_axis else 1

        self.declare_parameter("flip_y_axis", False)
        flip_y_axis = self.get_parameter("flip_y_axis").get_parameter_value().bool_value
        self.flip_y_axis = -1 if flip_y_axis else 1

        self.declare_parameter("min_l", 0.05)
        self.min_l = self.get_parameter('min_l').get_parameter_value().double_value

        self.declare_parameter("max_l", 1.0)
        self.max_l = self.get_parameter('max_l').get_parameter_value().double_value

        ## Variables
        self.ranges = []
        self.dbscan = DBSCAN(eps=dbscan_esp, min_samples=dbscan_min_samples)

        ## Cluster asociation
        self.tracked_objects = {}  # {id: {"corner": [x, y], "age": 0}}
        self.next_id = 0
        self.max_disappeared = 6   # Slightly higher to handle noise
        self.max_distance = 0.8    # Max distance the corner can move between scans

        ## TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        ## Subscriptions
        self.create_subscription(LaserScan, "scan", self.scan_callback, 10)

        ## Publishers
        self.clusters_publisher = self.create_publisher(ScanClusters, "lod_clusters", 10)
        self.objects_publisher = self.create_publisher(ObjectsArray, "lod_objects", 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "lod_markers", 10)

        ## Timer
        self.timer = self.create_timer(update_rate, self.on_timer)
        self.scan_time = rclpy.time.Time()
        
    def scan_callback(self, scan_msg):
        
        self.scan_time = rclpy.time.Time(seconds=scan_msg.header.stamp.sec, nanoseconds=scan_msg.header.stamp.nanosec)
        self.ranges = scan_msg.ranges

    def on_timer(self):
        
        current_lidar_angle = 0
        points = []

        clusters = ScanClusters()
        clusters.header.frame_id = self.frame_id
        clusters.header.stamp = self.get_clock().now().to_msg()
        clusters.points = []
        clusters.labels = []

        try:
            # Add a timeout (duration) to wait for the transform to become available
            t = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.lidar_frame_id,
                self.scan_time,
                timeout=rclpy.duration.Duration(seconds=0.1) # <--- Add this
            )
        except TransformException as ex:
            # Changed to warn to avoid spamming info if it's just a slight delay
            self.get_logger().warning(f'Could not transform {self.frame_id} to {self.lidar_frame_id}: {ex}')
            return
        
        theta_r = self._get_theta_from_quaternion(
            t.transform.rotation.x, 
            t.transform.rotation.y, 
            t.transform.rotation.z, 
            t.transform.rotation.w
        )

        for range in self.ranges:

            if range != float("+inf"):
                
                point_x = t.transform.translation.x + self.flip_x_axis*range*np.cos(theta_r - current_lidar_angle)
                point_y = t.transform.translation.y + self.flip_y_axis*range*np.sin(theta_r - current_lidar_angle)

                point = Pose2D()
                point.x = point_x
                point.y = point_y
                clusters.points.append(point)
                points.append([point.x, point.y])
            
            current_lidar_angle += self.lidar_ang_res

        if len(points) > 0:
            
            lidar_data = np.array(points)
            
            ## Clustering
            labels = self.dbscan.fit_predict(lidar_data)
         
            clusters.labels = labels.tolist()

            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[self.dbscan.core_sample_indices_] = True

            objects = ObjectsArray()
            objects.header.frame_id = self.frame_id
            objects.header.stamp = self.get_clock().now().to_msg()
            objects.objects = []

            for l in unique_labels:

                class_member_mask = labels == l

                xy = lidar_data[class_member_mask & core_samples_mask]

                ## Obtain L-Shapes
                if l != -1:
                    c1, theta, l1, l2 = self.cal_l_shape(xy)
                    
                    if l1 > self.min_l and l1 < self.max_l and l2 > self.min_l and l2 < self.max_l:

                        ## Rectangle center
                        x_cent = c1[0] + (l1*np.cos(theta) - l2*np.sin(theta))/2
                        y_cent = c1[1] + (l1*np.sin(theta) + l2*np.cos(theta))/2

                        obj = Object()
                        obj.id = int(l)
                        obj.l_shape.c1.x = float(c1[0])
                        obj.l_shape.c1.y = float(c1[1])
                        obj.l_shape.theta = float(theta)
                        obj.l_shape.l1 = float(l1)
                        obj.l_shape.l2 = float(l2)
                        obj.pose.x = float(x_cent)
                        obj.pose.y = float(y_cent)

                        objects.objects.append(obj)

            objects.objects = self._associate_clusters(objects.objects)

            self.clusters_publisher.publish(clusters)
            self.objects_publisher.publish(objects)

            if len(objects.objects) > 0:
                marker_array = self._get_marker_array(objects)
                self.marker_publisher.publish(marker_array)

    def variance_criterion(self, C1, C2):

        c1_max = np.max(C1)
        c1_min = np.min(C1)
        c2_max = np.max(C2)
        c2_min = np.min(C2)

        # Calculate distances d1 and d2
        d1 = np.minimum(np.abs(c1_max - C1), np.abs(C1 - c1_min))
        d2 = np.minimum(np.abs(c2_max - C2), np.abs(C2 - c2_min))

        e1 = []
        e2 = []

        # Compare distances
        for i in range(len(d1)):
            if d1[i] < d2[i]:
                e1.append(d1[i])
            else:
                e2.append(d2[i])

        v1 = -np.var(e1) if e1 else 0.0
        v2 = -np.var(e2) if e2 else 0.0

        gamma = v1 + v2

        return gamma
    
    def cal_l_shape(self, points):

        Q = []
        angle_step = 0.0174533 ## = 1 deg

        for search_theta in np.arange(0, np.pi/2 - angle_step, angle_step):

            e1 = np.array([np.cos(search_theta),np.sin(search_theta)]).T
            e2 = np.array([-np.sin(search_theta),np.cos(search_theta)]).T
            C1 = np.dot(points,e1)
            C2 = np.dot(points,e2)
            q = self.variance_criterion(C1,C2)
            Q.append([search_theta,q])

        Q = np.array(Q)
        i = np.argmax(Q[:,1],axis=0)
        theta_star = Q[i,0]

        C1_star = np.dot(points, np.array([np.cos(theta_star),np.sin(theta_star)]).T)
        C2_star = np.dot(points, np.array([-np.sin(theta_star),np.cos(theta_star)]).T)

        c1 = np.min(C1_star)
        c2 = np.min(C2_star)
        c3 = np.max(C1_star)
        c4 = np.max(C2_star)

        a1 = np.cos(theta_star)
        b1 = np.sin(theta_star)

        a2 = -np.sin(theta_star)
        b2 = np.cos(theta_star)

        # a3 = np.cos(theta_star)
        # b3 = np.sin(theta_star)

        # a4 = -np.sin(theta_star)
        # b4 = np.cos(theta_star)

        x1 = (b2*c1 - b1*c2)/(b2*a1 - b1*a2)
        y1 = (c2 - a2*x1)/b2

        # x2 = (b4*c1 - b1*c4)/(b4*a1 - b1*a4)
        # y2 = (c4 - a4*x2)/b4

        # x3 = (b4*c3 - b3*c4)/(b4*a3 - b3*a4)
        # y3 = (c4 - a4*x3)/b4

        # x4 = (b2*c3 - b3*c2)/(b2*a3 - b3*a2)
        # y4 = (c2 - a2*x4)/b2

        l1 = c3 - c1
        l2 = c4 - c2

        return (x1,y1), theta_star, l1, l2
    
    ###############
    ## Utilities ##
    ###############
    def _get_theta_from_quaternion(self, x, y, z, w):
        """
        Extracts the yaw (rotation around Z-axis) from a 3D quaternion.
        Used for processing TF transforms.
        """
        r = R.from_quat([x, y, z, w])
        # The last element of the rotvec is the rotation around the Z axis
        return r.as_rotvec()[-1]

    def _get_quaternion_from_theta(self, theta):
        """
        Converts a 2D yaw angle (theta) into a 4D quaternion [x, y, z, w].
        Used for publishing RViz Markers.
        """
        # Create rotation around Z-axis
        r = R.from_rotvec([0.0, 0.0, float(theta)])

        return r.as_quat()
    
    def _get_marker_array(self, objects_msg):
        marker_array = MarkerArray()
        
        for obj in objects_msg.objects:
            # 1. Bounding Box Marker (CUBE)
            bbox_marker = Marker()
            bbox_marker.header = objects_msg.header
            bbox_marker.ns = "bounding_boxes"
            bbox_marker.id = obj.id
            bbox_marker.type = Marker.CUBE
            bbox_marker.action = Marker.ADD
            
            # Position
            bbox_marker.pose.position.x = obj.pose.x
            bbox_marker.pose.position.y = obj.pose.y
            bbox_marker.pose.position.z = 0.1  # Slightly above ground
            
            # Orientation using our utility
            q = self._get_quaternion_from_theta(obj.l_shape.theta)
            bbox_marker.pose.orientation.x = q[0]
            bbox_marker.pose.orientation.y = q[1]
            bbox_marker.pose.orientation.z = q[2]
            bbox_marker.pose.orientation.w = q[3]
            
            # Scale (l1 and l2 from L-shape)
            bbox_marker.scale.x = obj.l_shape.l1
            bbox_marker.scale.y = obj.l_shape.l2
            bbox_marker.scale.z = 0.2 # Thickness of the box
            
            # Color (Semi-transparent green)
            bbox_marker.color.r = 0.0
            bbox_marker.color.g = 1.0
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.5
            
            bbox_marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(bbox_marker)

            # 2. ID Label Marker (TEXT)
            text_marker = Marker()
            text_marker.header = objects_msg.header
            text_marker.ns = "object_ids"
            text_marker.id = obj.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = obj.pose.x
            text_marker.pose.position.y = obj.pose.y
            text_marker.pose.position.z = 0.5 # Float above the box
            
            text_marker.scale.z = 0.2 # Text height
            text_marker.text = f"ID: {obj.id}"
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(text_marker)
            
        return marker_array
    
    def _associate_clusters(self, current_objects):
        if not current_objects:
            # Increment age for all tracked objects if no new detections
            for tid in list(self.tracked_objects.keys()):
                self.tracked_objects[tid]["age"] += 1
                if self.tracked_objects[tid]["age"] > self.max_disappeared:
                    del self.tracked_objects[tid]
            return current_objects

        # Extract new corners
        new_corners = np.array([[obj.l_shape.c1.x, obj.l_shape.c1.y] for obj in current_objects])
        
        tracked_ids = list(self.tracked_objects.keys())
        if not tracked_ids:
            # Register all as new
            for i, obj in enumerate(current_objects):
                obj.id = self._register_object(new_corners[i])
            return current_objects

        tracked_corners = np.array([self.tracked_objects[tid]["corner"] for tid in tracked_ids])

        # Distance matrix between old corners and new corners
        dist_matrix = np.linalg.norm(tracked_corners[:, np.newaxis] - new_corners, axis=2)

        # Hungarian Algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        assigned_new_indices = set()
        assigned_track_indices = set()

        for r, c in zip(row_ind, col_ind):
            # Only associate if the corner hasn't jumped too far
            if dist_matrix[r, c] < self.max_distance:
                tid = tracked_ids[r]
                current_objects[c].id = tid
                self.tracked_objects[tid]["corner"] = new_corners[c]
                self.tracked_objects[tid]["age"] = 0
                assigned_track_indices.add(r)
                assigned_new_indices.add(c)

        # Clean up lost tracks
        for r, tid in enumerate(tracked_ids):
            if r not in assigned_track_indices:
                self.tracked_objects[tid]["age"] += 1
                if self.tracked_objects[tid]["age"] > self.max_disappeared:
                    del self.tracked_objects[tid]

        # Register brand new objects
        for c in range(len(new_corners)):
            if c not in assigned_new_indices:
                current_objects[c].id = self._register_object(new_corners[c])

        return current_objects

    def _register_object(self, corner):
        new_id = self.next_id
        self.tracked_objects[new_id] = {"corner": corner, "age": 0}
        self.next_id += 1
        return new_id

def main(args=None):

    rclpy.init(args=args)

    lidar_object_detection = LidarObjectDetectionNode()

    rclpy.spin(lidar_object_detection)

    lidar_object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()