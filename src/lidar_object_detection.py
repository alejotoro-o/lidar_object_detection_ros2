#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from lidar_object_detection_ros2.msg import Object, ObjectsArray

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

class LidarObjectDetectionNode(Node):

    def __init__(self):

        super().__init__('lidar_object_detection')

        ## Parameters
        self.declare_parameter("dbscan_eps", 0.1)
        dbscan_esp = self.get_parameter("dbscan_eps").get_parameter_value().double_value

        self.declare_parameter("dbscan_min_samples", 5)
        dbscan_min_samples = self.get_parameter("dbscan_min_samples").get_parameter_value().double_value

        self.declare_parameter("frame_id", "")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        self.declare_parameter("lidar_frame_id", "laser_link")
        self.lidar_frame_id = self.get_parameter("lidar_frame_id").get_parameter_value().string_value

        self.declare_parameter("lidar_angular_resolution", 0.00872665)
        self.lidar_ang_res = self.get_parameter('lidar_angular_resolution').get_parameter_value().double_value

        self.declare_parameter("update_rate", 0.1)
        update_rate = self.get_parameter('update_rate').get_parameter_value().double_value

        ## Variables
        self.ranges = []
        self.dbscan = DBSCAN(eps=dbscan_esp, min_samples=dbscan_min_samples)

        ## TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ## Subscriptions
        self.create_subscription(LaserScan, "scan", self.scan_callback, 10)

        ## Publishers
        self.objects_publisher = self.create_publisher(ObjectsArray, "lod_objects", 10)

        ## Timer
        self.timer = self.create_timer(update_rate, self.on_timer)
        
    def scan_callback(self, scan_msg):
        
        self.ranges = []

        for range in scan_msg.ranges:       
            if range != float("+inf"):
                self.ranges.append(range)

    def on_timer(self):
        pass

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

def main(args=None):

    rclpy.init(args=args)

    lidar_object_detection = LidarObjectDetectionNode()

    rclpy.spin(lidar_object_detection)

    lidar_object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()