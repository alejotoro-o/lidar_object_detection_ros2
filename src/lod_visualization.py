#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from lidar_object_detection_ros2.msg import ObjectsArray, ScanClusters

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.spatial.transform import Rotation as R
import threading

class PlotLidar(Node):

    def __init__(self):

        super().__init__('plot_lidar')

        self.declare_parameter("x_lim", 5.0)
        self.declare_parameter("y_lim", 5.0)
        self.x_lim = self.get_parameter('x_lim').get_parameter_value().double_value
        self.y_lim = self.get_parameter('y_lim').get_parameter_value().double_value

        self.objects = ObjectsArray()
        self.ranges = []
        self.lidar_points = []
        self.labels = []
        self.create_subscription(ObjectsArray, "lod_objects", self.lod_objects_callback, 10)
        self.create_subscription(ScanClusters, "lod_clusters", self.lod_clusters_callback, 10)

        self.fig, self.ax = plt.subplots()
        self.ax.autoscale(False)
        self.ax.set_xlim(-self.x_lim,self.x_lim)
        self.ax.set_ylim(-self.y_lim,self.y_lim)

    def lod_clusters_callback(self, clusters_msg):

        self.lidar_points = []

        for p in clusters_msg.points:

            self.lidar_points.append([p.x, p.y])

        self.labels = clusters_msg.labels 
    
    def lod_objects_callback(self, objects_msg):

        self.objects = objects_msg

    def update_plot(self, frame):

        ## Plot Data
        self.ax.clear()
        self.ax.autoscale(False)
        self.ax.set_xlim(-self.x_lim,self.x_lim)
        self.ax.set_ylim(-self.y_lim,self.y_lim)

        if len(self.lidar_points) > 0:

            self.lidar_points = np.array(self.lidar_points)
            self.ax.scatter(self.lidar_points[:,0], self.lidar_points[:,1], s=2, c="b")

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(self.objects.objects))]

            for obj, col in zip(self.objects.objects, colors):
   
                rect = matplotlib.patches.Rectangle((obj.l_shape.c1.x, obj.l_shape.c1.y), obj.l_shape.l1, obj.l_shape.l2, np.rad2deg(obj.l_shape.theta), color = col, fill = False, lw=1)
                self.ax.add_patch(rect)
                self.ax.scatter([obj.pose.x], [obj.pose.y], s=10, c="r", marker="x")

        return self.ax      

    def _plt(self):
            
        self.ani = anim.FuncAnimation(self.fig, self.update_plot, interval=10)
        plt.show()

        

def main(args=None):

    rclpy.init(args=args)

    plot_lidar = PlotLidar()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(plot_lidar)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    plot_lidar._plt()

if __name__ == '__main__':
    main()