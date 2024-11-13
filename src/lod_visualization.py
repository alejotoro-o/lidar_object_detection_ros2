#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from lidar_object_detection_ros2.msg import ObjectsArray, ScanClusters

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.spatial.transform import Rotation as R
import threading

class PlotLidar(Node):

    def __init__(self):

        super().__init__('plot_lidar')

        ## Parameters
        self.declare_parameter("x_lims", [-5.0,5.0])
        self.declare_parameter("y_lims", [-5.0,5.0])
        self.x_lims = self.get_parameter('x_lims').get_parameter_value().double_array_value
        self.y_lims = self.get_parameter('y_lims').get_parameter_value().double_array_value

        self.declare_parameter("frame_id", "")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.declare_parameter("lidar_frame_id", "laser_link")
        self.lidar_frame_id = self.get_parameter("lidar_frame_id").get_parameter_value().string_value
        if self.frame_id == "":
            self.frame_id = self.lidar_frame_id

        ## Variables
        self.objects = ObjectsArray()
        self.ranges = []
        self.lidar_points = []
        self.labels = []

        ## Subscriptions
        self.create_subscription(ObjectsArray, "lod_objects", self.lod_objects_callback, 10)
        self.create_subscription(ScanClusters, "lod_clusters", self.lod_clusters_callback, 10)

        ## TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fig, self.ax = plt.subplots()
        self.ax.autoscale(False)
        self.ax.set_xlim(self.x_lims[0],self.x_lims[1])
        self.ax.set_ylim(self.y_lims[0],self.y_lims[1])

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
        self.ax.set_xlim(self.x_lims[0],self.x_lims[1])
        self.ax.set_ylim(self.y_lims[0],self.y_lims[1])
        self.ax.grid()
        plt.tight_layout()

        if len(self.lidar_points) > 0:

            ## Plot Frames
            arrow = matplotlib.patches.Arrow(0, 0, 0.15, 0, width=0.1, color="r")
            self.ax.add_patch(arrow)
            arrow = matplotlib.patches.Arrow(0, 0, 0, 0.15, width=0.1, color="g")
            self.ax.add_patch(arrow)

            try:
                t = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    self.lidar_frame_id,
                    rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform {self.frame_id} to {self.lidar_frame_id}: {ex}')
                return
            
            r = R.from_quat([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
            theta_r = r.as_rotvec()[-1]

            arrow = matplotlib.patches.Arrow(t.transform.translation.x, t.transform.translation.y,0.15*np.cos(theta_r),0.15*np.sin(theta_r), width=0.1, color="r")
            self.ax.add_patch(arrow)
            arrow = matplotlib.patches.Arrow(t.transform.translation.x, t.transform.translation.y, 0.15*np.cos(theta_r + np.pi/2), 0.15*np.sin(theta_r + np.pi/2), width=0.1, color="g")
            self.ax.add_patch(arrow)

            ## Plot Lidar Points
            lidar_points = np.array(self.lidar_points)
            self.ax.scatter(lidar_points[:,0], lidar_points[:,1], s=2, c="b")

            ## Plot Rectangles
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(self.objects.objects))]

            for obj, col in zip(self.objects.objects, colors):
   
                rect = matplotlib.patches.Rectangle((obj.l_shape.c1.x, obj.l_shape.c1.y), obj.l_shape.l1, obj.l_shape.l2, np.rad2deg(obj.l_shape.theta), color = col, fill = False, lw=2)
                self.ax.add_patch(rect)
                self.ax.scatter([obj.pose.x], [obj.pose.y], s=10, c="r", marker="x")

        return self.ax      

    def _plt(self):
            
        self.ani = anim.FuncAnimation(self.fig, self.update_plot, interval=100)
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