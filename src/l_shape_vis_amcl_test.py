import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

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

        self.declare_parameter("lidar_angular_resolution", 0.00872665)
        self.lidar_ang_res = self.get_parameter('lidar_angular_resolution').get_parameter_value().double_value
        self.declare_parameter("x_lim", 5.0)
        self.declare_parameter("y_lim", 5.0)
        self.x_lim = self.get_parameter('x_lim').get_parameter_value().double_value
        self.y_lim = self.get_parameter('y_lim').get_parameter_value().double_value

        self.pose = [0,0,0]
        self.ranges = []
        self.create_subscription(PoseWithCovarianceStamped, "amcl_pose", self.pose_callback, 10)
        self.create_subscription(LaserScan, "scan", self.scan_callback, 10)

        self.fig, self.ax = plt.subplots()

    def pose_callback(self, pose_msg):

        r = R.from_quat([pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w])
        theta = r.as_rotvec()[-1]

        self.pose = [pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    theta]
        
    def scan_callback(self, scan_msg):
        
        self.ranges = []

        for range in scan_msg.ranges:       
            if range != float("+inf"):
                self.ranges.append(range)

    def update_plot(self, frame):

        current_lidar_angle = 0
        points_x = []
        points_y = []

        for range in self.ranges:

            point_x = self.pose[0] + range*np.cos(self.pose[2] + current_lidar_angle)
            point_y = self.pose[1] + range*np.sin(self.pose[2] + current_lidar_angle)

            points_x.append(point_x)
            points_y.append(point_y)
            
            current_lidar_angle += self.lidar_ang_res
        
        points_x = np.array(points_x).reshape((-1,1))
        points_y = np.array(points_y).reshape((-1,1))

        if points_x.shape[0] > 0 and points_y.shape[0] > 0:

            ## Clustering
            lidar_data = np.concatenate((points_x,points_y),axis=1)

            dbscan = DBSCAN(eps=0.1, min_samples=5)
            labels = dbscan.fit_predict(lidar_data)

            ## Plot Data
            self.ax.clear()
            self.ax.autoscale(False)
            self.ax.set_xlim(-self.x_lim,self.x_lim)
            self.ax.set_ylim(-self.y_lim,self.y_lim)

            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True

            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = labels == k            

                xy = lidar_data[class_member_mask & core_samples_mask]
                self.ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=2,
                    c=np.array([[col]])
                )

                ## Obtain L-Shapes
                if k != -1 and xy.shape[0] > 20:
                    angle, c, pos = self.cal_l_shape(xy)
                    rect = matplotlib.patches.Rectangle(pos[0], c[2] - c[0], c[3] - c[1], np.rad2deg(angle), color = "b", fill = False)
                    self.ax.add_patch(rect)

                    ## Rectangle center
                    x_cent = pos[0][0] + ((c[2] - c[0])*np.cos(angle) - (c[3] - c[1])*np.sin(angle))/2
                    y_cent = pos[0][1] + ((c[2] - c[0])*np.sin(angle) + (c[3] - c[1])*np.cos(angle))/2
                    self.ax.scatter([x_cent], [y_cent], s=10, c="r", marker="x")

                xy = lidar_data[class_member_mask & ~core_samples_mask]
                self.ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=2,
                    c=np.array([[col]])
                )

            self.ax.scatter(self.pose[0],self.pose[1],s=14,c='r')       

        return self.ax
    
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
        angle_step = 0.0174533

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

        a3 = np.cos(theta_star)
        b3 = np.sin(theta_star)

        a4 = -np.sin(theta_star)
        b4 = np.cos(theta_star)

        x1 = (b2*c1 - b1*c2)/(b2*a1 - b1*a2)
        y1 = (c2 - a2*x1)/b2

        x2 = (b4*c1 - b1*c4)/(b4*a1 - b1*a4)
        y2 = (c4 - a4*x2)/b4

        x3 = (b4*c3 - b3*c4)/(b4*a3 - b3*a4)
        y3 = (c4 - a4*x3)/b4

        x4 = (b2*c3 - b3*c2)/(b2*a3 - b3*a2)
        y4 = (c2 - a2*x4)/b2

        return theta_star, [c1,c2,c3,c4], [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    def separate_clusters(self, points, labels):

        unique_labels = set(labels)
        clusters = []

        for l in unique_labels:

            indices = np.where(labels == l)

            cluster = points[indices[0]]
            clusters.append(cluster)

        return clusters
            

    def _plt(self):
            
        self.ani = anim.FuncAnimation(self.fig, self.update_plot, interval=10)
        plt.show()

        

def main(args=None):

    rclpy.init(args=args)

    plot_lidar = PlotLidar()

    # rclpy.spin(plot_lidar)

    # plot_lidar.destroy_node()
    # rclpy.shutdown()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(plot_lidar)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    plot_lidar._plt()

if __name__ == '__main__':
    main()