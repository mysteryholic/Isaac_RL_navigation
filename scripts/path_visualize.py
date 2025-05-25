import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from collections import deque
from std_msgs.msg import Float32MultiArray

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')
        self.maxlen = 100000
        self.odom_path = deque(maxlen=self.maxlen)
        self.window_name = 'Path Visualization'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.subscription = self.create_subscription(Float32MultiArray, '/robot_pose', self.pose_cb, 10)
        self.cylinder_sub = self.create_subscription(Float32MultiArray, '/cylinder_coords', self.cylinder_cb, 10)
        
        self.cylinder_coords = []
        self.pose = [0.0, 0.0]
        self.canvas_size = 1400
        self.canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        self.map_size = 1000
        self.width = self.height = (self.canvas_size + self.map_size) // 2
        self.scale = 200
    
    def pose_cb(self, msg):
        self.pose = msg.data
        self.visualize_path()
    
    def cylinder_cb(self, msg):
        data = msg.data
        coords = []
        for i in range(0, len(data), 2):
            coords.append([data[i], data[i+1]])
        self.cylinder_coords = coords
    
    def visualize_path(self):    
        self.odom_path.append([self.pose[0], self.pose[1]])
        
        if len(self.odom_path) >= 2:
            current_pos = np.array(self.odom_path[-1])
            last_pos = np.array(self.odom_path[-2])
            dt_dist = np.linalg.norm(current_pos - last_pos)
            if dt_dist >= 0.3:
                new_start = self.odom_path[-1]
                self.odom_path = deque([new_start], maxlen=self.maxlen)
        
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        for i in range(len(self.odom_path) - 1):
            start_x = int(self.width - self.odom_path[i][0] * self.scale)
            start_y = int(self.height - self.odom_path[i][1] * self.scale)
            end_x = int(self.width - self.odom_path[i+1][0] * self.scale)
            end_y = int(self.height - self.odom_path[i+1][1] * self.scale)
            cv2.line(canvas, (start_y, start_x), (end_y, end_x), (0, 255, 0), 2)
        
        current_x = int(self.width - self.pose[0] * self.scale)
        current_y = int(self.height - self.pose[1] * self.scale)
        cv2.circle(canvas, (current_y, current_x), int(0.33 * self.scale), (0, 0, 255), 1)
        
        for coord in self.cylinder_coords:
            px = int(self.width - coord[0] * self.scale)
            py = int(self.height - coord[1] * self.scale)
            radius = int(0.1 * self.scale)
            cv2.circle(canvas, (py, px), radius, (255, 0, 0), 2)
        
        goal_x = goal_y = int(self.canvas_size - self.width)
        cv2.rectangle(canvas, (int(goal_x - self.scale/2), int(goal_y - self.scale/2)), (int(goal_x + self.scale/2), int(goal_y + self.scale/2)), (0, 0, 255), 3)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

if __name__ == '__main__':
    rclpy.init(args=None)
    path_visualizer = PathVisualizer()
    rclpy.spin(path_visualizer)
    path_visualizer.destroy_node()
    rclpy.shutdown()