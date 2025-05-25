import rclpy
from rclpy.node import Node
import numpy as np
import math, threading
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from numpy.lib import recfunctions as rfn

class LidarPreprocessing(Node):
    def __init__(self, angle_bins, z_bins,division,num_selectpoints):
        super().__init__('sensor_preprocessing')
        self.angle_bins = angle_bins
        self.z_bins = z_bins
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.angle_bin_size = (self.angle_max - self.angle_min) / self.angle_bins
        self.z_min = -0.2
        self.z_max = 2.8
        self.z_bin_size = (self.z_max - self.z_min) / self.z_bins
        self.division = division
        self.num_selectpoints = num_selectpoints
        
        self.local_min_dist = 10.0
        self.empty_voxel = np.zeros((self.angle_bins, self.z_bins, 3), dtype=np.float32)
        self.local_thresh = 3
        
        # ROS2 Basic Settings
        self.qos = QoSProfile(depth=10, reliability = ReliabilityPolicy.RELIABLE, durability = DurabilityPolicy.VOLATILE)
        self.lidar_subscription = self.create_subscription(PointCloud2, '/lidar/pointcloud', self.lidar_cb, self.qos)
        self.voxel_data_publisher = self.create_publisher(Float32MultiArray, '/voxel_data', self.qos)
        self.local_min_dist_publisher = self.create_publisher(Float32MultiArray, '/local_min_dist', self.qos)
    
    def lidar_cb(self, msg):
        try:
            raw_data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))
            if len(raw_data) == 0:
                # return self.empty_voxel, self.local_min_dist
                return
            raw_data = np.array(raw_data)
            if raw_data.dtype.names is not None:
                raw_data = rfn.structured_to_unstructured(raw_data).astype(np.float32)
                # raw_data = np.stack([raw_data[name].astype(np.float32) for name in raw_data.dtype.names], axis=-1)
            raw_data = raw_data.astype(np.float32)
            voxel_data, local_min_dist = self.lidar_preprocessing(raw_data)
            if voxel_data is None:
                return
                # return self.empty_voxel, local_min_dist
            self.publish_voxel_data(voxel_data)
            self.publish_local_min_dist(local_min_dist)
        except Exception as e:
            self.get_logger().error(f"Error converting raw data: {str(e)}")
    
    def lidar_preprocessing(self, raw_data):
        max_range = local_min_dist = 10.0
        
        valid_filter_mask_1 = (np.isfinite(raw_data).all(axis=1) & (self.z_min <= raw_data[:, 2]) & (raw_data[:, 2] <= self.z_max))
        filtered_points = raw_data[valid_filter_mask_1]
        
        if filtered_points.size == 0:
            return np.zeros((self.angle_bins, self.z_bins, 3), dtype=np.float32), local_min_dist
        
        angles = np.arctan2(filtered_points[:, 1], filtered_points[:, 0])
        dist = np.sqrt(filtered_points[:, 0]**2 + filtered_points[:, 1]**2 + filtered_points[:, 2]**2)
        
        valid_filter_mask_2 = (self.angle_min <= angles) & (angles <= self.angle_max)
        filtered_points = filtered_points[valid_filter_mask_2]
        angles = angles[valid_filter_mask_2]
        dist = dist[valid_filter_mask_2]
        
        if dist.size == 0:
            return self.empty_voxel, local_min_dist
        local_min_dist = np.min(dist)
        
        angles_indices = np.clip((angles - self.angle_min) / self.angle_bin_size, 0, self.angle_bins - 1).astype(np.int32)
        z_indices = np.clip((filtered_points[:, 2] - self.z_min) / self.z_bin_size, 0, self.z_bins - 1).astype(np.int32)
        
        voxel_sum = self.empty_voxel.copy()
        voxelized_points = self.empty_voxel.copy()
        voxel_count = np.zeros((self.angle_bins, self.z_bins), dtype=np.int32)
        new_voxel=np.zeros((self.angle_bins, self.z_bins), dtype=np.int32)
        
        # np.add.at(voxel_sum, (angles_indices, z_indices), filtered_points)
        # np.add.at(voxel_count, (angles_indices, z_indices), 1)
        
        # has_data = voxel_count > 0
        
        # voxelized_points[has_data] = voxel_sum[has_data] / voxel_count[has_data][:, np.newaxis]

         # Initialize a matrix to store the minimum distance for each voxel
        voxel_min_dist = np.full((self.angle_bins, self.z_bins), np.inf, dtype=np.float32)
        voxel_min_points = np.zeros_like(voxel_sum)

        # Calculate distances for each point
        point_distances = np.linalg.norm(filtered_points, axis=1)

        # For each point, update the voxel if this point has a smaller distance
        for i in range(len(filtered_points)):
            a_idx, z_idx = angles_indices[i], z_indices[i]
            distance = point_distances[i]
            if distance < voxel_min_dist[a_idx, z_idx]:
                voxel_min_dist[a_idx, z_idx] = distance
                voxel_min_points[a_idx, z_idx] = filtered_points[i]

        # Update voxel count to track which voxels have data
        np.add.at(voxel_count, (angles_indices, z_indices), 1)
        has_data = voxel_count > 0

        # Set the min points to the voxelized_points
        voxelized_points[has_data] = voxel_min_points[has_data]
        
        if np.any(~has_data):
            angle_centers = self.angle_min + (np.arange(self.angle_bins) + 0.5) * self.angle_bin_size
            z_centers = self.z_min + (np.arange(self.z_bins) + 0.5) * self.z_bin_size
            
            angle_grid, z_grid = np.meshgrid(angle_centers, z_centers, indexing='ij')
            empty_mask = ~has_data
            voxelized_points[empty_mask, 0] = max_range * np.cos(angle_grid[empty_mask])
            voxelized_points[empty_mask, 1] = max_range * np.sin(angle_grid[empty_mask])
            voxelized_points[empty_mask, 2] = z_grid[empty_mask]
            # Calculate the Euclidean distance in the XY plane for all voxels
            new_voxel = np.sqrt(voxelized_points[..., 0]**2 + voxelized_points[..., 1]**2)
        new_voxel[has_data] = np.linalg.norm(voxelized_points[has_data, :2], axis=1)
        xy_distance = np.linalg.norm(voxelized_points[..., :2], axis=2)
        mask = xy_distance > self.local_thresh
        voxelized_points[mask] = np.array([5.1,5.1,5.1])
        # Divide new_voxel into 3 equal sections and select 10 minimum distance points from each
        section_size = self.angle_bins // self.division
        selected_points = []

        for section in range(self.division):
            # Calculate section boundaries
            start_idx = section * section_size
            end_idx = (section + 1) * section_size if section < self.division - 1 else self.angle_bins
            
            # Extract section data
            section_voxel = new_voxel[start_idx:end_idx, :]
            
            # Find valid points (non-zero and finite distance)
            valid_mask = (section_voxel > 0) & np.isfinite(section_voxel)
            
            if not np.any(valid_mask):
                continue  # Skip this section if no valid points
            
            # Get valid distances and their indices
            valid_indices = np.where(valid_mask)
            valid_distances = section_voxel[valid_mask]
            
            # Sort by distance and select up to 10 points
            sorted_indices = np.argsort(valid_distances)[:self.num_selectpoints]
            
            for idx in sorted_indices:
                # Get the original indices in the full voxel space
                a_idx = valid_indices[0][idx] + start_idx
                z_idx = valid_indices[1][idx]
                
                # Get the corresponding x, y coordinates
                x = voxelized_points[a_idx, z_idx, 0]
                y = voxelized_points[a_idx, z_idx, 1]
                selected_points.append((x, y))

        # Store the selected points
        self.selected_points = np.array(selected_points)
        return self.selected_points, local_min_dist

    def publish_voxel_data(self, voxel_data):
        try:
            msg = Float32MultiArray()
            msg.layout.dim = [
                MultiArrayDimension(label = "angle_bins", size = self.angle_bins, stride = 3 * self.angle_bins * self.z_bins),
                MultiArrayDimension(label = "z_bins", size = self.z_bins, stride = 3 * self.z_bins),
                MultiArrayDimension(label = "xyz", size = 3, stride = 3)
            ]
            msg.data = voxel_data.flatten().tolist()
            self.voxel_data_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error in publishing voxel data: {str(e)}")
    
    def publish_local_min_dist(self, local_min_dist):
        try:
            msg = Float32MultiArray()
            msg.data = [float(local_min_dist)]
            self.local_min_dist_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error in publishing local min dist: {str(e)}")

if __name__ == '__main__':
    rclpy.init()
    lidar_node = LidarPreprocessing(angle_bins=180, z_bins=10)
    try:
        rclpy.spin(lidar_node)
    except KeyboardInterrupt:
        lidar_node.get_logger().info("Keyboard Interrupt")
    finally:
        lidar_node.destroy_node()
        rclpy.shutdown()