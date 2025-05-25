import numpy as np

class PotentialFieldForce:
    def __init__(self, goal, pose, voxel_data):
        self.goal = goal
        self.pose = pose[:2]
        self.yaw = pose[2]
        self.voxel_data = voxel_data
        self.k_att = 1.0
        self.k_rep = 10000000.0
        self.repulsive_radius = 0.5
        
    def compute_force(self):
        F_att = self.k_att * (self.goal - self.pose)
        
        obstacle = np.stack([
            self.voxel_data[:, :, 0].ravel(),
            self.voxel_data[:, :, 1].ravel()
        ], axis=1)
        
        diff = obstacle - self.pose[:2]
        dist = np.linalg.norm(diff, axis=1)
        mask = dist < self.repulsive_radius
        valid_diff = diff[mask]
        valid_dist = dist[mask]
        
        coef = self.k_rep * (1.0 / valid_dist - 1.0 / self.repulsive_radius) * (1.0 / valid_dist**2)
        F_rep = np.sum((valid_diff.T * coef).T, axis=0)
        
        F_total = F_att + F_rep
        # print(f"F_att: {F_att}, F_rep: {F_rep}")
        return F_total

    def compute_action(self):
        F_total = self.compute_force()
        # print(f"F_total: {F_total}")
        angle = np.arctan2(F_total[1], F_total[0]) - self.yaw
        angle = 2 * ((angle + np.pi) % (2 * np.pi) - np.pi)
        action = np.clip(angle, -np.pi/4, np.pi/4)
        return action
    
class PotentialFieldEnergy:
    def __init__(self, goal, pose, voxel_data):
        self.goal = goal
        self.pose = pose[:2]
        self.voxel_data = voxel_data
        self.k_att = 1.0
        self.k_rep = 100.0
        self.repulsive_radius = 1.0
        
    def compute_potential(self):
        dist_to_goal = np.linalg.norm(self.pose - self.goal)
        U_att = 0.5 * self.k_att * (dist_to_goal**2)

        # 2) Repulsive potential: sum over obstacles within d0
        obstacle = np.stack([
            self.voxel_data[:, :, 0].ravel(),
            self.voxel_data[:, :, 1].ravel()
        ], axis=1)
        dists = np.linalg.norm(self.pose - obstacle, axis=1)
        mask = (dists > 0) & (dists < self.repulsive_radius)
        valid_dists = dists[mask]

        U_rep = np.sum(0.5 * self.k_rep * (1.0/valid_dists - 1.0/self.repulsive_radius)**2)

        U_total = U_att + U_rep
        return U_total