import numpy as np

class RewardFunction:
    def __init__(self, euclidean_dist: float, time_steps: int, max_episode_steps: int, collision_bool: bool):
        self.euclidean_dist = euclidean_dist
        self.time_steps = time_steps
        self.max_episode_steps = max_episode_steps
        self.collision_bool = collision_bool
        
        # Reward Coefficients
        self.done = False
        self.time_threshold = 2000
        self.sigma = euclidean_dist / 3
        self.kappa_d = 1
        self.kappa_p = self.kappa_d / 2
        self.goal_check = False
    
    def original(self, remain_dist: float):
        kappa = 50*np.sqrt(2)
        
        R_d = kappa * (self.euclidean_dist - remain_dist)
        R_t = -1
        if self.time_steps >= self.time_threshold:
            R_t = -np.exp(-0.4 * (self.time_steps - self.time_threshold) - 1)
        
        R = R_d + R_t
        if R <= -self.kappa_d or self.time_steps >= self.max_episode_steps:
            print(f"TIME OVER / REWARD -INF")
            R -= self.kappa_d
            self.done = True
        elif self.collision_bool == True:
            print(f"COLLISION OCCUR! DONE")
            R -= self.kappa_d
            self.done = True
            self.goal_check=False
        elif remain_dist <= np.sqrt(2):
            print(f"REACH GOAL! DONE")
            R += self.kappa_d
            self.done = True
            self.goal_check=True
        print(f"Reward : {R:.2f}, Dist Reward : {R_d:.2f}, Time Reward : {R_t:.2f}")
        return R, self.done,self.goal_check
    
    def optimized(self, remain_dist: float):
        beta = self.euclidean_dist / self.time_threshold
        
        ETA_d = self.kappa_d * np.exp(-((remain_dist**2) / (2 * self.sigma**2)))
        
        delta_p = (self.euclidean_dist - remain_dist) - beta * self.time_steps
        ETA_p = self.kappa_p * delta_p
        if delta_p < 0:
            ETA_p *= 2
        
        ETA_t = max(0, self.time_steps - self.time_threshold / 2) // 2
        
        R = ETA_d + ETA_p - ETA_t
        if R <= -self.kappa_d or self.time_steps >= self.max_episode_steps:
            print(f"TIME OVER or REWARD -INF")
            R -= self.kappa_d
            self.done = True
        elif self.collision_bool == True:
            print(f"COLLISION OCCUR! DONE")
            R -= self.kappa_d
            self.done = True
        elif remain_dist <= np.sqrt(2):
            print(f"REACH GOAL! DONE")
            R += self.kappa_d*100*(beta*self.time_steps)
            self.done = True
            self.goal_check=True
        print(f"Reward : {R:.2f}, Dist Reward : {ETA_d:.2f}, Progress Reward : {ETA_p:.2f}, Time Reward : -{ETA_t:.2f}")
        return R, self.done,self.goal_check