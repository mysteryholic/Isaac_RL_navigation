import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import threading, itertools, math, torch, argparse, datetime, os
from TD_CBAM_test import TD3
from torch.utils.tensorboard import SummaryWriter
from replay_memory_test import ReplayMemory
from reward_test import RewardFunction
from lidar_preprocessing import LidarPreprocessing
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from isaacsim import SimulationApp
from potential import PotentialFieldForce, PotentialFieldEnergy

simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils import stage as stage_utils
from omni.isaac.core.objects import DynamicCylinder
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema, Sdf

scripts_path = os.path.abspath(os.path.dirname(__file__))
pkg_path = os.path.dirname(scripts_path)
usd_file_path = os.path.join(pkg_path, "usd/carter.usd")

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
world = World(stage_units_in_meters=1.0, physics_dt=1/500, rendering_dt=1/60)
omni.usd.get_context().open_stage(usd_file_path)

simulation_app.update()
while stage_utils.is_stage_loading():
    simulation_app.update()

simulation_context = SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()


class IsaacEnvROS2(Node):
    def __init__(self, size, angle_bins: int, z_bins: int):
        super().__init__('env')
        self.size = size
        
        # Basic Parameters Settings
        self.cmd_vel = Twist()
        self.pose = np.zeros(3, dtype=np.float32)
        self.start = np.zeros(2, dtype=np.float32)
        self.goal = np.array([self.size, self.size], dtype=np.float32)
        self.euclidean_dist = np.linalg.norm(self.goal - self.start)
        # self.max_vel = 1.0
        self.min_vel = 0.5
        self.max_omega = math.pi / 4
        self.COLLISION_THRESHOLD = 0.5
        self.collision_bool = False
        self.done = False
        self.remain_dist = self.euclidean_dist
        self.local_min_dist = self.euclidean_dist
        
        # TD3 Network Settings
        self.voxel_data = torch.from_numpy(np.zeros((angle_bins, z_bins, 3), dtype=np.float32))
        self.goal_state = torch.from_numpy(self.goal - self.pose[:2])
        self.next_state = (self.voxel_data, self.goal_state)
        self.next_state_add = self.goal - self.pose[:2]
        
        # ROS2 Basic Settings
        self.qos = QoSProfile(depth=10, reliability = ReliabilityPolicy.RELIABLE, durability = DurabilityPolicy.VOLATILE)
        self.lidar_subscription = self.create_subscription(Float32MultiArray, '/voxel_data', self.lidar_cb, self.qos)
        self.local_min_dist_subscription = self.create_subscription(Float32MultiArray, '/local_min_dist', self.min_dist_cb, self.qos)
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', self.qos)
        self.publisher_cylinder_coords = self.create_publisher(Float32MultiArray, '/cylinder_coords', self.qos)
        self.publisher_robot_pose = self.create_publisher(Float32MultiArray, '/robot_pose', self.qos)
        
        self.create_timer(0.01, self.cmd_cb)
        self.create_timer(0.01, self.robot_pose)

    def robot_pose(self):
        try:
            robot_prim_path = "/World/Nova_Carter_ROS/chassis_link"
            # robot_prim_path = "/World/mobile_manipulator/base_link/base_body"
            robot_prim = world.stage.GetPrimAtPath(robot_prim_path)
            if robot_prim.IsValid():
                xform = UsdGeom.Xformable(robot_prim)
                transform_matrix = xform.GetLocalTransformation()
                translation = transform_matrix.ExtractTranslation()
                rotation = transform_matrix.ExtractRotation()
                _, _, yaw = self.axis_angle_to_euler(rotation)
                
                self.pose = np.array([translation[0], translation[1], yaw], dtype=np.float32)
                self.remain_dist = np.linalg.norm(self.goal - self.pose[:2])
                self.publisher_robot_pose.publish(Float32MultiArray(data=[self.pose[0], self.pose[1], yaw]))
            else:
                print(f"Error: Robot prim not found at {robot_prim_path}")
        except Exception as e:
            print(f"Failed to update robot pose from sim: {str(e)}")

    def axis_angle_to_euler(self, rotation):
        axis = rotation.axis
        angle = rotation.angle
        
        w = np.cos(angle/2)
        x = axis[0] * np.sin(angle/2)
        y = axis[1] * np.sin(angle/2)
        z = axis[2] * np.sin(angle/2)
        
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        # Normalize angles to be within -pi to pi
        roll = math.atan2(math.sin(roll), math.cos(roll))
        pitch = math.atan2(math.sin(pitch), math.cos(pitch))
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        
        return roll, pitch, yaw

    def cmd_cb(self):
        self.publisher_cmd_vel.publish(self.cmd_vel)

    def min_dist_cb(self, msg):
        self.local_min_dist = msg.data[0]

    def lidar_cb(self, msg):
        self.angle_bins = msg.layout.dim[0].size
        self.z_bins = msg.layout.dim[1].size
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.angle_bin_size = (self.angle_max - self.angle_min) / self.angle_bins
        self.z_min = -0.2
        self.z_max = 2.8
        self.z_bin_size = (self.z_max - self.z_min) / self.z_bins
        self.voxel_data = torch.from_numpy(np.array(msg.data, dtype=np.float32).reshape(self.angle_bins, self.z_bins, 3))
        self.next_state = self.voxel_data
    
    def parameters_reset(self):
        self.done = False
        self.collision_bool = False
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.local_min_dist = 10.0
    
    def reset(self):
        simulation_context.reset()
        self.parameters_reset()
        obstacle = Obstacle(self.size)
        cylinder_coords = obstacle.three_static()
        self.next_state_add = self.goal - self.pose[:2]
        self.publish_cylinder_coords(cylinder_coords)
        return self.next_state, self.next_state_add
    
    def publish_cylinder_coords(self, cylinder_coords):
        try:
            msg = Float32MultiArray()
            data = []
            for coords in cylinder_coords:
                data.extend(coords[:2])
            msg.data = data
            self.publisher_cylinder_coords.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing cylinder coordinates: {str(e)}")
    
    def step(self, action, time_steps, max_episode_steps):
        self.done = False
        
        omega = action[0]
        if np.abs(action[0]) > self.max_omega:
            omega = np.clip(action[0], -self.max_omega, self.max_omega)
        
        self.cmd_vel.linear.x = self.min_vel
        self.cmd_vel.angular.z = float(omega)
        
        simulation_context.step(render=True)
        
        self.next_state_add = self.goal - self.pose[:2]
        
        if self.local_min_dist > self.COLLISION_THRESHOLD:
            self.collision_bool = False
        elif 0.2 < self.local_min_dist < self.COLLISION_THRESHOLD:
            self.collision_bool = True
        reward_function = RewardFunction(self.euclidean_dist, time_steps, max_episode_steps, self.collision_bool)
        reward, self.done,self.goal_check = reward_function.optimized(self.remain_dist)
        return self.next_state, self.next_state_add, reward, self.done,self.goal_check

class Obstacle():
    def __init__(self, size):
        self.size = size
        self.wall_height = 3.0
        self.wall_thickness = 1.0
        self.bias = 2
        self.cylinder_radius = self.size / 50
        self.cylinder_height = 3.0
    
    def create_wall(self):
        wall_configs = {
            "bottom": ([ -self.bias + self.wall_thickness / 2, self.size / 2, self.wall_height / 2], 
                       [self.wall_thickness, self.size + 2 * self.bias, self.wall_height]),
            "top": ([self.size + self.bias - self.wall_thickness / 2, self.size / 2, self.wall_height / 2], 
                    [self.wall_thickness, self.size + 2 * self.bias, self.wall_height]),
            "left": ([self.size / 2, self.size + self.bias - self.wall_thickness / 2, self.wall_height / 2], 
                     [self.size + self.bias, self.wall_thickness, self.wall_height]),
            "right": ([self.size / 2, -self.bias + self.wall_thickness / 2, self.wall_height / 2], 
                      [self.size + self.bias, self.wall_thickness, self.wall_height])
        }
        
        for wall_name, (position, scale) in wall_configs.items():
            wall_prim_path = f"/World/Wall/{wall_name}_wall"
            wall_xform = XFormPrim(wall_prim_path)
            cube_geom = UsdGeom.Cube.Define(world.stage, f"{wall_prim_path}/Cube")
            cube_geom.GetSizeAttr().Set(1.0)
            UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(wall_prim_path))
            UsdPhysics.CollisionAPI.Apply(world.stage.GetPrimAtPath(wall_prim_path))
            wall_xform.set_world_pose(position)
            wall_xform.set_local_scale(scale)
    
    def create_cylinder(self, prim_path, coords):
        cylinder_xform = XFormPrim(prim_path)
        cylinder_geom = UsdGeom.Cylinder.Define(world.stage, f"{prim_path}")
        cylinder_geom.GetRadiusAttr().Set(self.cylinder_radius)
        cylinder_geom.GetHeightAttr().Set(self.cylinder_height)
        UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(prim_path))
        UsdPhysics.CollisionAPI.Apply(world.stage.GetPrimAtPath(prim_path))
        cylinder_xform.set_world_pose(coords, [0, 0, 0, 1])
    
    def three_static(self):
        self.create_wall()
        cylinder_coords = [
            [1.0, 4.0, self.cylinder_height / 2],
            [2.5, 2.5, self.cylinder_height / 2],
            [4.0, 1.0, self.cylinder_height / 2]
        ]
        for i, coords in enumerate(cylinder_coords):
            self.create_cylinder(f"/World/Obstacles/Cylinder_{i+1}", coords)
        return cylinder_coords
    
    def two_random(self):
        self.create_wall()
        cylinder_coords = []
        one_side_parts = 4
        region_size = self.size / one_side_parts
        obstacle_num = 2
        for i in range(obstacle_num):
            coords = [
                np.random.uniform(0, self.size),
                np.random.uniform(region_size, self.size - region_size),
                self.cylinder_height / 2
            ]
            self.create_cylinder(f"/World/Obstacles/Cylinder_{i}", coords)
            cylinder_coords.append(coords)
        return cylinder_coords

if __name__ == '__main__':
    expl_decay_steps =(180000)
    expl_noise=1
    expl_min=0.1
    rclpy.init(args=None)
    parser = argparse.ArgumentParser(description='TD3 Args')
    parser.add_argument('--env-name', default="obstacle_avoidance", help='quadruped_isaac')
    parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--policy_freq', type=int, default=2, metavar='G', help='policy frequency for TD3 updates')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=12001, metavar='N', help='maximum number of steps (default: 5000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=500000, metavar='N',help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=400000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--cuda', action="store_true", default=True, help='run on CUDA (default: False)')
    args = parser.parse_args()

    size = 5.0
    angle_bins, z_bins = 180, 10
    lidar_preprocessing = LidarPreprocessing(angle_bins, z_bins)
    env = IsaacEnvROS2(size, angle_bins, z_bins)

    executor = MultiThreadedExecutor()
    executor.add_node(lidar_preprocessing)
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = env.create_rate(2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    action_space = 1
    env.get_logger().info(f"state_space:{len(env.next_state)}")
    env.get_logger().info(f"action_space:{action_space}")
    env.get_logger().info(f"args : {args}")
    input_shape = (angle_bins, z_bins, 3)
    agent = TD3(input_shape, len(env.next_state), action_space, args)
    
    file_name = "checkpoints"
    try:
        agent.load_checkpoint(pkg_path, file_name)
        env.get_logger().info("Checkpoint loaded successfully.")
    except:
        env.get_logger().info("No checkpoint found, start training from scratch.")

    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%m%d_%H-%M")}')

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)
    memory.clear()
    total_numsteps = 0
    max_episode_steps = 10000000
    
    try:
        while rclpy.ok():
            for i_episode in itertools.count(1):
                episode_reward = 0
                episode_steps = 0
                done = False
                share = 10000
                
                state, state_add = env.reset()
                print(f"env.pose:{env.pose}, env.goal:{env.goal}")
                while not done:
                    print(f"episode step:{episode_steps}, total steps:{total_numsteps},i_episode:{i_episode}")
                    if expl_noise>expl_min:
                        expl_noise=expl_noise-((1-expl_noise)/expl_decay_steps)
                    if total_numsteps<10:
                        action=np.random.uniform(-math.pi/4, math.pi/4, size=action_space)
                    else:
                        action=agent.select_action(state, state_add)
                        noise=np.random.normal(-0.2, 0.2, size=action_space)
                        action = (action+noise).clip(-math.pi/4, math.pi/4)
                    
                    if len(memory) > args.batch_size:
                        av_critic_loss, av_Q, max_Q = agent.update_parameters(memory, args)

                        writer.add_scalar('av_critic_loss', av_critic_loss, total_numsteps)
                        writer.add_scalar('av_Q', av_Q, total_numsteps)
                        writer.add_scalar('max_Q', max_Q, total_numsteps)
                        
                    next_state, next_state_add, reward, done,goal_check = env.step(action, episode_steps, max_episode_steps)
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    mask = 1 if episode_steps == max_episode_steps else float(not done)
                    memory.push(state, state_add, action, reward, next_state, next_state_add, mask,goal_check)

                    state = next_state
                    state_add = next_state_add

                writer.add_scalar('reward/train', episode_reward, i_episode)
                print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

                if i_episode % 10 == 0 and args.eval is True:
                    avg_reward = 0.0
                    episodes = 10
                    for i  in range(episodes):
                        print(f"eval episode{i}")
                        state, state_add = env.reset()
                        episode_reward = 0
                        eval_steps = 0
                        done = False
                        while not done:
                            action = agent.select_action(state, state_add)
                            next_state, next_state_add, reward, done,goal_check = env.step(action, eval_steps, max_episode_steps)
                            episode_reward += reward

                            eval_steps += 1
                            state = next_state
                        avg_reward += episode_reward
                    avg_reward /= episodes

                    writer.add_scalar('avg_reward/test', avg_reward, i_episode)

                    print("--------------------------------------------------------------------------------")
                    print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}")
                    print("--------------------------------------------------------------------------------")
                    
                if i_episode % 20 == 0:
                    agent.save_checkpoint(pkg_path, file_name, i_episode)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
