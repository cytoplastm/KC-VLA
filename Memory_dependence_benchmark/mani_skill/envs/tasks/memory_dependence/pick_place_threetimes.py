from typing import Any, Dict, Union, List

import numpy as np
import sapien
import torch
import torch.random

from mani_skill.agents.robots import Panda, Fetch, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

# Standard configuration import pattern
try:
    from .pick_place_threetimes_cfgs import PICK_PLACE_THREETIMES_CONFIGS
except ImportError:
    from mani_skill.envs.tasks.memory_dependence.pick_place_threetimes_cfgs import PICK_PLACE_THREETIMES_CONFIGS

@register_env("PickPlaceThreetimes-v1", max_episode_steps=1000)
class PickPlaceThreetimesEnv(BaseEnv):
    """
    Task: Pick Red -> Place -> Pick Green -> Place -> Pick Blue -> Place.
    Strict order enforcement based on lift height > 0.1m.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch", "xarm6_robotiq"]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    
    # Type hinting for variables loaded from config
    cube_half_size: float
    sensor_cam_eye_pos: list
    sensor_cam_target_pos: list
    human_cam_eye_pos: list
    human_cam_target_pos: list
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # 1. Select configuration based on robot_uids
        if robot_uids in PICK_PLACE_THREETIMES_CONFIGS:
            cfg = PICK_PLACE_THREETIMES_CONFIGS[robot_uids]
        else:
            # Fallback to panda if specific robot config is missing, but warn about it
            print(f"Warning: Configuration for {robot_uids} not found, using 'panda' default.")
            cfg = PICK_PLACE_THREETIMES_CONFIGS["panda"]

        # 2. Strict parameter loading (will raise KeyError if config is missing)
        # Using .get(key, default) is removed to ensure configs are managed in the _cfgs.py file
        self.cube_half_size = cfg.get("cube_half_size", 0.02) # Optional: keep default for physics props or move to cfg
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.table = self.table_scene.table
        
        self.cube_red = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube_red", initial_pose=sapien.Pose(p=[0,0,0.5]))
        self.cube_green = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cube_green", initial_pose=sapien.Pose(p=[0,0,0.5]))
        self.cube_blue = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[0, 0, 1, 1], name="cube_blue", initial_pose=sapien.Pose(p=[0,0,0.5]))

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # 1. Initialize Robot
            if self.agent is not None:
                qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0])
                qpos = torch.from_numpy(qpos).float().to(self.device)
                self.agent.reset(init_qpos=qpos.repeat(b, 1))

            # 2. Generate mutually exclusive random coordinates (Using TORCH for determinism)
            # [Fix] 使用 torch.zeros 替代 np.zeros
            final_positions = torch.zeros((b, 3, 2), device=self.device)
            
            # 由于 Rejection Sampling (while True) 在并行环境很难写
            # 这里我们在 CPU 上用 torch 的生成器来做循环 (受 torch 种子控制)
            # 或者为了保持逻辑一致性，我们可以简单的用 torch 生成随机数
            
            for i in range(b):
                while True:
                    # [Fix] 将 np.random.uniform 替换为 torch.rand
                    # torch.rand 生成 [0, 1]，需要缩放
                    # r: [0, 0.2] -> torch.rand(3) * 0.2
                    r = torch.rand(3, device=self.device) * 0.2
                    
                    # theta: [0, 2*pi] -> torch.rand(3) * 2 * np.pi
                    theta = torch.rand(3, device=self.device) * 2 * np.pi
                    
                    # 计算坐标: x = r * cos - 0.1, y = r * sin
                    x = r * torch.cos(theta) - 0.1
                    y = r * torch.sin(theta)
                    points = torch.stack([x, y], dim=1) # shape (3, 2)
                    
                    # 计算距离
                    d12 = torch.norm(points[0] - points[1])
                    d13 = torch.norm(points[0] - points[2])
                    d23 = torch.norm(points[1] - points[2])
                    
                    if d12 > 0.1 and d13 > 0.1 and d23 > 0.1:
                        final_positions[i] = points
                        break
            
            # final_positions 已经是 torch tensor 了，不需要再转换
            positions = final_positions
            z_height = torch.tensor([self.cube_half_size], device=self.device).repeat(b, 1)
            default_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device).repeat(b, 1)

            pos_red = torch.cat([positions[:, 0, :], z_height], dim=1)
            pos_green = torch.cat([positions[:, 1, :], z_height], dim=1)
            pos_blue = torch.cat([positions[:, 2, :], z_height], dim=1)

            self.cube_red.set_pose(Pose.create_from_pq(pos_red, default_quat))
            self.cube_green.set_pose(Pose.create_from_pq(pos_green, default_quat))
            self.cube_blue.set_pose(Pose.create_from_pq(pos_blue, default_quat))

            self.target_pos_red = pos_red.clone()
            self.target_pos_green = pos_green.clone()
            self.target_pos_blue = pos_blue.clone()

            # --- 3. [Key] Initialize State Machine ---
            self.task_stage = torch.zeros(b, dtype=torch.int, device=self.device)
            self.has_failed = torch.zeros(b, dtype=torch.bool, device=self.device)

    def evaluate(self):
        # 1. Get Height (Check if lifted)
        # Threshold is 10cm (0.1m)
        z_red = self.cube_red.pose.p[:, 2]
        z_green = self.cube_green.pose.p[:, 2]
        z_blue = self.cube_blue.pose.p[:, 2]
        
        is_red_lifted = z_red > 0.15
        is_green_lifted = z_green > 0.15
        is_blue_lifted = z_blue > 0.15

        # 2. State Machine Update (Vectorized)
        
        # --- Check Failure Conditions ---
        # Stage 0 (Target Red): Green or Blue lifted -> Fail
        fail_stage_0 = (self.task_stage == 0) & (is_green_lifted | is_blue_lifted)
        # Stage 1 (Target Green): Blue lifted -> Fail (Red lifted is allowed as it might be placing)
        fail_stage_1 = (self.task_stage == 1) & (is_blue_lifted)
        
        # Update Failure State (Permanent)
        self.has_failed = self.has_failed | fail_stage_0 | fail_stage_1

        # --- Check Transitions ---
        # Only transition if not failed
        
        # Stage 0 -> 1: Red lifted
        transition_0_to_1 = (self.task_stage == 0) & is_red_lifted & (~self.has_failed)
        self.task_stage[transition_0_to_1] = 1
        
        # Stage 1 -> 2: Green lifted
        transition_1_to_2 = (self.task_stage == 1) & is_green_lifted & (~self.has_failed)
        self.task_stage[transition_1_to_2] = 2
        
        # Stage 2 -> 3: Blue lifted
        transition_2_to_3 = (self.task_stage == 2) & is_blue_lifted & (~self.has_failed)
        self.task_stage[transition_2_to_3] = 3

        # 3. Check Final Placement
        dist_red = torch.linalg.norm(self.cube_red.pose.p - self.target_pos_red, axis=1)
        dist_green = torch.linalg.norm(self.cube_green.pose.p - self.target_pos_green, axis=1)
        dist_blue = torch.linalg.norm(self.cube_blue.pose.p - self.target_pos_blue, axis=1)
        
        all_at_target = (dist_red < 0.05) & (dist_green < 0.05) & (dist_blue < 0.05)

        # 4. Determine Success
        # Must satisfy:
        # 1. Stage reached 3 (All lifted in order)
        # 2. No failure recorded (Order correct)
        # 3. All cubes back at target
        success = (self.task_stage == 3) & (~self.has_failed) & all_at_target
        
        return {
            "success": success,
            "stage": self.task_stage,
            "has_failed": self.has_failed,
            "dist_red": dist_red
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:
             obs.update(
                cube_red_pose=self.cube_red.pose.raw_pose,
                cube_green_pose=self.cube_green.pose.raw_pose,
                cube_blue_pose=self.cube_blue.pose.raw_pose,
                task_stage=self.task_stage.unsqueeze(1), # Add state to observation
            )
        return obs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(1, device=self.device)
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)