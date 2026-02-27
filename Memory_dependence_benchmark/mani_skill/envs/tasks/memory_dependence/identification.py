from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random

from mani_skill.agents.robots import Panda, Fetch, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder

try:
    from .identification_cfgs import IDENTIFICATION_CONFIGS
except ImportError:
    from mani_skill.envs.tasks.memory_dependence.identification_cfgs import IDENTIFICATION_CONFIGS

@register_env("Identification-v1", max_episode_steps=400)
class IdentificationEnv(BaseEnv):
    """
    Task Description:
    Three red cubes (A, B, C) are placed with randomized positions.
    Relative order is fixed (A is +0.15y from B, C is -0.15y from B).
    B's position is randomized in x=[-0.2, -0.1], y=[-0.05, 0.05].
    A random instruction (1-6) dictates which two cubes to swap.
    The robot must perform the swap via a temporary buffer zone and then lift the center cube (Cube B).
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq", "so100", "widowxai"]
    agent: Union[Panda, Fetch, XArm6Robotiq]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in IDENTIFICATION_CONFIGS:
            cfg = IDENTIFICATION_CONFIGS[robot_uids]
        else:
            cfg = IDENTIFICATION_CONFIGS["panda"]
            
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.max_goal_height = cfg["max_goal_height"]
        
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

        # Build cubes (Initial pose here is just a placeholder, will be reset in _initialize_episode)
        self.cubeA = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeA", 
            initial_pose=sapien.Pose(p=[-0.2, 0.15, self.cube_half_size])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeB", 
            initial_pose=sapien.Pose(p=[-0.2, 0, self.cube_half_size])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeC", 
            initial_pose=sapien.Pose(p=[-0.2, -0.15, self.cube_half_size])
        )
        
        self.cubes = [self.cubeA, self.cubeB, self.cubeC]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- Randomization Logic Starts Here ---
            
            # 1. Randomize Cube B Position (Center reference)
            # X range: [-0.2, -0.1] -> width 0.1, start -0.2
            # Y range: [-0.05, 0.05] -> width 0.1, start -0.05
            rand_x = torch.rand(b, device=self.device) * 0.1 
            rand_y = torch.rand(b, device=self.device) * 0.1 - 0.05
            
            # 2. Define relative offset for A and C
            offset = 0.15

            # 3. Calculate Positions
            # Cube B (Center)
            self.init_pos_B = torch.zeros((b, 3), device=self.device)
            self.init_pos_B[:, 0] = rand_x
            self.init_pos_B[:, 1] = rand_y
            self.init_pos_B[:, 2] = self.cube_half_size

            # Cube A (Left/Top relative to B) -> y + offset
            self.init_pos_A = torch.zeros((b, 3), device=self.device)
            self.init_pos_A[:, 0] = rand_x  # Keep aligned in X
            self.init_pos_A[:, 1] = rand_y + offset
            self.init_pos_A[:, 2] = self.cube_half_size
            
            # Cube C (Right/Bottom relative to B) -> y - offset
            self.init_pos_C = torch.zeros((b, 3), device=self.device)
            self.init_pos_C[:, 0] = rand_x  # Keep aligned in X
            self.init_pos_C[:, 1] = rand_y - offset
            self.init_pos_C[:, 2] = self.cube_half_size

            # 4. Set Poses
            self.cubeA.set_pose(Pose.create_from_pq(p=self.init_pos_A))
            self.cubeB.set_pose(Pose.create_from_pq(p=self.init_pos_B))
            self.cubeC.set_pose(Pose.create_from_pq(p=self.init_pos_C))

            # --- Randomization Logic Ends Here ---

            # Generate Instruction (1-6)
            self.target_indicator = torch.randint(1, 7, (b,), device=self.device)
            
            # Robot init   
            if self.agent is not None:
                qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0])
                qpos = torch.from_numpy(qpos).float().to(self.device)
                self.agent.reset(init_qpos=qpos.repeat(b, 1))

    def evaluate(self):
        # Note: evaluate uses self.init_pos_A/B/C which are correctly updated 
        # in _initialize_episode, so no changes needed here.
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p
        
        b_lifted = pos_B[:, 2] > self.max_goal_height-0.15
        
        swap_AB_done = (torch.linalg.norm(pos_A[:, :2] - self.init_pos_B[:, :2], axis=1) < self.goal_thresh) & \
                       (torch.linalg.norm(pos_B[:, :2] - self.init_pos_A[:, :2], axis=1) < self.goal_thresh)
                       
        swap_AC_done = (torch.linalg.norm(pos_A[:, :2] - self.init_pos_C[:, :2], axis=1) < self.goal_thresh) & \
                       (torch.linalg.norm(pos_C[:, :2] - self.init_pos_A[:, :2], axis=1) < self.goal_thresh)

        swap_BC_done = (torch.linalg.norm(pos_B[:, :2] - self.init_pos_C[:, :2], axis=1) < self.goal_thresh) & \
                       (torch.linalg.norm(pos_C[:, :2] - self.init_pos_B[:, :2], axis=1) < self.goal_thresh)

        success = torch.zeros_like(b_lifted, dtype=torch.bool)
        
        idx_1_2 = (self.target_indicator == 1) | (self.target_indicator == 2)
        idx_3_4 = (self.target_indicator == 3) | (self.target_indicator == 4)
        idx_5_6 = (self.target_indicator == 5) | (self.target_indicator == 6)
        
        success[idx_1_2] = swap_AB_done[idx_1_2] & b_lifted[idx_1_2]
        success[idx_3_4] = swap_AC_done[idx_3_4] & b_lifted[idx_3_4]
        success[idx_5_6] = swap_BC_done[idx_5_6] & b_lifted[idx_5_6]

        return {
            "success": success,
            "target_indicator": self.target_indicator
        }

    def _get_obs_extra(self, info: Dict):
        # [修复] 添加 tcp_pose
        obs = dict(
            target_indicator=self.target_indicator,
            tcp_pose=self.agent.tcp.pose.raw_pose,  # <--- 必须加上这一行
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        success = self.evaluate()["success"]
        b_height = self.cubeB.pose.p[:, 2]
        lift_reward = torch.clamp(b_height / self.max_goal_height, 0, 1)
        return success.float() * 10.0 + lift_reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 10.0