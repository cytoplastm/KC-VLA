from typing import Any, Dict, Union

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
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

try:
    from .swap_cube_cfgs import SWAP_CUBE_CONFIGS
except ImportError:
    from mani_skill.envs.tasks.memory_dependence.swap_cube_cfgs import SWAP_CUBE_CONFIGS

@register_env("SwapThreeCubes-v1", max_episode_steps=300)
class SwapThreeCubesEnv(BaseEnv):
    """
    Task Description:
    Three cubes (A, B, C) are stacked (A on bottom, B middle, C top).
    The goal is to swap the positions of the bottom cube (A) and the middle cube (B).
    Final state should be: B (bottom) -> A (middle) -> C (top).
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq"]
    agent: Union[Panda, Fetch, XArm6Robotiq]

    # 定义成员变量
    cube_half_size: float
    sensor_cam_eye_pos: list
    sensor_cam_target_pos: list
    human_cam_eye_pos: list
    human_cam_target_pos: list

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        if robot_uids in SWAP_CUBE_CONFIGS:
            cfg = SWAP_CUBE_CONFIGS[robot_uids]
        else:
            cfg = SWAP_CUBE_CONFIGS["panda"]
            
        self.cube_half_size = cfg["cube_half_size"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        self.spawn_offset_x = cfg.get("spawn_offset_x", 0.0)
        self.spawn_offset_y = cfg.get("spawn_offset_y", 0.0)

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
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.table = self.table_scene.table
        
        # 创建时放在高空并不影响，因为下一帧 reset 就会被覆盖
        # 这里只是为了避免 warning
        self.cubeA = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0,0,0.5]))
        self.cubeB = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cubeB", initial_pose=sapien.Pose(p=[0,0,0.6]))
        self.cubeC = actors.build_cube(self.scene, half_size=self.cube_half_size, color=[0, 0, 1, 1], name="cubeC", initial_pose=sapien.Pose(p=[0,0,0.7]))

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            table_height = 0.0
            base_x, base_y = self.spawn_offset_x, self.spawn_offset_y

            # -------------------------------------------------------------------------- #
            # 1. 修复机器人初始姿态
            # -------------------------------------------------------------------------- #
            if self.agent is not None:
                qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0])
                qpos = torch.from_numpy(qpos).float().to(self.device)
                batch_qpos = qpos.repeat(b, 1)
                self.agent.reset(init_qpos=batch_qpos)

            # -------------------------------------------------------------------------- #
            # 2. 初始化方块位置
            # -------------------------------------------------------------------------- #
            z_bottom = table_height + self.cube_half_size
            z_mid    = z_bottom + self.cube_half_size * 2
            z_top    = z_mid + self.cube_half_size * 2
            
            z_options = torch.tensor([z_bottom, z_mid, z_top], device=self.device, dtype=torch.float)
            z_options = z_options.repeat(b, 1)

            # perm_indices: Shape (b, 3)
            # 含义：[CubeA的高度索引, CubeB的高度索引, CubeC的高度索引]
            # 例如 [2, 0, 1] 代表: A在顶(2), B在底(0), C在中(1)
            perm_indices = torch.argsort(torch.rand(b, 3, device=self.device), dim=1)

            # --- [关键修复：定义目标变量] ---
            # 我们需要知道"谁在底部"、"谁在中间"。
            # 对 perm_indices 再做一次 argsort，就能得到 [底部Cube的ID, 中间Cube的ID, 顶部Cube的ID]
            # 例如 perm=[2,0,1], argsort后变成 [1, 2, 0]。意思是:
            # 索引1(CubeB)高度是0(底); 索引2(CubeC)高度是1(中); 索引0(CubeA)高度是2(顶)
            current_roles = torch.argsort(perm_indices, dim=1)
            
            idx_bot = current_roles[:, 0] # 当前在底部的方块ID
            idx_mid = current_roles[:, 1] # 当前在中间的方块ID
            idx_top = current_roles[:, 2] # 当前在顶部的方块ID

            # 定义任务目标：交换底部和中间
            self.target_bottom_idx = idx_mid  # 原来的中间 -> 变成底部
            self.target_middle_idx = idx_bot  # 原来的底部 -> 变成中间
            self.target_top_idx    = idx_top  # 原来的顶部 -> 保持不变
            # -----------------------------

            # 分配物理高度
            z_A = z_options[torch.arange(b), perm_indices[:, 0]]
            z_B = z_options[torch.arange(b), perm_indices[:, 1]]
            z_C = z_options[torch.arange(b), perm_indices[:, 2]]

            # -------------------------------------------------------------------------- #
            # 3. 设置 Pose
            # -------------------------------------------------------------------------- #
            xy = torch.zeros((b, 2), device=self.device)
            
            #新：x=-0.1~0，y=-0.1~0.1
            xy[:, 0] = -0.1 + 0.1 * torch.rand(b, device=self.device)
            xy[:, 1] = -0.1 + 0.2 * torch.rand(b, device=self.device)

            #原：xy位置固定在(0,0)
            # xy[:, 0] = base_x
            # xy[:, 1] = base_y
            
            default_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device).repeat(b, 1)

            # Cube A
            xyz_A = torch.cat([xy, z_A.unsqueeze(1)], dim=1)
            self.cubeA.set_pose(Pose.create_from_pq(xyz_A, default_quat))

            # Cube B
            xyz_B = torch.cat([xy, z_B.unsqueeze(1)], dim=1)
            self.cubeB.set_pose(Pose.create_from_pq(xyz_B, default_quat))

            # Cube C
            xyz_C = torch.cat([xy, z_C.unsqueeze(1)], dim=1)
            self.cubeC.set_pose(Pose.create_from_pq(xyz_C, default_quat))

    def evaluate(self):
        # 1. 获取所有方块的 Pose
        # p_A, p_B, p_C 的 shape 都是 (b, 3)
        p_A = self.cubeA.pose.p
        p_B = self.cubeB.pose.p
        p_C = self.cubeC.pose.p
        
        # 2. 将它们堆叠起来，方便按索引取值
        # stack_poses shape: (b, 3, 3) -> (Batch, Cube_ID, XYZ)
        # 索引对应关系: 0->A, 1->B, 2->C
        stack_poses = torch.stack([p_A, p_B, p_C], dim=1)
        
        b = stack_poses.shape[0]
        batch_idx = torch.arange(b, device=self.device)

        # 3. 根据初始化时记录的目标索引，提取对应的方块坐标
        # 提取"应该是底部"的那个方块的坐标
        p_target_bottom = stack_poses[batch_idx, self.target_bottom_idx]
        # 提取"应该是中间"的那个方块的坐标
        p_target_middle = stack_poses[batch_idx, self.target_middle_idx]
        # 提取"应该是顶部"的那个方块的坐标
        p_target_top    = stack_poses[batch_idx, self.target_top_idx]

        # 4. 开始判定 (逻辑与之前类似，但对象变了)
        xy_threshold = 0.02
        z_stack_err = 0.005 

        # A) 判定底部方块是否在桌面上
        # 它的 Z 轴高度应该等于 half_size
        is_bot_on_table = torch.abs(p_target_bottom[:, 2] - self.cube_half_size) < z_stack_err

        # B) 判定中间方块是否在底部方块上
        # XY 距离要近
        mid_on_bot_xy = torch.linalg.norm(p_target_middle[:, :2] - p_target_bottom[:, :2], axis=1) < xy_threshold
        # Z 轴高度差应该是 2 * half_size
        mid_on_bot_z  = torch.abs(p_target_middle[:, 2] - p_target_bottom[:, 2] - self.cube_half_size * 2) < z_stack_err
        is_mid_on_bot = mid_on_bot_xy & mid_on_bot_z

        # C) 判定顶部方块是否在中间方块上
        top_on_mid_xy = torch.linalg.norm(p_target_top[:, :2] - p_target_middle[:, :2], axis=1) < xy_threshold
        top_on_mid_z  = torch.abs(p_target_top[:, 2] - p_target_middle[:, 2] - self.cube_half_size * 2) < z_stack_err
        is_top_on_mid = top_on_mid_xy & top_on_mid_z

        # 5. 综合判定
        success = is_bot_on_table & is_mid_on_bot & is_top_on_mid
        
        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
             obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(1, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)