from typing import Any, Dict, Union, List

import numpy as np
import sapien
import sapien.render  # [Crucial] Import render module explicitly
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, Fetch, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

try:
    from .push_cube_with_signal_cfgs import PUSH_CUBE_WITH_SIGNAL_CONFIGS
except ImportError:
    from mani_skill.envs.tasks.memory_dependence.push_cube_with_signal_cfgs import PUSH_CUBE_WITH_SIGNAL_CONFIGS

@register_env("PushCubeWithSignal-v1", max_episode_steps=600)
class PushCubeWithSignalEnv(BaseEnv):
    """
    Task: Wait for light signals -> Push cube to fixed target.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    
    goal_radius = 0.1     
    success_thresh = 0.08 
    
    cube_half_size: float
    sensor_cam_eye_pos: list
    sensor_cam_target_pos: list
    human_cam_eye_pos: list
    human_cam_target_pos: list
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        if robot_uids in PUSH_CUBE_WITH_SIGNAL_CONFIGS:
            cfg = PUSH_CUBE_WITH_SIGNAL_CONFIGS[robot_uids]
        else:
            cfg = PUSH_CUBE_WITH_SIGNAL_CONFIGS["panda"]

        self.cube_half_size = cfg["cube_half_size"]
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

    # --- 1. Load Scene ---
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.table = self.table_scene.table
        
        # A. Cube
        self.cube = actors.build_cube(
            self.scene, 
            half_size=self.cube_half_size, 
            color=[0, 0, 1, 1], 
            name="target_cube", 
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )

        # B. Goal Target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

        # C. LED Indicator
        self._build_led()

    def _build_led(self):
        """
        Builds a replica of the reference image: Black double-layer base + Grey pear bulb.
        Corrects cylinder orientation to stand vertically on the table.
        """
        builder = self.scene.create_actor_builder()
        
        # 1. Create Materials
        self.mat_base = sapien.render.RenderMaterial()
        self.mat_base.base_color = [0.05, 0.05, 0.05, 1.0]
        self.mat_base.roughness = 0.8
        
        self.mat_bulb = sapien.render.RenderMaterial()
        self.mat_bulb.base_color = [0.5, 0.5, 0.5, 1.0] 
        self.mat_bulb.roughness = 0.2
        self.mat_bulb.metallic = 0.0
        self.mat_bulb.specular = 0.5
        self.mat_bulb.emission = [0, 0, 0, 1]

        # 2. Build Geometry (Double Layer Base)
        
        # [CRITICAL FIX] Rotation to align cylinder with Z-axis (Standing up)
        # Default cylinder is along X-axis. Rotate 90 deg around Y.
        rotate_to_z = euler2quat(0, np.pi/2, 0)
        
        # Layer 1: Bottom (Wide & Flat) -> r=4cm, h=1cm
        r_base_1 = 0.04   
        h_base_1 = 0.01
        z_base_1 = h_base_1
        
        builder.add_cylinder_visual(
            radius=r_base_1, half_length=h_base_1, 
            material=self.mat_base, 
            pose=sapien.Pose(p=[0, 0, z_base_1], q=rotate_to_z) # Apply rotation
        )
        builder.add_cylinder_collision(
            radius=r_base_1, half_length=h_base_1, 
            pose=sapien.Pose(p=[0, 0, z_base_1], q=rotate_to_z) # Apply rotation
        )
        
        # Layer 2: Top (Narrow Connector) -> r=2.5cm, h=1.5cm
        r_base_2 = 0.02  
        h_base_2 = 0.01
        z_base_2 = (2 * h_base_1) + h_base_2
        
        builder.add_cylinder_visual(
            radius=r_base_2, half_length=h_base_2, 
            material=self.mat_base, 
            pose=sapien.Pose(p=[0, 0, z_base_2], q=rotate_to_z) # Apply rotation
        )
        builder.add_cylinder_collision(
            radius=r_base_2, half_length=h_base_2, 
            pose=sapien.Pose(p=[0, 0, z_base_2], q=rotate_to_z) # Apply rotation
        )
        
        # --- Composite Bulb Shape (Pear) ---
        bulb_start_z = (2 * h_base_1) + (2 * h_base_2)

        # Part 1: Lower Bulb (Neck)
        r_lower = 0.02
        z_lower = bulb_start_z + r_lower
        pose_lower = sapien.Pose(p=[0, 0, z_lower])
        
        builder.add_sphere_visual(radius=r_lower, material=self.mat_bulb, pose=pose_lower)
        builder.add_sphere_collision(radius=r_lower, pose=pose_lower)

        # Part 2: Upper Bulb (Body)
        r_upper = 0.032
        z_upper = z_lower + 0.025 
        pose_upper = sapien.Pose(p=[0, 0, z_upper])

        builder.add_sphere_visual(radius=r_upper, material=self.mat_bulb, pose=pose_upper)
        builder.add_sphere_collision(radius=r_upper, pose=pose_upper)
        
        # Finalize Actor
        base_x, base_y, base_z = -0.1, 0.3, 0.0 
        builder.initial_pose = sapien.Pose(p=[base_x, base_y, base_z])
        self.led_actor = builder.build_static(name="led_indicator")

        # 3. Add Point Light (At center of upper bulb)
        light_world_pos = np.array([base_x, base_y, base_z + z_upper])
        self.led_light = self.scene.add_point_light(
            light_world_pos,      
            color=[0, 0, 0], 
            shadow=False,    
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # --- 2. Initialization ---
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            if self.agent is not None:
                qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0])
                qpos = torch.from_numpy(qpos).float().to(self.device)
                self.agent.reset(init_qpos=qpos.repeat(b, 1))

            # Cube Position
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[..., 0] = torch.rand((b,), device=self.device) * 0.1 - 0.2
            xyz[..., 1] = (torch.rand((b,), device=self.device) - 0.5) * 0.1
            xyz[..., 2] = self.cube_half_size
            p = xyz.clone()
            self.cube.set_pose(Pose.create_from_pq(p=p))
            self.cube_init_pos = p.clone()

            # Target Position
            target_pos = torch.zeros((b, 3), device=self.device)
            target_pos[..., 0] = 0.1
            target_pos[..., 1] = 0.0
            target_pos[..., 2] = 1e-3
            flat_quat = euler2quat(0, np.pi / 2, 0)
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_pos,
                    q=torch.tensor(flat_quat, device=self.device).repeat(b, 1)
                )
            )

            # Signal Logic
            freq = self.control_freq
            t0 = (torch.rand(b, device=self.device) * 1.0 + 2.0) * freq
            t1 = (torch.rand(b, device=self.device) * 3.0 + 1.0) * freq
            t2 = (torch.rand(b, device=self.device) * 2.0 + 2.0) * freq
            t3 = (torch.rand(b, device=self.device) * 3.0 + 1.0) * freq
            
            self.phase_durations = torch.stack([t0, t1, t2, t3], dim=1).long()
            self.current_phase = torch.zeros(b, dtype=torch.long, device=self.device)
            self.phase_elapsed = torch.zeros(b, dtype=torch.long, device=self.device)
            self.task_ready = torch.zeros(b, dtype=torch.bool, device=self.device)
            self.has_moved_early = torch.zeros(b, dtype=torch.bool, device=self.device)

            self._update_led_visuals()

    # --- 3. Control Logic ---
    def _before_control_step(self):
        not_done = self.current_phase < 4
        self.phase_elapsed[not_done] += 1
        
        active_indices = torch.where(not_done)[0]
        if len(active_indices) > 0:
            current_p_idx = self.current_phase[active_indices]
            durations = self.phase_durations[active_indices, current_p_idx]
            elapsed = self.phase_elapsed[active_indices]
            
            need_switch = elapsed >= durations
            switch_env_idx = active_indices[need_switch]
            
            self.phase_elapsed[switch_env_idx] = 0
            self.current_phase[switch_env_idx] += 1
            
            self.task_ready = (self.current_phase >= 4)
            self._update_led_visuals()

    def _update_led_visuals(self):
        """
        Updates the LED visuals by directly modifying the stored material properties.
        """
        phase = self.current_phase[0].item()
        
        if phase == 1 or phase == 3:
            # === ON ===
            is_on = True
            light_scale = 8.0 
        else:
            # === OFF ===
            is_on = False
            light_scale = 0.0

        # 1. Update Point Light
        self.led_light.set_color([1.0 * light_scale, 1.0 * light_scale, 1.0 * light_scale])

        # 2. Update Bulb Material (Directly modify self.mat_bulb)
        if hasattr(self, "mat_bulb"):
            if is_on:
                # Bright White, High Emission
                self.mat_bulb.base_color = [1.0, 1.0, 1.0, 1.0]
                self.mat_bulb.emission = [20.0, 20.0, 20.0, 1.0]
                self.mat_bulb.roughness = 0.5
            else:
                # [Mod] Medium Grey (0.5) for OFF state
                self.mat_bulb.base_color = [0.5, 0.5, 0.5, 1.0]
                self.mat_bulb.emission = [0.0, 0.0, 0.0, 1.0]
                self.mat_bulb.roughness = 0.2 # Semi-Matte
                self.mat_bulb.specular = 0.5

    # --- 4. Evaluation ---
    def evaluate(self):
        dist_from_start = torch.linalg.norm(self.cube.pose.p - self.cube_init_pos, axis=1)
        moved = dist_from_start > 0.01
        self.has_moved_early = self.has_moved_early | (moved & (~self.task_ready))
        
        dist_to_goal = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        
        is_obj_placed = (dist_to_goal < self.success_thresh) & \
                        (self.cube.pose.p[..., 2] < self.cube_half_size + 0.005)

        success = is_obj_placed & (~self.has_moved_early) & self.task_ready
        
        return {
            "success": success,
            "has_moved_early": self.has_moved_early,
            "dist_to_goal": dist_to_goal
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:
             obs.update(
                cube_pose=self.cube.pose.raw_pose,
                goal_pos=self.goal_region.pose.p,
                led_signal=self.task_ready.float().unsqueeze(1), 
                task_phase=self.current_phase.float().unsqueeze(1)
            )
        return obs

    def compute_dense_reward(self, obs, action, info):
        reward = torch.zeros(self.num_envs, device=self.device)
        reward[self.has_moved_early] = -10.0
        
        tcp_push_pose = self.cube.pose.p + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_push_pose - self.agent.tcp.pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        reached = tcp_to_push_pose_dist < 0.02
        dist_to_goal = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * dist_to_goal)
        reward += place_reward * reached
        
        reward[info["success"]] = 5.0
        
        return reward
    
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0