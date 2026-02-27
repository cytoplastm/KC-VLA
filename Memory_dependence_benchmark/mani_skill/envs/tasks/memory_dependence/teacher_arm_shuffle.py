# try:
#     from .teacher_arm_shuffle_cfgs import TEACHER_ARM_SHUFFLE_CONFIGS
# except ImportError:
#     from mani_skill.envs.tasks.memory_dependence.teacher_arm_shuffle_cfgs import TEACHER_ARM_SHUFFLE_CONFIGS

from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
import torch.random

from mani_skill.agents.robots import Panda, Fetch, XArm6Robotiq, WidowXAI
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

@register_env("TeacherArmShuffle-v1", max_episode_steps=700)
class TeacherArmShuffleEnv(BaseEnv):
    """
    Task: Teacher Arm performs a realistic, PHYSICS-BASED swap.
    FIX: Robust Type Conversion for IK (Tensor -> Numpy -> Sapien Pose).
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq", "widowxai", "so100"]
    agent: Union[Panda, Fetch, XArm6Robotiq, WidowXAI]

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
        
        center_x = 0.0
        # 初始高度 +5mm
        spawn_z = self.cube_half_size + 0.005 
        self.slot_positions = torch.tensor([
            [center_x, 0.15, spawn_z],   # Slot 0
            [center_x, 0.0,  spawn_z],   # Slot 1
            [center_x, -0.15, spawn_z]   # Slot 2
        ])
        
        self.temp_slot_pos = torch.tensor([-0.15, 0, spawn_z])
        
        self.phase_highlight_steps = 0 
        self.phase_shuffle_steps = 600 
        self.total_pre_steps = self.phase_shuffle_steps 
        
        # 教师臂位置 (0.55m)
        self.teacher_base_pose = sapien.Pose(p=[0.55, 0, 0], q=[0, 0, 0, 1])
        
        self.last_commanded_qpos = None

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _set_robot_drive_target(self, qpos):
        alpha = 0.15 
        if self.last_commanded_qpos is None:
            self.last_commanded_qpos = qpos
        else:
            self.last_commanded_qpos = self.last_commanded_qpos + alpha * (qpos - self.last_commanded_qpos)
            
        active_joints = self.teacher_robot.get_active_joints()
        for i, joint in enumerate(active_joints):
            val = self.last_commanded_qpos[i]
            if isinstance(val, (torch.Tensor, np.ndarray)):
                val = float(val)
            joint.set_drive_target(val)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.cubes = []
        for i in range(3):
            cube = actors.build_cube(
                self.scene, half_size=self.cube_half_size, 
                color=[1, 0, 0, 1], name=f"cube_{i}", 
                initial_pose=sapien.Pose(p=self.slot_positions[i])
            )
            
            shapes = []
            if hasattr(cube, "collision_shapes"): shapes = cube.collision_shapes
            elif hasattr(cube, "get_collision_shapes"): shapes = cube.get_collision_shapes()
            
            for s in shapes:
                if hasattr(s, "physical_material") and s.physical_material is not None:
                    mat = s.physical_material
                    mat.static_friction = 3.0   
                    mat.dynamic_friction = 3.0  
                    mat.restitution = 0.0       
            self.cubes.append(cube)

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        import mani_skill
        urdf_path = os.path.join(os.path.dirname(mani_skill.__file__), "assets/robots/panda/panda_v2.urdf")
        if not os.path.exists(urdf_path):
            urdf_path = os.path.join(os.path.dirname(sapien.__file__), "assets/robot/panda/panda.urdf")
        if not os.path.exists(urdf_path):
             urdf_path = "/home/chenyipeng/ManiSkill/mani_skill/assets/robots/panda/panda_v2.urdf"

        self.teacher_robot = loader.load(urdf_path)
        self.teacher_robot.set_root_pose(self.teacher_base_pose)
        self.teacher_robot.name = "teacher_panda"
        
        # 手指高摩擦
        for link in self.teacher_robot.get_links():
            if "finger" in link.name: 
                shapes = []
                if hasattr(link, "collision_shapes"): shapes = link.collision_shapes
                elif hasattr(link, "get_collision_shapes"): shapes = link.get_collision_shapes()
                
                for s in shapes:
                    if hasattr(s, "physical_material") and s.physical_material is not None:
                        mat = s.physical_material
                        mat.static_friction = 5.0  
                        mat.dynamic_friction = 5.0
                        mat.restitution = 0.0

        arm_stiffness = 1e3
        arm_damping = 1e2
        gripper_stiffness = 1e5  
        gripper_damping = 1e3
        
        active_joints = self.teacher_robot.get_active_joints()
        for i, joint in enumerate(active_joints):
            s, d = (arm_stiffness, arm_damping) if i < 7 else (gripper_stiffness, gripper_damping)
            if hasattr(joint, "set_drive_properties"):
                joint.set_drive_properties(stiffness=s, damping=d)
            elif hasattr(joint, "set_drive_property"):
                joint.set_drive_property(stiffness=s, damping=d)

        self.teacher_model = self.teacher_robot.create_pinocchio_model()
        link_names = [l.name for l in self.teacher_robot.get_links()]
        if "panda_hand" in link_names: self.teacher_ee_link_idx = link_names.index("panda_hand")
        elif "panda_link8" in link_names: self.teacher_ee_link_idx = link_names.index("panda_link8")
        else: self.teacher_ee_link_idx = len(link_names) - 1
        
        self.teacher_home_qpos = np.array([0, -0.5, 0, -2.0, 0, 2.0, 0.785, 0.04, 0.04])
        self.teacher_robot.set_qpos(self.teacher_home_qpos)
        self.last_commanded_qpos = self.teacher_home_qpos
        self._set_robot_drive_target(self.teacher_home_qpos)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.teacher_robot.set_root_pose(self.teacher_base_pose)
            # 🟢 [修改] 强制目标为 Index 1 (即最开始在中间的那个方块)
            # 无论之后它被搬到了哪里，系统认定的 Target 永远是这个物理对象
            self.target_idx = torch.ones((b,), dtype=torch.long, device=self.device) * 1
            self.slot_content = torch.tensor([[0, 1, 2]], device=self.device).repeat(b, 1)
            self.swap_instruction = torch.randint(1, 7, (b,), device=self.device)
            
            for i in range(3):
                self.cubes[i].set_pose(Pose.create_from_pq(p=self.slot_positions[i].to(self.device)))
            
            if self.agent is not None:
                qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0])
                qpos = torch.from_numpy(qpos).float().to(self.device)
                self.agent.reset(init_qpos=qpos.repeat(b, 1))
            
            self.teacher_robot.set_qpos(self.teacher_home_qpos)
            self.last_commanded_qpos = self.teacher_home_qpos
            self._set_robot_drive_target(self.teacher_home_qpos)

    def step(self, action):
        current_step = self.elapsed_steps[0].item()
        
        # 1. 教师臂逻辑 (只负责动老师，不动学生)
        if current_step < self.total_pre_steps:
            shuffle_progress = current_step / self.phase_shuffle_steps
            self._teacher_action_script(shuffle_progress)

            # 🟢 [新增] 在 Phase 3 刚刚松开夹爪准备撤退的那一帧 (Step 580)，强制归位
            if current_step == 590:
                # 1. 先更新逻辑状态 (Logic Update): 告诉系统现在哪个方块在哪个槽里
                self._update_slot_content_based_on_swap()
                # 2. 再执行物理瞬移 (Physical Snap): 把方块移动到新状态对应的标准位置
                self._force_cubes_to_slots()
        else:
            self._set_robot_drive_target(self.teacher_home_qpos)

        # 2. 直接将 action 传给物理引擎
        # 此时 action 完全由你的 solve() 函数决定
        obs, reward, terminated, truncated, info = super().step(action)
        
        return obs, reward, terminated, truncated, info
    
    def _update_slot_content_based_on_swap(self):
        """
        根据 swap_instruction 更新 self.slot_content
        确保 _force_cubes_to_slots 知道方块的新位置
        """
        for b_idx in range(self.num_envs):
            instr = self.swap_instruction[b_idx].item()
            slot_a, slot_b = -1, -1
            
            # 解析交换指令
            if instr == 1: slot_a, slot_b = 0, 1 
            elif instr == 2: slot_a, slot_b = 1, 0 
            elif instr == 3: slot_a, slot_b = 0, 2
            elif instr == 4: slot_a, slot_b = 2, 0
            elif instr == 5: slot_a, slot_b = 1, 2
            elif instr == 6: slot_a, slot_b = 2, 1
            
            # 🟢 执行逻辑交换
            # 获取两个 Slot 当前的内容
            cube_in_a = self.slot_content[b_idx, slot_a].item()
            cube_in_b = self.slot_content[b_idx, slot_b].item()
            
            # 交换
            self.slot_content[b_idx, slot_a] = cube_in_b
            self.slot_content[b_idx, slot_b] = cube_in_a

    def _force_cubes_to_slots(self):
        """
        强制将所有方块瞬移到它们所属 Slot 的标准中心位置。
        这会根据更新后的 self.slot_content 来执行，确保方块去的是交换后的新位置。
        """
        with torch.device(self.device):
            b = self.num_envs
            target_pos_for_cubes = torch.zeros((b, 3, 3), device=self.device)
            standard_slot_pos = self.slot_positions.to(self.device)
            batch_indices = torch.arange(b, device=self.device)

            # 遍历 3 个 Slot
            for slot_idx in range(3):
                # 这个 Slot 的标准位置
                pos = standard_slot_pos[slot_idx].unsqueeze(0).expand(b, -1)
                
                # 这个 Slot 现在(逻辑上)装的是哪个方块?
                cube_indices = self.slot_content[:, slot_idx]
                
                # 让那个方块瞬移到这里
                target_pos_for_cubes[batch_indices, cube_indices] = pos

            default_quat = torch.tensor([1, 0, 0, 0], dtype=torch.float, device=self.device).repeat(b, 1)
            zero_vel = torch.zeros((b, 3), device=self.device)
            
            for i in range(3):
                target_p = target_pos_for_cubes[:, i, :]
                
                # 设置位置和姿态
                self.cubes[i].set_pose(Pose.create_from_pq(target_p, default_quat))
                
                # 清空速度，防止起飞
                self.cubes[i].set_linear_velocity(zero_vel)
                self.cubes[i].set_angular_velocity(zero_vel)

    def _teacher_action_script(self, progress):
        phase_len = 1.0 / 3.0
        
        if progress < phase_len:
            phase = 1
            t = progress / phase_len
        elif progress < 2 * phase_len:
            phase = 2
            t = (progress - phase_len) / phase_len
        else:
            phase = 3
            t = (progress - 2 * phase_len) / phase_len

        for b_idx in range(self.num_envs):
            instr = self.swap_instruction[b_idx].item()
            
            slot_a, slot_b = -1, -1
            if instr == 1: slot_a, slot_b = 0, 1 
            elif instr == 2: slot_a, slot_b = 1, 0 
            elif instr == 3: slot_a, slot_b = 0, 2
            elif instr == 4: slot_a, slot_b = 2, 0
            elif instr == 5: slot_a, slot_b = 1, 2
            elif instr == 6: slot_a, slot_b = 2, 1
            
            cube_a_idx = self.slot_content[b_idx, slot_a]
            cube_b_idx = self.slot_content[b_idx, slot_b]
            
            p_cube_a = self.cubes[cube_a_idx].pose.p[b_idx]
            p_cube_b = self.cubes[cube_b_idx].pose.p[b_idx]
            
            p_slot_a = self.slot_positions[slot_a].to(self.device)
            p_slot_b = self.slot_positions[slot_b].to(self.device)
            p_temp = self.temp_slot_pos.to(self.device)

            # 🟢 [修改] 添加 X 轴偏移量 (-0.02)
            # 机器人基座在 0.55，目标在 0.0。
            # 要伸得更远（靠近 0.0 甚至负数），需要减小 X 值。
            offset = torch.tensor([-0.01, 0, 0], device=self.device)
            
            # 对所有目标位置应用偏移
            p_cube_a = p_cube_a + offset
            p_cube_b = p_cube_b + offset
            p_slot_a = p_slot_a + offset
            p_slot_b = p_slot_b + offset
            p_temp = p_temp + offset

            target_hand_pos = None
            gripper_closed = False
            
            if phase == 1: 
                target_hand_pos, gripper_closed = self._compute_trajectory(t, p_cube_a, p_temp)
            elif phase == 2: 
                target_hand_pos, gripper_closed = self._compute_trajectory(t, p_cube_b, p_slot_a)
            elif phase == 3: 
                target_hand_pos, gripper_closed = self._compute_trajectory(t, p_cube_a, p_slot_b)
            
            if b_idx == 0 and target_hand_pos is not None:
                self._apply_teacher_ik(target_hand_pos, gripper_closed)
    
    def _compute_trajectory(self, t, start_pos, end_pos):
        # 抬起高度和抓取高度
        hover_z = 0.25
        grasp_z = 0.12
        
        pos = start_pos.clone() # 默认锁定在起始位置的 X,Y
        closed = False
        
        # 1. [0.00 - 0.25] 下潜接近 (Approach)
        if t < 0.25: 
            # 只改变 Z 轴，XY 保持不动
            alpha = t / 0.25
            pos[2] = hover_z * (1-alpha) + grasp_z * alpha
            closed = False 
            
        # 2. [0.25 - 0.40] 闭合等待 (Grasp & Stabilize) -> 关键修改！
        elif t < 0.40: 
            # 手臂完全静止在 grasp_z，只闭合夹爪
            # 给物理引擎 15% 的时间建立接触和摩擦力
            pos[2] = grasp_z
            closed = True 
            
        # 3. [0.40 - 0.50] 垂直抬起 (Lift Vertical) -> 关键修改！
        elif t < 0.50: 
            # 夹爪保持闭合，XY 依然不动，只抬高 Z 轴
            # 防止水平分力拖动方块
            alpha = (t - 0.40) / 0.10
            pos[2] = grasp_z * (1-alpha) + hover_z * alpha
            closed = True 
            
        # 4. [0.50 - 0.80] 空中平移 (Move to Target)
        elif t < 0.80: 
            # 此时高度已安全，开始插值移动 X/Y 到目标
            alpha = (t - 0.50) / 0.30
            pos = torch.lerp(start_pos, end_pos, alpha)
            pos[2] = hover_z 
            # 可选：加一点微小的抛物线，看起来更自然
            pos[2] += np.sin(alpha * np.pi) * 0.05 
            closed = True
            
        # 5. [0.80 - 0.90] 下放 (Lower)
        elif t < 0.90: 
            # 到达目标上方，锁定 X/Y，垂直下放
            pos = end_pos.clone()
            alpha = (t - 0.80) / 0.10
            pos[2] = hover_z * (1-alpha) + grasp_z * alpha
            closed = True
            
        # 6. [0.90 - 1.00] 松开撤退 (Release & Retreat)
        else: 
            pos = end_pos.clone()
            # 松开夹爪并稍微抬起一点
            alpha = (t - 0.90) / 0.10
            pos[2] = grasp_z * (1-alpha) + hover_z * alpha
            closed = False
            
        return pos, closed

    def _apply_teacher_ik(self, target_pos, closed):
        """
        Robust IK with Type Cleaning (ManiSkill Pose -> Sapien Pose)
        """
        # 1. 目标位置转 NumPy
        if isinstance(target_pos, torch.Tensor):
            target_pos_np = target_pos.cpu().numpy()
        else:
            target_pos_np = target_pos
            
        # 2. 构建目标的 World Pose (Sapien Type)
        target_pose_world = sapien.Pose(p=target_pos_np, q=[0, 1, 0, 0])
        
        # 3. 获取机器人基座 Pose，并强制清洗为 Sapien Pose
        # ManiSkill 的 robot.get_pose() 返回的是封装对象(含Tensor)，必须拆包
        raw_base_pose = self.teacher_robot.get_pose()
        if hasattr(raw_base_pose, "raw_pose"): # 检测是否为 ManiSkill 封装的 Pose
            # 取出第一个环境的数据 (假设所有环境基座位置相同，或者我们只处理 env 0)
            p = raw_base_pose.p[0].cpu().numpy()
            q = raw_base_pose.q[0].cpu().numpy()
            base_pose_sapien = sapien.Pose(p=p, q=q)
        else:
            # 已经是 Sapien Pose
            base_pose_sapien = raw_base_pose
            
        # 4. 计算局部坐标 (Local Frame)
        # 纯粹的 Sapien Pose 计算，结果也是 Sapien Pose
        target_pose_local = base_pose_sapien.inv() * target_pose_world
        
        # 5. 获取当前关节角度 (NumPy)
        current_qpos = self.teacher_robot.get_qpos()
        if isinstance(current_qpos, torch.Tensor):
            current_qpos = current_qpos.cpu().numpy()
        current_qpos = current_qpos.flatten().astype(np.float64)
        
        # 6. IK 解算
        result, success, error = self.teacher_model.compute_inverse_kinematics(
            self.teacher_ee_link_idx, 
            target_pose_local, # 这里必须是干净的 sapien.Pose
            initial_qpos=current_qpos, 
            active_qmask=[1,1,1,1,1,1,1,0,0],
            damp=0.05 
        )

        if success:
            qpos = result
            # 🟢 [修改核心] 柔顺抓取逻辑
            if closed:
                # 目标不是 0，而是比方块略小一点点 (挤压 2-3mm)
                # self.cube_half_size 通常是 0.02
                safe_close_val = max(0.0, self.cube_half_size - 0.003) 
                val = safe_close_val
            else:
                val = 0.04 # 打开状态
                
            qpos[-2:] = val
            self._set_robot_drive_target(qpos)

    def evaluate(self):
        # 1. 获取当前步数
        current_step = self.elapsed_steps[0].item()
        
        # 2. 时间锁: 洗牌阶段前 600 步强制失败
        if current_step < self.total_pre_steps:
             return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 3. 遍历检查
        # 设定阈值为 0.1m
        thresh = 0.1 
        
        # 获取 batch 中的第一个环境的目标索引 (用于调试打印)
        debug_target_idx = self.target_idx[0].item()
        
        for i in range(3):
            # 检查高度
            cube_z = self.cubes[i].pose.p[:, 2]
            is_lifted = cube_z > thresh
            
            # 检查是否为目标
            # 🟢 [修改] 显式转换 i 为 tensor 以确保比较正确，或者直接比较值
            is_target = (self.target_idx == i)
            
            # 计算当前方块是否成功 (既是目标，又被抓起)
            is_success = is_lifted & is_target
            success = success | is_success
            
            # 🟢 [调试打印] 仅当方块被抓起时打印，方便排查
            # 只打印第一个环境 (Batch 0) 的情况，防止刷屏
            # if is_lifted[0].item():
            #     print(f"   [DEBUG Step {current_step}] Cube {i} is LIFTED (Height: {cube_z[0].item():.3f}m).")
            #     print(f"   -> Is it Target? {'YES' if is_target[0].item() else 'NO'} (Target is Cube {debug_target_idx})")
            #     if not is_target[0].item():
            #         print(f"   -> ❌ 抓错了！你抓的是 Cube {i}，但目标是 Cube {debug_target_idx}")
            #     else:
            #         print(f"   -> ✅ 抓对了！判定成功。")

        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        target_indices = self.target_idx.view(-1, 1, 1).expand(-1, 1, 3)
        all_cube_pos = torch.stack([c.pose.p for c in self.cubes], dim=1)
        target_pos = torch.gather(all_cube_pos, 1, target_indices).squeeze(1)
        dist = torch.linalg.norm(tcp_pos - target_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * dist)
        success = self.evaluate()["success"]
        return reaching_reward + success.float() * 10.0
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 10.0