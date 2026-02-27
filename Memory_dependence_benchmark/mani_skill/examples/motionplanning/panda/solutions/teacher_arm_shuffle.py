import numpy as np
import sapien
import gymnasium as gym
import torch

from mani_skill.envs.tasks.memory_dependence.teacher_arm_shuffle import TeacherArmShuffleEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb
)

def solve(env: TeacherArmShuffleEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    env_unwrapped = env.unwrapped
    FINGER_LENGTH = 0.025
    
    # ==============================================================================
    # 1. 等待阶段 (Wait)
    # ==============================================================================
    if debug: print(f"⏳ Waiting for shuffle phase to complete...")
    
    # 构造保持动作
    current_qpos = env.agent.robot.get_qpos()
    if isinstance(current_qpos, torch.Tensor): current_qpos = current_qpos.cpu().numpy()
    if current_qpos.ndim == 1: current_qpos = current_qpos[None, :]
    
    arm_qpos = current_qpos[:, :7]
    finger_pos = current_qpos[:, 7]
    gripper_action = np.where(finger_pos > 0.035, 1.0, -1.0)[:, None]
    wait_action = np.hstack([arm_qpos, gripper_action])
    
    # 手动执行 step，不 yield
    wait_steps = env_unwrapped.total_pre_steps + 20
    for _ in range(wait_steps):
        env.step(wait_action)

    # ==============================================================================
    # 2. 识别目标
    # ==============================================================================
    target_idx = env_unwrapped.target_idx[0].item()
    target_cube = env_unwrapped.cubes[target_idx]
    if debug: print(f"🎯 Target Found: Cube {target_idx}")

    # ==============================================================================
    # 3. 规划并执行抓取 (Planner 内部会 step)
    # ==============================================================================
    planner.open_gripper()
    
    obb = get_actor_obb(target_cube)
    approaching = np.array([0, 0, -1])
    
    tcp_mat = env.agent.tcp.pose.to_transformation_matrix()
    target_closing = tcp_mat[0, :3, 1] if len(tcp_mat.shape) == 3 else tcp_mat[:3, 1]
    if isinstance(target_closing, torch.Tensor): target_closing = target_closing.cpu().numpy()
        
    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # A. 悬停
    hover_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(hover_pose)

    # B. 下抓
    planner.move_to_pose_with_screw(grasp_pose)
    
    # C. 闭合
    planner.close_gripper()
    
    # D. 抬起 0.2m (关键数据)
    lift_height = 0.2
    lift_pose = grasp_pose * sapien.Pose([0, 0, -lift_height])
    planner.move_to_pose_with_screw(lift_pose)
    
    planner.close()
    
    # ==============================================================================
    # 4. 返回最终结果
    # ==============================================================================
    # 1. 获取成功状态 (Tensor)
    success_tensor = env.evaluate()["success"]
    # 2. 获取当前总步数 (Tensor)
    # 注意：env.elapsed_steps 通常是一个 Tensor，run.py 需要用 .item() 读取它
    elapsed_steps_tensor = env.elapsed_steps
    # 3. 构造 run.py 期望的完整字典
    # run.py 会读取: res[-1]["success"] 和 res[-1]["elapsed_steps"]
    return [{
        "success": success_tensor,
        "elapsed_steps": elapsed_steps_tensor
    }]
    # success = env.evaluate()["success"].item()
    # return success

if __name__ == "__main__":
    env = gym.make("TeacherArmShuffle-v1", num_envs=1, render_mode="rgb_array", control_mode="pd_joint_pos")
    
    # 🔴 关键：save_trajectory=True
    env = RecordEpisode(
        env, 
        output_dir="demos_teacher_shuffle", 
        save_trajectory=True, 
        save_video=True, 
        info_on_video=True
    )
    
    NUM_EPISODES = 10
    print(f"🚀 开始采集 {NUM_EPISODES} 条演示数据 ...")

    for episode_idx in range(NUM_EPISODES):
        print(f"🎬 正在执行第 {episode_idx + 1}/{NUM_EPISODES} 个 Episode...")
        
        try:
            # 🟢 直接调用 solve 函数，不再用 for 循环
            # solve 函数内部会处理所有的 step，直到完成
            is_success = solve(env, seed=episode_idx, debug=True, vis=False)
            
            # 拿到最终步数
            total_steps = env.elapsed_steps.item()
            status = "✅ 成功" if is_success else "❌ 失败"
            print(f"   -> 结束: {status} (Total Steps: {total_steps})")
            
        except Exception as e:
            print(f"   -> ❌ 出错: {e}")
            import traceback
            traceback.print_exc()
            env.reset()

    env.close()
    print(f"🎉 全部完成！")