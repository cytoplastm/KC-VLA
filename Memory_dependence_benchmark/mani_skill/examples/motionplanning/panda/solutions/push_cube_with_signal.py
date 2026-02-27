import numpy as np
import sapien.core as sapien
import gymnasium as gym
import torch

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.utils.wrappers import RecordEpisode

def solve(env, seed=None, debug=False, vis=False):
    # 1. 重置环境
    env.reset(seed=seed)
    
    # 2. 初始化规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    env_unwrapped = env.unwrapped
    
    # ==============================================================================
    # 阶段 1: 等待信号 (Wait for Signal)
    # 流程: 初始灭灯(4-6s) -> 亮灯(2-4s) -> 灭灯(Ready!)
    # ==============================================================================
    if debug: print("🚦 正在等待信号序列: [灭 -> 亮 -> 灭(GO)]...")
    
    # 确保规划器内部记录为闭合
    planner.close_gripper()

    while not env_unwrapped.task_ready.any():
        # 获取当前关节状态
        current_qpos = env.agent.robot.get_qpos()
        
        # --- 超强力闭合 (Max Force Close) ---
        gripper_action = torch.ones((current_qpos.shape[0], 1), device=current_qpos.device) * -1.0
        action = torch.cat([current_qpos[:, :7], gripper_action], dim=1)
        
        # 执行动作，原地保持不动，等待 task_ready 变为 True
        obs, reward, terminated, truncated, info = env.step(action)
        
        if vis: env.render_human()
        if terminated or truncated:
            planner.close()
            return -1

    if debug: print("✅ 信号就绪 (灯灭)！开始执行推方块任务...")

    # ==============================================================================
    # 阶段 2: 执行推方块 (Direct Push)
    # ==============================================================================
    
    # 再次确认闭合
    planner.close_gripper()

    # 1. 获取关键位置
    cube_pos = env_unwrapped.cube.pose.p[0].cpu().numpy() 
    goal_pos = env_unwrapped.goal_region.pose.p[0].cpu().numpy()

    # 2. 计算推行向量
    diff_vec = goal_pos[:2] - cube_pos[:2] 
    dist = np.linalg.norm(diff_vec)
    direction = diff_vec / dist 

    # 3. 定义关键点参数
    PUSH_HEIGHT = 0.035  
    PRE_DIST = 0.08      

    # 4. 计算路径点
    # A. 预备点 (方块后方)
    pre_push_pos = cube_pos.copy()
    pre_push_pos[:2] -= direction * PRE_DIST
    pre_push_pos[2] = PUSH_HEIGHT

    # B. 终点 (靶心位置)
    end_push_pos = goal_pos.copy()
    end_push_pos[2] = PUSH_HEIGHT

    push_quat = [0, 1, 0, 0] 

    # --- 动作序列 ---

    if debug: print("🚀 移动到预备位置...")

    # 1. 直接下落到预备点 (Direct Reach)
    res = planner.move_to_pose_with_screw(sapien.Pose(pre_push_pos, push_quat))
    if res == -1: return -1

    if debug: print("🚀 推向目标...")
    # 2. 推 (Push)
    res = planner.move_to_pose_with_screw(sapien.Pose(end_push_pos, push_quat))
    if res == -1: return -1

    planner.close()
    
    # 返回结果
    return [{
        "success": env.evaluate()["success"], 
        "elapsed_steps": torch.as_tensor(env.elapsed_steps)
    }]

if __name__ == "__main__":
    from mani_skill.envs.tasks.memory_dependence.push_cube_with_signal import PushCubeWithSignalEnv
    
    env = gym.make("PushCubeWithSignal-v1", render_mode="rgb_array", control_mode="pd_joint_pos")
    
    env = RecordEpisode(
        env, 
        "demos_push_cube_signal", 
        save_trajectory=False, 
        save_video=True, 
        info_on_video=True
    )
    
    res = solve(env, seed=0, debug=True, vis=False)
    print(f"Result: {res}")
    
    env.close()
    print("✅ 任务完成，视频已保存！")