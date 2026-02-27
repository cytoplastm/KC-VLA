import numpy as np
import sapien.core as sapien
import gymnasium as gym
import torch

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, 
    get_actor_obb
)
from mani_skill.utils.wrappers import RecordEpisode

# 定义 Panda 机械臂的标准展开姿态
HOME_QPOS = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0]

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
    FINGER_LENGTH = 0.025
    CUBE_HALF_SIZE = 0.02

    # -------------------------------------------------------------------------
    # 辅助函数: 抓取并放回原位 (Pick & Place Back)
    # -------------------------------------------------------------------------
    def pick_and_place_back(obj_to_pick, target_pos_np):
        # target_pos_np: [x, y, z] 方块的目标位置(原位)
        
        # --- 步骤 1: 计算抓取位姿 ---
        obb = get_actor_obb(obj_to_pick)
        approaching = np.array([0, 0, -1]) # 垂直向下
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj_to_pick.pose.sp.p)

        # --- 步骤 2: 预备抓取 (Reach) ---
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return False

        # --- 步骤 3: 下落抓取 (Grasp) ---
        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1: return False
        
        planner.close_gripper()

        # --- 步骤 4: 抬起 (Lift) ---
        # 抬起高度稍微高一点，体现出"抓起"的动作，比如 20cm
        lift_pose = grasp_pose * sapien.Pose([0, 0, -0.20]) 
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1: return False

        # --- 步骤 5: 移动到目标上方 (Hover) ---
        # 目标是放回原位，坐标还是 target_pos_np
        place_z = target_pos_np[2] + CUBE_HALF_SIZE + 0.002
        hover_pose = sapien.Pose([target_pos_np[0], target_pos_np[1], lift_pose.p[2]], lift_pose.q)
        
        res = planner.move_to_pose_with_screw(hover_pose) 
        if res == -1: return False

        # --- 步骤 6: 下放到位 (Place) ---
        final_place_pose = sapien.Pose([target_pos_np[0], target_pos_np[1], place_z], lift_pose.q)
        res = planner.move_to_pose_with_screw(final_place_pose)
        if res == -1: return False

        # --- 步骤 7: 松开并撤退 (Release & Retreat) ---
        planner.open_gripper()
        
        retreat_pose = final_place_pose * sapien.Pose([0, 0, -0.1])
        res = planner.move_to_pose_with_screw(retreat_pose)
        if res == -1: return False

        # 回到高处或者一个中间点，避免撞到下一个方块
        # 这里我们简单地保持在当前上方，直接进行下一个循环
        
        return True

    # ==============================================================================
    # 主任务逻辑：红 -> 绿 -> 蓝
    # ==============================================================================
    
    # 获取对象引用
    cube_red = env_unwrapped.cube_red
    cube_green = env_unwrapped.cube_green
    cube_blue = env_unwrapped.cube_blue
    
    # 获取目标位置 (第0个环境)
    target_pos_red = env_unwrapped.target_pos_red[0].cpu().numpy()
    target_pos_green = env_unwrapped.target_pos_green[0].cpu().numpy()
    target_pos_blue = env_unwrapped.target_pos_blue[0].cpu().numpy()

    # 1. 🟥 处理红色方块
    # print("Processing Red...")
    if not pick_and_place_back(cube_red, target_pos_red):
        planner.close()
        return -1

    # 2. 🟩 处理绿色方块
    # print("Processing Green...")
    if not pick_and_place_back(cube_green, target_pos_green):
        planner.close()
        return -1

    # 3. 🟦 处理蓝色方块
    # print("Processing Blue...")
    if not pick_and_place_back(cube_blue, target_pos_blue):
        planner.close()
        return -1

    planner.close()
    
    # ==============================================================================
    # 返回结果
    # ==============================================================================
    return [{
        "success": env.evaluate()["success"], 
        "elapsed_steps": torch.as_tensor(env.elapsed_steps)
    }]

if __name__ == "__main__":
    # 记得导入你的环境
    from mani_skill.envs.tasks.memory_dependence.pick_place_threetimes import PickPlaceThreetimesEnv
    
    # 使用新环境名
    env = gym.make("PickPlaceThreetimes-v1", render_mode="rgb_array", control_mode="pd_joint_pos")
    
    env = RecordEpisode(
        env, 
        "demos_pick_place_threetimes", 
        save_trajectory=False, 
        save_video=True, 
        info_on_video=True
    )
    
    # 服务器上请务必保持 vis=False
    res = solve(env, seed=0, debug=True, vis=False)
    print(f"Result: {res}")
    
    env.close()
    print("✅ 任务完成，视频已保存！")