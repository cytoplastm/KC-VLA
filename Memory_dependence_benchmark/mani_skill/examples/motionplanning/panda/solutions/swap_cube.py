import numpy as np
import sapien
import gymnasium as gym
import torch
import mani_skill.envs.tasks.memory_dependence 

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers import RecordEpisode

# 定义 Panda 机械臂的标准展开姿态 (Home Pose)
# 这是一个经典的“准备工作”姿态，手臂高高抬起，远离桌面
HOME_QPOS = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0] # 最后两个0是夹爪

def solve(env, seed=None, debug=False, vis=False):
    # 1. 重置环境
    env.reset(seed=seed)
    
    # 2. 初始化机械臂姿态
    # print("🤖 正在初始化机械臂姿态 (Unfolding)...") 
    # 如果你在 collectdata.sh 里并发运行，建议把 print 注释掉减少刷屏
    # env.unwrapped.agent.robot.set_qpos(HOME_QPOS)
    # env.step(env.action_space.sample()) 
    
    # 3. 初始化规划器
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env = env.unwrapped
    FINGER_LENGTH = 0.025
    CUBE_HALF_SIZE = 0.02
    CUBE_SIZE = 0.04

    def pick_and_place(obj_to_pick, target_pos_np):
        # print(f"\n👉 [操作] 准备移动: {obj_to_pick.name}")
        
        # --- 步骤 1: 计算抓取位姿 ---
        obb = get_actor_obb(obj_to_pick)
        approaching = np.array([0, 0, -1]) 
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, obj_to_pick.pose.sp.p)

        # --- 步骤 2: 预备抓取 (Reach) ---
        # 悬停在方块上方 5cm
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        
        # print("  -> 规划路径: Reach...")
        # 【改回 Screw】因为现在机械臂已经展开了，走直线应该没问题
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: 
            print(f"  ❌ Reach 失败！(请检查是否离目标太远)")
            return False

        # --- 步骤 3: 下落抓取 (Approach & Grasp) ---
        # print("  -> 规划路径: Approach...")
        res = planner.move_to_pose_with_screw(grasp_pose)
        if res == -1: return False
        
        # print("  -> 执行: 闭合夹爪")
        planner.close_gripper()

        # --- 步骤 4: 抬起 (Lift) ---
        # 抬高 20cm
        lift_pose = grasp_pose * sapien.Pose([0, 0, -0.2]) 
        # print("  -> 规划路径: Lift...")
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1: return False

        # --- 步骤 5: 移动到目标上方 (Hover) ---
        place_z = target_pos_np[2] + CUBE_HALF_SIZE + 0.002
        hover_pose = sapien.Pose([target_pos_np[0], target_pos_np[1], lift_pose.p[2]], lift_pose.q)
        
        # print("  -> 规划路径: Move to Target...")
        res = planner.move_to_pose_with_screw(hover_pose) 
        if res == -1: return False

        # --- 步骤 6: 下放到位 (Place) ---
        final_place_pose = sapien.Pose([target_pos_np[0], target_pos_np[1], place_z], lift_pose.q)
        # print("  -> 规划路径: Place Down...")
        res = planner.move_to_pose_with_screw(final_place_pose)
        if res == -1: return False

        # --- 步骤 7: 松开并撤退 (Release & Retreat) ---
        # print("  -> 执行: 松开夹爪")
        planner.open_gripper()
        
        retreat_pose = final_place_pose * sapien.Pose([0, 0, -0.1])
        # print("  -> 规划路径: Retreat...")
        res = planner.move_to_pose_with_screw(retreat_pose)
        
        return True

    # ==============================================================================
    # 主任务逻辑
    # ==============================================================================
    
    # 1. 感知
    all_cubes = [env.cubeA, env.cubeB, env.cubeC]
    all_cubes.sort(key=lambda actor: actor.pose.p[0, 2].item())
    obj_bottom = all_cubes[0]
    obj_middle = all_cubes[1]
    obj_top    = all_cubes[2]

    current_bot_pos = obj_bottom.pose.p[0].cpu().numpy()
    TABLE_Z = current_bot_pos[2] - CUBE_HALF_SIZE 

    # 🟢 [新增] 获取当前方块堆叠的中心 XY 坐标
    stack_center_x = current_bot_pos[0]
    stack_center_y = current_bot_pos[1]
    
    #原圆心在(0,0)
    # 2. 生成随机点
    # def get_random_annulus_point(r_min=0.1, r_max=0.2):
    #     theta = np.random.uniform(0, 2 * np.pi)
    #     r = np.random.uniform(r_min, r_max)
    #     x = r * np.cos(theta)
    #     y = r * np.sin(theta)
    #     return x, y
    
    # while True:
    #     cx, cy = get_random_annulus_point()
    #     bx, by = get_random_annulus_point()
    #     if np.sqrt((cx - bx)**2 + (cy - by)**2) >= 0.1:
    #         break

    #新
    def get_random_annulus_point(center_x, center_y, r_min=0.1, r_max=0.15):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(r_min, r_max)
        
        # 🟢 [核心修改] 加上圆心偏移量
        x = center_x + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        return x, y

    while True:
        # 🟢 [修改] 传入方块堆叠的中心作为圆心
        cx, cy = get_random_annulus_point(stack_center_x, stack_center_y)
        bx, by = get_random_annulus_point(stack_center_x, stack_center_y)
        
        # 计算两点距离
        distance = np.sqrt((cx - bx)**2 + (cy - by)**2)
        # 🟢 [修改] 同时满足大于等于 0.1 且小于 0.15
        if distance >= 0.1 and distance < 0.15:
            break
    
    temp_pos_top = np.array([cx, cy, TABLE_Z])
    new_base_pos = np.array([bx, by, TABLE_Z])

    # 3. 执行序列 (【关键修改】：检测返回值)
    # 如果任何一步返回 False，直接返回 -1，这样 collectdata 就会记录为失败并重试
    
    if not pick_and_place(obj_top, temp_pos_top): 
        planner.close()
        return -1
        
    if not pick_and_place(obj_middle, new_base_pos):
        planner.close()
        return -1
        
    pos_on_new_base = np.array([new_base_pos[0], new_base_pos[1], TABLE_Z + CUBE_SIZE])
    if not pick_and_place(obj_bottom, pos_on_new_base):
        planner.close()
        return -1
        
    pos_on_top = np.array([new_base_pos[0], new_base_pos[1], TABLE_Z + CUBE_SIZE * 2])
    if not pick_and_place(obj_top, pos_on_top):
        planner.close()
        return -1

    planner.close()
    
    # ==============================================================================
    # 【核心修复】：返回 run.py 能读懂的数据格式
    # ==============================================================================
    # run.py 需要一个列表，且最后一项包含 "success" 和 "elapsed_steps"
    return [{
        "success": env.evaluate()["success"], 
        "elapsed_steps": torch.as_tensor(env.elapsed_steps)
    }]

if __name__ == "__main__":
    # 创建环境
    env = gym.make("SwapThreeCubes-v1", render_mode="rgb_array", control_mode="pd_joint_pos")
    # 设置录制包装器
    # max_steps_per_episode 设大一点防止还没做完就被截断（虽然 solve 内部控制了流程）
    env = RecordEpisode(
        env, 
        output_dir="demos_swap_cube", 
        save_trajectory=False, 
        save_video=True, 
        info_on_video=True
    )
    
    NUM_EPISODES = 10
    
    for i in range(NUM_EPISODES):
        print(f"\n🎬 正在生成第 {i+1}/{NUM_EPISODES} 个演示 (Seed={i})...")
        
        # 传入不同的 seed 以保证每次方块的初始位置都不一样
        result = solve(env, seed=i, debug=False, vis=False)
        
        if result == -1:
            print(f"⚠️ 第 {i+1} 个演示生成失败 (规划错误)，跳过或重试...")
        else:
            print(f"✅ 第 {i+1} 个演示生成成功！")
    
    env.close()
    print(f"\n🎉 全部完成！共生成 {NUM_EPISODES} 个视频，保存在 demos_swap_cube 文件夹下。")