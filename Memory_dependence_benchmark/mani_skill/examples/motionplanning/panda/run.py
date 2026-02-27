import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube, solveDrawTriangle, solveDrawSVG, solvePlaceSphere

# [新增] 导入你的 solve 函数
from mani_skill.examples.motionplanning.panda.solutions.swap_cube import solve as solveSwapCube
from mani_skill.examples.motionplanning.panda.solutions.pick_place_threetimes import solve as solvePickPlaceThreetimes
from mani_skill.examples.motionplanning.panda.solutions.push_cube_with_signal import solve as solvePushCubeWithSignal
from mani_skill.examples.motionplanning.panda.solutions.identification import solve as solveIdentification
from mani_skill.examples.motionplanning.panda.solutions.teacher_arm_shuffle import solve as solveTeacherArmShuffle

import mani_skill.envs.tasks.memory_dependence

MP_SOLUTIONS_LIST = {
    "SwapThreeCubes-v1": solveSwapCube, 
    "PickPlaceThreetimes-v1": solvePickPlaceThreetimes,  
    "PushCubeWithSignal-v1": solvePushCubeWithSignal, 
    "Identification-v1": solveIdentification, 
    "PegInsertionSide-v1": solvePegInsertionSide,
    "TeacherArmShuffle-v1": solveTeacherArmShuffle,
}
ROBOT_UIDS_LIST = {
    "SwapThreeCubes-v1": "panda_wristcam", 
    "PickPlaceThreetimes-v1": "panda_wristcam", 
    "PushCubeWithSignal-v1": "panda_wristcam", 
    "Identification-v1": "panda_wristcam",
    "PegInsertionSide-v1": "panda_wristcam",
    "TeacherArmShuffle-v1": "panda_wristcam",
}
REWARD_MODE_LIST = {
    "SwapThreeCubes-v1": "dense",
    "PickPlaceThreetimes-v1": "dense", 
    "PushCubeWithSignal-v1": "dense",
    "Identification-v1": "dense",
    "PegInsertionSide-v1": "dense",
    "TeacherArmShuffle-v1": "dense",
}

# MP_SOLUTIONS_LIST = {
#     "DrawTriangle-v1": solveDrawTriangle,
#     "DrawSVG-v1": solveDrawSVG,
#     "PickCube-v1": solvePickCube,
#     "StackCube-v1": solveStackCube,
#     "PegInsertionSide-v1": solvePegInsertionSide,
#     "PlugCharger-v1": solvePlugCharger,
#     "PlaceSphere-v1": solvePlaceSphere,
#     "PushCube-v1": solvePushCube,
#     "PullCubeTool-v1": solvePullCubeTool,
#     "LiftPegUpright-v1": solveLiftPegUpright,
#     "PullCube-v1": solvePullCube,
    
# }
# ROBOT_UIDS_LIST = {
#     "PickCube-v1": "panda_wristcam",
#     "PushCube-v1": "panda_wristcam",
#     "StackCube-v1": "panda_wristcam",
#     "PlugCharger-v1": "panda_wristcam",
#     "PullCube-v1": "panda_wristcam",
#     "PullCubeTool-v1": "panda_wristcam",
#     "DrawTriangle-v1": "panda_stick_wristcam",
#     "DrawSVG-v1": "panda_stick_wristcam",
#     "PlaceSphere-v1": "panda_wristcam",
#     "LiftPegUpright-v1": "panda_wristcam",
#     "PegInsertionSide-v1": "panda_wristcam",
# }
# REWARD_MODE_LIST = {
#     "PickCube-v1": "dense",
#     "PushCube-v1": "dense",
#     "StackCube-v1": "dense",
#     "PlugCharger-v1": "dense",
#     "PullCube-v1": "dense",
#     "PullCubeTool-v1": "dense",
#     "DrawTriangle-v1": "sparse",
#     "DrawSVG-v1": "sparse",
#     "PlaceSphere-v1": "dense",
#     "LiftPegUpright-v1": "dense",
#     "PegInsertionSide-v1": "dense",
# }

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="DrawTriangle-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS_LIST.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", default=True, action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="/home/chenyipeng/ManiSkill/data/panda", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    return parser.parse_args()

# def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
#     env_id = args.env_id
#     env = gym.make(
#         env_id,
#         obs_mode=args.obs_mode,
#         control_mode="pd_joint_pos",
#         render_mode=args.render_mode,
#         robot_uids=ROBOT_UIDS_LIST[env_id],
#         reward_mode=REWARD_MODE_LIST[env_id],
#         sensor_configs=dict(shader_pack=args.shader),
#         human_render_camera_configs=dict(shader_pack=args.shader),
#         viewer_camera_configs=dict(shader_pack=args.shader),
#         sim_backend=args.sim_backend
#     )
#     if env_id not in MP_SOLUTIONS_LIST:
#         raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS_LIST.keys())}")

#     if not args.traj_name:
#         new_traj_name = time.strftime("%Y%m%d_%H%M%S")
#     else:
#         new_traj_name = args.traj_name

#     if args.num_procs > 1:
#         new_traj_name = new_traj_name + "." + str(proc_id)
#     env = RecordEpisode(
#         env,
#         output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
#         trajectory_name=new_traj_name, save_video=args.save_video,
#         source_type="motionplanning",
#         source_desc="official motion planning solution from ManiSkill contributors",
#         video_fps=30,
#         record_reward=False,
#         save_on_reset=False
#     )
#     output_h5_path = env._h5_file.filename
#     solve = MP_SOLUTIONS_LIST[env_id]
#     print(f"Motion Planning Running on {env_id}")
#     pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
#     seed = start_seed
#     successes = []
#     solution_episode_lengths = []
#     failed_motion_plans = 0
#     passed = 0
#     while True:
#         try:
#             res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
#         except Exception as e:
#             print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
#             res = -1

#         if res == -1:
#             success = False
#             failed_motion_plans += 1
#         else:
#             success = res[-1]["success"].item()
#             elapsed_steps = res[-1]["elapsed_steps"].item()
#             solution_episode_lengths.append(elapsed_steps)
#         successes.append(success)
#         if args.only_count_success and not success:
#             seed += 1
#             env.flush_trajectory(save=False)
#             if args.save_video:
#                 env.flush_video(save=False)
#             continue
#         else:
#             env.flush_trajectory()
#             if args.save_video:
#                 env.flush_video()
#             pbar.update(1)

#             pbar.set_postfix(
#                 dict(
#                     success_rate=np.mean(successes),
#                     failed_motion_plan_rate=failed_motion_plans / (seed + 1),
#                     avg_episode_length=np.mean(solution_episode_lengths),
#                     max_episode_length=np.max(solution_episode_lengths),
#                     # min_episode_length=np.min(solution_episode_lengths)
#                 )
#             )
#             seed += 1
#             passed += 1
#             if passed == args.num_traj:
#                 break
#     env.close()
#     return output_h5_path

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        robot_uids=ROBOT_UIDS_LIST[env_id],
        reward_mode=REWARD_MODE_LIST[env_id],
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend
    )
    if env_id not in MP_SOLUTIONS_LIST:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS_LIST.keys())}")

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    
    # 初始化录像包装器
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, 
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    
    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS_LIST[env_id]
    print(f"Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0 # 这是当前进程内已成功的计数器 (0, 1, 2, 3, 4, 5)
    
    while True:
        try:
            res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        except Exception as e:
            print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        
        successes.append(success)
        
        if args.only_count_success and not success:
            # 失败情况：不保存
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            # 成功情况：保存数据
            env.flush_trajectory()
            
            if args.save_video:
                # ----------------------------------------------------------
                # [核心修改] 计算全局唯一的 ID (0-47)
                # 公式：(进程ID * 每个进程的任务数) + 当前进程已完成数
                # ----------------------------------------------------------
                global_video_id = proc_id * args.num_traj + passed
                
                # 转换为字符串，RecordEpisode 会自动加上 .mp4
                simple_video_name = str(global_video_id)
                
                # 显式传入 name，覆盖默认逻辑
                env.flush_video(name=simple_video_name)
                
                # 可选：打印一下
                # tqdm.write(f"✅ Video saved: {simple_video_name}.mp4")
            
            pbar.update(1)

            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)

if __name__ == "__main__":
    # start = time.time()
    mp.set_start_method("spawn")
    main(parse_args())
    # print(f"Total time taken: {time.time() - start}")
