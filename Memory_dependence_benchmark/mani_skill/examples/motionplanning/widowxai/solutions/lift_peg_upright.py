import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import LiftPegUprightEnv
from mani_skill.examples.motionplanning.widowxai.motionplanner import WidowXArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def main():
    env: LiftPegUprightEnv = gym.make(
        "LiftPegUpright-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()

def solve(env: LiftPegUprightEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = WidowXArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    peg_init_pose = env.peg.pose

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    # offset = sapien.Pose([0.10, 0, 0])
    offset = sapien.Pose([0.0, 0, 0.04])
    grasp_pose = grasp_pose * offset

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.01])
    res = planner.move_to_pose_with_RRTStar(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    planner.close_gripper(gripper_state=-0.12)

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    # lift_pose = sapien.Pose([0, 0, 0.30]) * grasp_pose
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Place upright
    # -------------------------------------------------------------------------- #
    theta = np.pi/15
    # rotation_quat = np.array([np.cos(theta), 0, np.sin(theta), 0])  
    rotation_quat = np.array([np.cos(theta), 0, -np.sin(theta), 0])
    
    final_pose = lift_pose * sapien.Pose(
        p=[0, 0, 0],
        q=rotation_quat
    )
    res = planner.move_to_pose_with_screw(final_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose([0, 0, -0.04]) * final_pose
    res = planner.move_to_pose_with_screw(lower_pose)
    if res == -1: return res

    planner.close()
    
    planner.open_gripper()
    return res

if __name__ == "__main__":
    main()