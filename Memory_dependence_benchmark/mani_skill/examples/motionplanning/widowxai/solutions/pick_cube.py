import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
# from mani_skill.examples.motionplanning.xarm6.motionplanner import \
#     XArm6RobotiqMotionPlanningSolver, XArm6PandaGripperMotionPlanningSolver
from mani_skill.examples.motionplanning.widowxai.motionplanner import \
    WidowXArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
# from pdb import set_trace as st

def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    planner = WidowXArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.045
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1]) 
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information
    x = env.agent.tcp.pose[0]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([-0.05, 0, 0])
    # res = planner.move_to_pose_with_screw(reach_pose)
    res = planner.move_to_pose_with_RRTStar(grasp_pose)
    # -------------------------------------------------------------------------- #
    # Grasp
    # # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_RRTStar(grasp_pose)
    obs, reward, terminated, truncated, info = planner.close_gripper(gripper_state=-0.5)
    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    # relative_z = (env.goal_site.pose.sp.p[2] - grasp_pose.p[2])
    # print(relative_z)
    # goal_pose = sapien.Pose(env.goal_site.pose.sp.p + np.array([0, 0, -relative_z+relative_z*0.5]), grasp_pose.q)
    # center = env.goal_site.pose.p           # 目标位置
    # goal_pose = env.agent.build_grasp_pose(approaching, closing, env.goal_site.pose.sp.p)
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_RRTStar(goal_pose)

    planner.close()
    return res

