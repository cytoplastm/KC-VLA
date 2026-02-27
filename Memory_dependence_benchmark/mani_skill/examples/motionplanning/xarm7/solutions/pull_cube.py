import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.xarm7.motionplanner import XArm7RobotiqMotionPlanningSolver

def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    planner = XArm7RobotiqMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.close_gripper(gripper_state=0.5)
    # planner.close_gripper()
    print('+'*30)

    # -------------------------------------------------------------------------- #
    # Move to initial pose
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.05, 0, 0.1]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_RRTStar(reach_pose)

    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([0.05, 0, 0.02]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
