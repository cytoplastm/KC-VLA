import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import \
    XArm6RobotiqMotionPlanningSolver, XArm6PandaGripperMotionPlanningSolver

def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    planner = XArm6RobotiqMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    planner.close_gripper()

    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0.1]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_RRTStar(reach_pose)

    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p,q=env.agent.tcp.pose.sp.q)
    offset = sapien.Pose([0, 0, -0.02])
    goal_pose = goal_pose * offset
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
