from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class XArm6Stick(BaseAgent):
    uid = "xarm6_stick"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_stick.urdf"
    urdf_config = dict()
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    1.56280772e-03,
                    -1.10912404e00,
                    -9.71343926e-02,
                    1.52969832e-04,
                    1.20606723e00,
                    1.66234924e-03,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    ee_link_name = "panda_hand_tcp"  # 新末端执行器点

    arm_stiffness = 1e4
    arm_damping = 1e3
    arm_friction = 0.1
    arm_force_limit = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        pd_joint_target_delta_pos = deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # PD ee position - 使用新的stick末端
        pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        pd_ee_target_delta_pos = deepcopy(pd_ee_delta_pos)
        pd_ee_target_delta_pos.use_target = True
        pd_ee_target_delta_pose = deepcopy(pd_ee_delta_pose)
        pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=pd_joint_delta_pos),
            pd_joint_pos=dict(arm=pd_joint_pos),
            pd_ee_delta_pos=dict(arm=pd_ee_delta_pos),
            pd_ee_delta_pose=dict(arm=pd_ee_delta_pose),
            pd_ee_pose=dict(arm=pd_ee_pose),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(arm=pd_joint_target_delta_pos),
            pd_ee_target_delta_pos=dict(arm=pd_ee_target_delta_pos),
            pd_ee_target_delta_pose=dict(arm=pd_ee_target_delta_pose),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=pd_joint_vel),
            pd_joint_pos_vel=dict(arm=pd_joint_pos_vel),
            pd_joint_delta_pos_vel=dict(arm=pd_joint_delta_pos_vel),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        # 用于接触力查询
        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class XArm6RobotiqWristCamera(XArm6Stick):
    uid = "xarm6_stick_wristcam"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            ),
            CameraConfig(
                uid="third_view_camera",
                # pose=sapien.Pose(p=[0.6, 0.7, 0.6], q=[0.707, 0, 0.707, 0]),
                # pose=sapien.Pose(p=[0.6, 0.7, 0.6], q=[0.707, 0, 0.707, 0]),
                pose=sapien_utils.look_at(eye=[0.5, -0.5, 0.4], target=[0.0, 0.0, 0.15]),
                width=256,
                height=256,
                fov=1,
                near=0.01,
                far=100,
                mount=None,  # 不绑定到任何连杆，固定在世界坐标系
            ),
        ]
