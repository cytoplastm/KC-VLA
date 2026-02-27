from copy import deepcopy
import numpy as np
import sapien.core as sapien
import torch

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

@register_agent(asset_download_ids=["xarm6"])
class XArm6AllegroLeft(BaseAgent):
    uid = "xarm6_allegro_left"
    urdf_path = f"{ASSET_DIR}/robots/xarm6/xarm6_allegro_left.urdf"

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            link_3_0_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            link_7_0_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            link_11_0_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            link_15_0_tip=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0, 0.22, -1.23, 0, 1.01, 0,  # Arm joints
                    0, 0, 0, 0,  # Index finger
                    0, 0, 0, 0,  # Middle finger
                    0, 0, 0, 0,  # Ring finger
                    0.263, 0, 0, 0  # Thumb
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.zeros(16),
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

    arm_stiffness = 1e4
    arm_damping = 1e3
    arm_friction = 0.1
    arm_force_limit = 100

    hand_stiffness = 1e5
    hand_damping = 2000
    hand_force_limit = 0.1
    hand_friction = 1
    ee_link_name = "link_15_0_tip"  # Using thumb tip as EE for now

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # Arm controllers (same as XArm6Robotiq)
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            normalize_action=False,
        )

        # Hand controllers
        hand_joint_names = [
            "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",  # Index
            "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",  # Middle
            "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",  # Ring
            "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"  # Thumb
        ]

        hand_pd_joint_pos = PDJointPosControllerConfig(
            hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.hand_stiffness,
            damping=self.hand_damping,
            force_limit=self.hand_force_limit,
            friction=self.hand_friction,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                hand=hand_pd_joint_pos,
            ),
            # Add other controller configurations as needed
        )

        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        # Disable collisions between hand parts
        hand_links = [
            f"link_{i}.0" for i in range(16)
        ] + ["base_link", "link_15.0_tip", "link_11.0_tip", "link_7.0_tip", "link_3.0_tip"]
        for link_name in hand_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose