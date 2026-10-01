# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen copy of main's unkeyed Franka IK and joint-position embodiments.

This file pins main's (upstream commit 7d75c9593) Franka configuration classes, the Franka
constructor, and the getters the instance-key change rewrote, so a parity test can compare the
unkeyed production configuration against them. The constants, the mimic environment, the gripper,
the recorder builder, and the base-class behavior this change left alone come from production.
Keep this file frozen when the production code changes.
"""

from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import (
    _DEFAULT_CAMERA_OFFSET,
    _FRANKA_JOINT_NAMES,
    _FRANKA_READY_POSE,
    _FRANKA_ROBOT_PRIM,
    _FRANKA_STAND_PRIM,
    FrankaMimicEnv,
)
from isaaclab_arena.embodiments.franka.observations import gripper_pos
from isaaclab_arena.embodiments.gripper import PandaGripper
from isaaclab_arena.embodiments.robot_on_stand_utils import compose_on_stand_usd
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
from isaaclab_arena.utils.pose import Pose


class FrozenFrankaEmbodimentBase(EmbodimentBase):
    """Unkeyed Franka embodiment built with the pre-change constructor and getters.

    Args:
        robot_cfg: The arm articulation, placed on the stand at the unkeyed prim path.
        action_cfg_type: The action configclass, instantiated with its unkeyed defaults.
        enable_cameras: Whether the wrist camera is added to the scene and observations.
    """

    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(self, robot_cfg: ArticulationCfg, action_cfg_type: type, enable_cameras: bool = False):
        super().__init__(enable_cameras=enable_cameras)
        self.gripper = PandaGripper()
        self.event_config = FrankaEventCfg()
        self.reward_config = FrankaRewardsCfg()
        self.mimic_env = FrankaMimicEnv
        self.camera_config = FrankaCameraCfg()
        self.scene_config = FrankaSceneCfg()
        self.observation_config = FrankaObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.add_camera_variations(self.camera_config)
        self.scene_config.robot = _franka_robot_cfg_on_stand(robot_cfg.copy())
        self.action_config = action_cfg_type()

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        robot = self.scene_config.robot
        robot.init_state = robot.init_state.replace(joint_pos=dict(zip(_FRANKA_JOINT_NAMES, initial_joint_pose)))

    def get_events_cfg(self) -> Any:
        if self._pose_event_cfg is None:
            return self.event_config
        pose_reset_cfg = make_configclass(
            "EmbodimentPoseResetCfg",
            [("robot_reset_pose", EventTerm, self._pose_event_cfg)],
        )()
        return combine_configclass_instances("EventsCfg", self.event_config, pose_reset_cfg)

    def get_recorder_term_cfg(self, record_trajectories: bool = False) -> Any:
        if not record_trajectories:
            return None
        from isaaclab_arena.terms.recorders import make_trajectory_recorder_terms_cfg

        return make_trajectory_recorder_terms_cfg(frame_transformer_names=["ee_frame"], asset_name="robot")

    def get_scene_key(self) -> str:
        return "robot"

    def get_ee_frame_transformer_names(self) -> list[str]:
        return ["ee_frame"]

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        scene_config.robot.init_state.pos = pose.position_xyz
        scene_config.robot.init_state.rot = pose.rotation_xyzw
        return scene_config


class FrozenFrankaIKEmbodiment(FrozenFrankaEmbodimentBase):
    """Unkeyed Franka with differential IK arm control and high-PD defaults, as on main."""

    name = "franka_ik"
    tags = ["embodiment", "default"]

    def __init__(self, enable_cameras: bool = False):
        super().__init__(FRANKA_PANDA_HIGH_PD_CFG, FrankaIKActionCfg, enable_cameras=enable_cameras)

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name


class FrozenFrankaJointPosEmbodiment(FrozenFrankaEmbodimentBase):
    """Unkeyed Franka with joint-position arm control and standard PD gains, as on main."""

    name = "franka_joint_pos"

    def __init__(self, enable_cameras: bool = False):
        super().__init__(FRANKA_PANDA_CFG, FrankaJointPosActionsCfg, enable_cameras=enable_cameras)

    def get_command_body_name(self) -> str:
        return "panda_hand"


@configclass
class FrankaIKActionCfg:
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaJointPosActionsCfg:
    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=False,
        offset={
            name: value
            for name, value in FRANKA_PANDA_CFG.init_state.joint_pos.items()
            if name.startswith("panda_joint")
        },
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaSceneCfg:
    robot: ArticulationCfg | None = None

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class FrankaCameraCfg(ArenaCameraCfg):
    wrist_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_DEFAULT_CAMERA_OFFSET.position_xyz,
            rot=_DEFAULT_CAMERA_OFFSET.rotation_xyzw,
            convention="ros",
        ),
    )


@configclass
class FrankaObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaEventCfg:
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FrankaRewardsCfg:
    action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(
        func=mdp_isaac_lab.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )


def _franka_robot_cfg_on_stand(robot_cfg: ArticulationCfg) -> ArticulationCfg:
    cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.init_state = cfg.init_state.replace(joint_pos=_FRANKA_READY_POSE)
    cfg.spawn.usd_path = compose_on_stand_usd(
        _FRANKA_ROBOT_PRIM,
        _FRANKA_STAND_PRIM,
        stand_height_m=_FRANKA_STAND_PRIM.stand_default_height,
        output_basename="franka_panda_on_stand",
    )
    return cfg
