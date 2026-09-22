# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise robot composition, recorded identity, and single-robot compatibility."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def make_two_robot_definition(enable_cameras=False, mixed=False, relations=False):
    """Build two spaced robots with optional cameras and placement relations."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    robots = [FrankaJointPosEmbodiment(instance_key="left", enable_cameras=enable_cameras)]
    robots.append(
        registry.get_asset_by_name("g1_wbc_joint")()
        if mixed
        else FrankaJointPosEmbodiment(instance_key="right", enable_cameras=enable_cameras)
    )
    for index, robot in enumerate(robots):
        x = -2.0 if index == 0 else 2.0
        if relations:
            robot.add_relation(AtPosition(x=x, y=0.0, z=0.0))
        else:
            robot.set_initial_pose(
                Pose(
                    position_xyz=(x, 0.0, 0.78 if mixed and index else 0.0),
                    rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                )
            )
        if enable_cameras:
            robot.get_variations()[0].enable()
    assets = [registry.get_asset_by_name(name)() for name in ("ground_plane", "light")]
    if relations:
        anchor = registry.get_asset_by_name("dex_cube")()
        anchor.set_initial_pose(Pose(position_xyz=(0.0, 5.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        anchor.add_relation(IsAnchor())
        assets.append(anchor)
    return IsaacLabArenaEnvironment(
        name="multi_robot_test",
        scene=Scene(assets=assets),
        embodiments=robots,
        placer_params=ObjectPlacerParams(min_unique_layouts_per_env=1),
    )


def _test_two_robots(simulation_app, output_dir, cameras=False, mixed=False):
    import h5py
    import json
    import torch
    from dataclasses import fields
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.terms.recorders import TrajectoryRecorderTermsBaseCfg

    definition = make_two_robot_definition(enable_cameras=cameras, mixed=mixed, relations=cameras)
    if not mixed:
        definition.embodiments.reverse()
    builder = ArenaEnvBuilder(
        definition,
        ArenaEnvBuilderCfg(
            num_envs=2,
            solve_relations=cameras,
            record_trajectories=True,
            recorder_dataset_export_dir_path=str(output_dir),
            recorder_dataset_filename="two_robots",
        ),
    )
    with patch(
        "isaaclab_arena.environments.arena_env_builder.solve_and_apply_relation_placement",
        wraps=solve_and_apply_relation_placement,
    ) as solve:
        env = builder.make_registered()
        if cameras:
            assert {"left", "right"} <= {asset.get_scene_key() for asset in solve.call_args.args[0]}
    path = output_dir / "episodes.jsonl"
    keys = {robot.get_scene_key() for robot in definition.embodiments}
    try:
        env.unwrapped.episode_recorder.set_output_path(path)
        observations, _ = env.reset()
        assert set(env.unwrapped.scene.articulations) == keys
        if mixed:
            assert env.unwrapped.cfg.events.reset_all.params["asset_cfg"].name == "robot"
            assert "asset_cfg" not in definition.embodiments[1].event_config.reset_all.params
            left = env.unwrapped.scene["left"]
            assert not torch.equal(left.data.joint_pos.torch, left.data.default_joint_pos.torch)
        if cameras:
            for key, x in (("left", -2.0), ("right", 2.0)):
                position = env.unwrapped.scene[key].data.root_pos_w.torch - env.unwrapped.scene.env_origins
                assert torch.allclose(position[:, 0], torch.full_like(position[:, 0], x), atol=0.02)
            assert set(observations["camera_obs"]) == {"left_wrist_cam_rgb", "right_wrist_cam_rgb"}
            assert env.unwrapped.cfg.demo_recorder_config is not None
            assert {"left", "right"} <= set(builder.get_all_variations())
            assert (
                env.unwrapped.cfg.events.left_wrist_cam_extrinsics_variation.params["asset_cfg"].name
                == "left_wrist_cam"
            )
            assert (
                env.unwrapped.cfg.events.right_wrist_cam_extrinsics_variation.params["asset_cfg"].name
                == "right_wrist_cam"
            )
        recorders = env.unwrapped.cfg.recorders
        for field in fields(TrajectoryRecorderTermsBaseCfg):
            assert getattr(recorders, field.name) is not None
            assert not hasattr(recorders, f"left_{field.name}")
            assert not hasattr(recorders, f"right_{field.name}")
        assert recorders.left_record_end_effector_poses_0.asset_name == "left"
        if not mixed:
            assert recorders.right_record_end_effector_poses_0.asset_name == "right"
        assert env.action_space.shape[-1] == sum(env.unwrapped.action_manager.action_term_dim)
        if not mixed:
            assert env.unwrapped.action_manager.active_terms == [
                "right_arm_action",
                "right_gripper_action",
                "left_arm_action",
                "left_gripper_action",
            ]
        widths = {}
        for key in keys:
            widths[key] = sum(
                width
                for name, width in zip(
                    env.unwrapped.action_manager.active_terms, env.unwrapped.action_manager.action_term_dim, strict=True
                )
                if env.unwrapped.action_manager.get_term(name).cfg.asset_name == key
            )
        assert all(width > 0 for width in widths.values())
        if not mixed:
            assert widths == {"left": 8, "right": 8}
        for _ in range(2):
            observations, _, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        if not mixed:
            manager = env.unwrapped.action_manager
            manager.action.zero_()
            manager.prev_action.zero_()
            manager.action[:, :8] = 1.0
            rewards = env.unwrapped.cfg.rewards
            assert torch.equal(
                rewards.right_action_rate.func(env.unwrapped, **rewards.right_action_rate.params),
                torch.full((2,), 8.0, device=env.unwrapped.device),
            )
            assert torch.equal(
                rewards.left_action_rate.func(env.unwrapped, **rewards.left_action_rate.params),
                torch.zeros(2, device=env.unwrapped.device),
            )
        if mixed:
            assert observations["policy"]["actions"].shape[-1] == widths["robot"]
        env.reset()
    finally:
        env.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    expected = {robot.get_scene_key(): robot.embodiment_type for robot in definition.embodiments}
    assert all(record["embodiments"] == expected for record in records)
    if not mixed:
        with h5py.File(output_dir / "two_robots.hdf5", "r") as dataset:
            assert len(dataset["data"]) == 2
            for demo in dataset["data"].values():
                for key in ("left", "right"):
                    assert demo[f"states/kinematics/{key}_end_effector/position"].shape == (2, 3)
                    assert demo[f"initial_state/kinematics/{key}_end_effector/position"].shape == (1, 3)
    if cameras:
        for record in records:
            assert any(name.startswith("left.") for name in record["variations"])
            assert any(name.startswith("right.") for name in record["variations"])
    return True


@pytest.mark.with_cameras
def test_two_keyed_frankas_with_cameras_and_relations(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_two_robots, enable_cameras=True, cameras=True, output_dir=tmp_path
    )


def test_keyed_franka_with_unkeyed_g1(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_two_robots, mixed=True, output_dir=tmp_path)


def _test_single_compatibility(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tests.utils.legacy_single_embodiment_builder import LegacySingleEmbodimentBuilder
    from isaaclab_arena_environments.cube_goal_pose_environment import (
        CubeGoalPoseEnvironment,
        CubeGoalPoseEnvironmentCfg,
    )

    cfg = ArenaEnvBuilderCfg(
        num_envs=2, solve_relations=False, record_trajectories=True, recorder_dataset_filename="single_reference"
    )
    factory = CubeGoalPoseEnvironment()
    before, _ = LegacySingleEmbodimentBuilder(
        factory.build(CubeGoalPoseEnvironmentCfg(enable_cameras=True)), cfg
    ).compose_manager_cfg()
    after, _ = ArenaEnvBuilder(
        factory.build(CubeGoalPoseEnvironmentCfg(enable_cameras=True)), cfg
    ).compose_manager_cfg()
    assert after.to_dict() == before.to_dict()
    assert after.episode_recorders.core.params["embodiments"] == {"robot": "franka_ik"}
    return True


def test_single_robot_matches_prechange_assembly():
    assert run_function_with_persistent_simulation_app(_test_single_compatibility)


def _test_invalid_compositions(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    for robots in (
        [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()],
        [FrankaJointPosEmbodiment(instance_key="arm"), FrankaJointPosEmbodiment(instance_key="arm")],
    ):
        with pytest.raises(AssertionError, match="unique|unkeyed"):
            IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=robots)
    with pytest.raises(AssertionError, match="not both"):
        IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiment=FrankaJointPosEmbodiment(), embodiments=[])
    unkeyed = [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()]
    unkeyed[1].get_scene_key = lambda: "custom_articulation"
    with pytest.raises(AssertionError, match="unkeyed"):
        IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=unkeyed)
    definition = make_two_robot_definition()
    with pytest.raises(AssertionError, match="Use embodiments"):
        _ = definition.embodiment
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(mimic=True)).compose_manager_cfg()
    definition.teleop_device = object()
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.teleop_device = None
    with patch(
        "isaaclab_arena.environments.arena_env_builder.get_settings_manager",
        return_value=SimpleNamespace(get=lambda *args: True),
    ):
        with pytest.raises(AssertionError, match="XR require exactly one"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.embodiments.append(definition.embodiments[0])
    with pytest.raises(AssertionError, match="unique"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.embodiment = FrankaJointPosEmbodiment()
    assert definition.embodiments == [definition.embodiment]
    definition.embodiment = None
    assert definition.embodiments == []
    definition.embodiment = FrankaJointPosEmbodiment(instance_key="prepared")
    builder = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(solve_relations=False))
    prepared, _ = builder.compose_manager_cfg()
    builder.cfg.mimic = True
    with pytest.raises(AssertionError, match="without an instance key"):
        builder.get_entry_point()
    with pytest.raises(AssertionError, match="without an instance key"):
        builder.build_registered(prepared)
    return True


def test_invalid_compositions_fail_before_building():
    assert run_function_with_persistent_simulation_app(_test_invalid_compositions)


def _test_observation_precedence(simulation_app):
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.cameras import combine_observation_cfgs
    from isaaclab_arena.utils.configclass import make_configclass

    ordinary_type = make_configclass("PolicyObservations", [("value", int, 1)], bases=(ObservationGroupCfg,))
    base = make_configclass("RobotObservations", [("policy", ordinary_type, ordinary_type())])()
    override = make_configclass("TaskObservations", [("policy", ordinary_type, ordinary_type(value=2))])()
    cameras = []
    for name in ("left_image", "right_image"):
        group = make_configclass(
            "CameraGroup",
            [(name, ObservationTermCfg, ObservationTermCfg(func=lambda env: None))],
            bases=(ObservationGroupCfg,),
        )()
        cameras.append(make_configclass("Cameras", [("camera_obs", type(group), group)])())
    for count in range(3):
        combined = combine_observation_cfgs(base, *cameras[:count], override)
        assert combined.policy.value == 2
        assert base.policy.value == 1
        definition = make_two_robot_definition()
        for index, robot in enumerate(definition.embodiments):
            robot_observations = combine_observation_cfgs(base, cameras[index] if index < count else None)
            robot.get_observation_cfg = lambda cfg=robot_observations: cfg
        assert (
            sum(
                getattr(robot.get_observation_cfg(), "camera_obs", None) is not None for robot in definition.embodiments
            )
            == count
        )
        with pytest.raises(AssertionError, match="duplicate observation groups"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(solve_relations=False)).compose_manager_cfg()
    cameras[1].camera_obs.enable_corruption = not cameras[0].camera_obs.enable_corruption
    with pytest.raises(AssertionError, match="Camera groups disagree"):
        combine_observation_cfgs(*cameras)
    return True


def test_observation_overrides_do_not_depend_on_camera_count():
    assert run_function_with_persistent_simulation_app(_test_observation_precedence)


def _test_scene_articulation_reset(simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
    from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange
    from isaaclab_arena.utils.velocity import Velocity

    position = torch.full((2, 3), 9.0)
    velocity = torch.full((2, 3), 8.0)
    root_pose = torch.full((2, 7), 7.0)
    root_velocity = torch.full((2, 6), 6.0)
    defaults = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=defaults),
            default_joint_vel=SimpleNamespace(torch=torch.zeros_like(defaults)),
        ),
        write_joint_position_to_sim_index=lambda position, env_ids: apply_positions(position, env_ids),
        write_joint_velocity_to_sim_index=lambda velocity, env_ids: apply_velocities(velocity, env_ids),
        write_root_pose_to_sim=lambda pose, env_ids: apply_root_pose(pose, env_ids),
        write_root_velocity_to_sim=lambda velocity, env_ids: apply_root_velocity(velocity, env_ids),
    )

    def apply_positions(values, env_ids):
        position[env_ids] = values

    def apply_velocities(values, env_ids):
        velocity[env_ids] = values

    def apply_root_pose(values, env_ids):
        root_pose[env_ids] = values

    def apply_root_velocity(values, env_ids):
        root_velocity[env_ids] = values

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])

    obj = Object(name="drawer", usd_path="/unused/drawer.usd", object_type=ObjectType.ARTICULATION)
    term = obj.get_event_cfg()[1]
    assert term.func is reset_articulation_pose_and_joints
    env = SimpleNamespace(scene=Scene(drawer=asset), device="cpu")
    env_ids = torch.tensor([1])
    term.func(env, env_ids, **term.params)
    assert torch.equal(position[0], torch.full((3,), 9.0))
    assert torch.equal(position[1], defaults[1])
    assert torch.equal(velocity[0], torch.full((3,), 8.0))
    assert torch.equal(velocity[1], torch.zeros(3))
    assert torch.equal(root_pose, torch.full((2, 7), 7.0))
    assert torch.equal(root_velocity, torch.full((2, 6), 6.0))
    cases = (
        (Pose(position_xyz=(1.0, 2.0, 3.0)), (11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 1.0)),
        (
            PosePerEnv([Pose(position_xyz=(4.0, 5.0, 6.0)), Pose(position_xyz=(7.0, 8.0, 9.0))]),
            (17.0, 28.0, 39.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (PoseRange(position_xyz_min=(2.0, 3.0, 4.0), position_xyz_max=(2.0, 3.0, 4.0)), None),
    )
    for pose, expected_root in cases:
        position.fill_(9.0)
        velocity.fill_(8.0)
        root_pose.fill_(7.0)
        root_velocity.fill_(6.0)
        obj.set_initial_pose(pose)
        term = obj.get_event_cfg()[1]
        with patch("isaaclab_arena.terms.events.randomize_object_pose") as randomize:
            term.func(env, env_ids, **term.params)
            if isinstance(pose, PoseRange):
                randomize.assert_called_once_with(
                    env, env_ids, pose_range=pose.to_dict(), asset_cfgs=[term.params["asset_cfg"]]
                )
            else:
                randomize.assert_not_called()
                assert torch.equal(root_pose[1], torch.tensor(expected_root))
                assert torch.equal(root_velocity[1], torch.zeros(6))
        assert torch.equal(position[0], torch.full((3,), 9.0))
        assert torch.equal(position[1], defaults[1])
        assert torch.equal(velocity[0], torch.full((3,), 8.0))
        assert torch.equal(velocity[1], torch.zeros(3))
        assert torch.equal(root_pose[0], torch.full((7,), 7.0))
        assert torch.equal(root_velocity[0], torch.full((6,), 6.0))
    configured_velocity = Velocity(linear_xyz=(1.0, 2.0, 3.0), angular_xyz=(4.0, 5.0, 6.0))
    for pose, expected_root in ((None, None), *cases):
        obj = Object(name="drawer", usd_path="/unused/drawer.usd", object_type=ObjectType.ARTICULATION)
        if pose is not None:
            obj.set_initial_pose(pose)
        obj.set_initial_velocity(configured_velocity)
        term = obj.get_event_cfg()[1]
        position.fill_(9.0)
        velocity.fill_(8.0)
        root_pose.fill_(7.0)
        root_velocity.fill_(6.0)
        with patch(
            "isaaclab_arena.terms.events.randomize_object_pose",
            side_effect=lambda env, env_ids, **kwargs: apply_root_velocity(torch.zeros(len(env_ids), 6), env_ids),
        ):
            term.func(env, env_ids, **term.params)
        torch.testing.assert_close(root_velocity[1], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert torch.equal(root_velocity[0], torch.full((6,), 6.0))
        assert torch.equal(root_pose[0], torch.full((7,), 7.0))
        if expected_root is not None:
            assert torch.equal(root_pose[1], torch.tensor(expected_root))
        elif pose is None:
            assert torch.equal(root_pose[1], torch.full((7,), 7.0))
            assert not obj.has_pose_reset_event()
        assert torch.equal(position[0], torch.full((3,), 9.0))
        assert torch.equal(position[1], defaults[1])
        assert torch.equal(velocity[0], torch.full((3,), 8.0))
        assert torch.equal(velocity[1], torch.zeros(3))
    return True


def test_scene_articulation_owns_selected_joint_reset():
    assert run_function_with_persistent_simulation_app(_test_scene_articulation_reset)
