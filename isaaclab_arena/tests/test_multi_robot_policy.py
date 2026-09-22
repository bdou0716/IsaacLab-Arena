# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check shared policy batching in real environments and at the tensor boundary."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_shared_policy(simulation_app, robot_count, separate_third):
    import torch

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg
    from isaaclab_arena.tests.test_multi_embodiment_environment import make_two_robot_definition
    from isaaclab_arena.utils.pose import Pose

    definition = make_two_robot_definition()
    if robot_count == 3:
        definition.embodiments.append(
            FrankaJointPosEmbodiment(
                instance_key="middle",
                initial_pose=Pose.identity(),
            )
        )
    env = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    policies = {"shared": {"type": "zero_action"}}
    if separate_third:
        policies["solo"] = {"type": "zero_action"}
    assignments = {
        robot.get_scene_key(): "solo" if separate_third and index == 2 else "shared"
        for index, robot in enumerate(definition.embodiments)
    }
    policy = MultiRobotPolicy(MultiRobotPolicyCfg(policies=policies, assignments=assignments))
    calls = {name: [] for name in policies}
    for name, inner in policy.policies.items():
        original = inner.get_action
        rows = 2 * sum(assignment == name for assignment in assignments.values())

        def checked(view, observation, name=name, rows=rows, original=original):
            """Check that each policy sees its robot interface and assigned batch."""
            assert view.num_envs == rows
            assert set(observation) == {"policy"}
            assert observation["policy"]["joint_pos"].shape[0] == rows
            assert observation["policy"]["actions"].shape == (rows, 8)
            assert view.action_manager.active_terms == ["arm_action", "gripper_action"]
            calls[name].append(rows)
            return original(view, observation)

        inner.get_action = checked
    try:
        observation, _ = env.reset()
        for _ in range(2):
            actions = policy.get_action(env, observation)
            assert actions.shape == (2, 8 * robot_count)
            assert torch.count_nonzero(actions) == 0
            observation, _, _, _, _ = env.step(actions)
        assert calls == {
            name: [2 * sum(assignment == name for assignment in assignments.values())] * 2 for name in policies
        }
    finally:
        policy.close()
        env.close()
    return True


@pytest.mark.parametrize("robot_count,separate_third", [(2, False), (3, False), (3, True)])
def test_robots_share_policy_instances(robot_count, separate_third):
    assert run_function_with_persistent_simulation_app(
        _test_shared_policy, robot_count=robot_count, separate_third=separate_third
    )


def _test_flat_experiment(simulation_app, output_dir):
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg

    path = output_dir / "experiment.yaml"
    path.write_text("""
runs:
  shared_robots:
    environment:
      type: cube_goal_pose
    policy:
      type: multi_robot
      policies:
        arms:
          type: zero_action
          params: {}
      assignments:
        left: arms
        right: arms
    environment_builder:
      num_envs: 2
    rollout_limit:
      num_episodes: 2
""")
    experiment = load_arena_experiment_from_config_file(path, device="cuda:0")
    cfg = experiment.runs["shared_robots"].policy
    assert isinstance(cfg, MultiRobotPolicyCfg)
    assert cfg.assignments == {"left": "arms", "right": "arms"}
    policy = MultiRobotPolicy(cfg)
    assert len(policy.policies) == 1
    policy.close()
    return True


def test_flat_experiment_selects_registered_inner_policies(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_flat_experiment, output_dir=tmp_path)


def _test_policy_values(simulation_app):
    """Check exact row and column ownership at the policy's environment interface."""
    import gymnasium as gym
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg, _stack_observations
    from isaaclab_arena.utils.instance_rename import robot_last_action

    action = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    definitions = [
        ("left_arm", "left", 2),
        ("right_arm", "right", 2),
        ("drive", "robot", 1),
        ("left_gripper", "left", 1),
        ("right_gripper", "right", 1),
    ]
    terms = {}
    column = 0
    for name, owner, width in definitions:
        raw = action[:, column : column + width]
        terms[name] = SimpleNamespace(
            cfg=SimpleNamespace(asset_name=owner), raw_actions=raw, processed_actions=raw * 10
        )
        column += width
    manager = SimpleNamespace(
        active_terms=list(terms),
        action_term_dim=[width for _, _, width in definitions],
        get_term=terms.__getitem__,
        action=action,
        prev_action=action + 30,
    )
    single_space = gym.spaces.Box(-1000.0, 1000.0, (7,))
    env = SimpleNamespace(
        action_manager=manager,
        num_envs=2,
        device="cpu",
        single_action_space=single_space,
        action_space=gym.vector.utils.batch_space(single_space, 2),
    )
    env.unwrapped = env
    from isaaclab_arena.utils.observation_bindings import ObservationBinding

    env.cfg = SimpleNamespace(
        observation_bindings=[
            ObservationBinding("left_policy", None, "left", "policy"),
            ObservationBinding("right_policy", None, "right", "policy"),
            ObservationBinding("policy", None, "robot", "policy"),
            ObservationBinding("camera_obs", "left_wrist", "left", "camera_obs", "wrist"),
            ObservationBinding("camera_obs", "right_wrist", "right", "camera_obs", "wrist"),
            ObservationBinding("camera_obs", "wrist", "robot", "camera_obs", "wrist"),
        ]
    )
    left = robot_last_action(env, ("left_arm", "left_gripper"))
    right = robot_last_action(env, ("right_arm", "right_gripper"))
    humanoid = robot_last_action(env, ("drive",))
    torch.testing.assert_close(left, torch.tensor([[0.0, 1.0, 5.0], [7.0, 8.0, 12.0]]))
    torch.testing.assert_close(right, torch.tensor([[2.0, 3.0, 6.0], [9.0, 10.0, 13.0]]))
    torch.testing.assert_close(humanoid, torch.tensor([[4.0], [11.0]]))
    observations = {
        "left_policy": {"actions": left},
        "right_policy": {"actions": right},
        "policy": {"actions": humanoid},
        "camera_obs": {"left_wrist": left[:, :1], "right_wrist": right[:, :1], "wrist": humanoid},
    }

    class RecordingPolicy:
        """Record policy inputs and lifecycle calls for value-level assertions."""

        def __init__(self, expected, offset, remote=False):
            self.expected = expected
            self.offset = offset
            self.is_remote = remote
            self.calls = 0
            self.resets = []
            self.closed = 0
            self.recorded = False
            self.recorded_length = None

        def get_action(self, view, observation):
            """Return values that reveal the owning robot and environment row."""
            self.calls += 1
            assert set(observation) == {"policy", "camera_obs"}
            assert set(observation["camera_obs"]) == {"wrist"}
            assert view.num_envs == self.expected.shape[0]
            assert view.single_action_space.shape == self.expected.shape[1:]
            torch.testing.assert_close(observation["policy"]["actions"], self.expected)
            torch.testing.assert_close(observation["camera_obs"]["wrist"], self.expected[:, :1])
            return self.expected + self.offset

        def reset(self, indices=None):
            """Record the expanded rows reset by the composite policy."""
            self.resets.append(None if indices is None else indices.clone())

        def set_task_description(self, description):
            """Record the task description forwarded to the shared policy."""
            self.description = description

        def close(self):
            """Count shutdown calls to the shared policy."""
            self.closed += 1

        def has_length(self):
            """Declare a finite policy horizon for agreement checks."""
            return self.recorded

        def length(self):
            """Return the inner policy horizon."""
            return self.recorded_length

    policy = MultiRobotPolicy(
        MultiRobotPolicyCfg(
            policies={"arms": {"type": "zero_action"}, "humanoid": {"type": "zero_action"}},
            assignments={"left": "arms", "right": "arms", "robot": "humanoid"},
        )
    )
    arms = RecordingPolicy(torch.cat([left, right]), 100)
    body = RecordingPolicy(humanoid, 500, remote=True)
    policy.policies = {"arms": arms, "humanoid": body}
    single_space.low[2] = -999.0
    for _ in range(2):
        with pytest.raises(AssertionError, match="equal lower bounds"):
            policy.get_action(env, observations)
    assert arms.calls == body.calls == 0
    single_space.low[2] = -1000.0
    expected = action + torch.tensor([100, 100, 100, 100, 500, 100, 100])
    torch.testing.assert_close(policy.get_action(env, observations), expected)
    torch.testing.assert_close(policy.get_action(env, observations), expected)
    assert arms.calls == body.calls == 2
    policy.reset(torch.tensor([1]))
    assert arms.resets[-1].tolist() == [1, 3]
    assert body.resets[-1].tolist() == [1]
    policy.reset(torch.tensor([], dtype=torch.long))
    assert arms.resets[-1].numel() == body.resets[-1].numel() == 0
    with pytest.raises(AssertionError, match="out of range"):
        policy.reset(torch.tensor([2]))
    policy.reset()
    assert arms.resets[-1] is body.resets[-1] is None
    policy.set_task_description("Move each robot to its goal")
    assert arms.description == body.description == "Move each robot to its goal"
    assert policy.is_remote
    assert not policy.has_length()
    assert policy.length() is None
    arms.recorded = True
    with pytest.raises(AssertionError, match="agree"):
        policy.has_length()
    body.recorded = True
    assert policy.has_length()
    arms.recorded_length = 10
    body.recorded_length = 20
    with pytest.raises(AssertionError, match="agree"):
        policy.length()
    body.recorded_length = 10
    assert policy.length() == 10
    policy.close()
    assert arms.closed == body.closed == 1
    with pytest.raises(AssertionError, match="matching observation groups"):
        _stack_observations([{"a": left}, {"b": right}], 2)
    with pytest.raises(AssertionError, match="matching observation tensor shapes"):
        _stack_observations([left, right[:, :1]], 2)
    with pytest.raises(AssertionError, match="rows"):
        _stack_observations([left[:1], right[:1]], 2)
    return True


def test_policy_action_values_and_lifecycle():
    assert run_function_with_persistent_simulation_app(_test_policy_values)


def _test_cleanup_failures(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy

    with patch.object(ZeroActionPolicy, "close") as close:
        with pytest.raises(AssertionError):
            MultiRobotPolicy(
                MultiRobotPolicyCfg(
                    policies={"first": {"type": "zero_action"}, "invalid": {"type": "not_a_registered_policy"}},
                    assignments={"left": "first", "right": "invalid"},
                )
            )
        close.assert_called_once()
    calls = []

    def fail():
        """Record a cleanup attempt before simulating a socket failure."""
        calls.append("first")
        raise RuntimeError("socket cleanup failed")

    policy = MultiRobotPolicy.__new__(MultiRobotPolicy)
    policy.policies = {
        "first": SimpleNamespace(close=fail),
        "second": SimpleNamespace(close=lambda: calls.append("second")),
    }
    with pytest.raises(RuntimeError, match="socket cleanup failed"):
        policy.close()
    assert calls == ["first", "second"]
    return True


def test_policy_failures_release_all_constructed_children():
    assert run_function_with_persistent_simulation_app(_test_cleanup_failures)


def _test_observation_ownership(simulation_app):
    import torch

    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.policy.multi_robot_policy import _robot_observation, _stack_observations
    from isaaclab_arena.tests.test_multi_embodiment_environment import make_two_robot_definition
    from isaaclab_arena.utils.configclass import make_configclass
    from isaaclab_arena.utils.observation_bindings import ObservationBinding
    from isaaclab_arena.utils.pose import Pose

    definition = make_two_robot_definition()
    definition.embodiments = [
        FrankaJointPosEmbodiment(instance_key=key, enable_cameras=True, initial_pose=Pose.identity())
        for key in ("left", "left_wrist")
    ]
    group = make_configclass(
        "SharedGoal",
        [("goal", ObservationTermCfg, ObservationTermCfg(func=lambda env: None))],
        bases=(ObservationGroupCfg,),
    )()
    task_cfg = make_configclass(
        "TaskObservations",
        [
            ("shared_goal", type(group), group),
            ("task_owned", type(group), group),
        ],
    )()
    definition.task.get_observation_cfg = lambda: task_cfg
    definition.task.get_observation_bindings = lambda: [ObservationBinding("task_owned", None, "left", "task_goal")]
    cfg, _ = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(solve_relations=False)).compose_manager_cfg()
    bindings = cfg.observation_bindings
    observation = {}
    for index, binding in enumerate(bindings):
        value = torch.full((2, 1), float(index))
        if binding.source_term is None:
            observation[binding.source_group] = value
        else:
            observation.setdefault(binding.source_group, {})[binding.source_term] = value
    left = _robot_observation(observation, "left", bindings)
    right = _robot_observation(observation, "left_wrist", bindings)
    assert left["shared_goal"] is right["shared_goal"]
    assert "task_goal" in left and "task_goal" not in right
    assert set(left["camera_obs"]) == set(right["camera_obs"])
    for camera in left["camera_obs"]:
        assert not torch.equal(left["camera_obs"][camera], right["camera_obs"][camera])
    with pytest.raises(AssertionError, match="matching observation groups"):
        _stack_observations([left, right], 2)
    with pytest.raises(AssertionError, match="match runtime observations"):
        _robot_observation({**observation, "unbound": torch.ones(2, 1)}, "left", bindings)
    source_actions = torch.ones(2, 1)
    source_group = {"actions": source_actions}
    collision_observation = {"owned": source_group, "camera_obs": {"image": torch.zeros(2, 1)}}
    collision_bindings = [
        ObservationBinding("owned", None, "left", "policy"),
        ObservationBinding("camera_obs", "image", "left", "policy", "image"),
    ]
    for ordered_bindings in (collision_bindings, collision_bindings[::-1]):
        with pytest.raises(AssertionError, match="Duplicate local observation group"):
            _robot_observation(collision_observation, "left", ordered_bindings)
        assert set(source_group) == {"actions"}
        assert collision_observation["owned"] is source_group
        assert source_group["actions"] is source_actions
    return True


@pytest.mark.with_cameras
def test_explicit_observation_owners_survive_overlapping_robot_names():
    assert run_function_with_persistent_simulation_app(_test_observation_ownership, enable_cameras=True)


def _test_child_configuration(simulation_app):
    from dataclasses import dataclass
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import PolicyRegistry
    from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg
    from isaaclab_arena.policy.policy_base import PolicyCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy

    @dataclass
    class TypedChildCfg(PolicyCfg):
        count: int = 1

    def build(selector, params):
        return MultiRobotPolicy(
            MultiRobotPolicyCfg(
                policies={"child": {"type": selector, "params": params}},
                assignments={"left": "child"},
            )
        )

    registry = PolicyRegistry()
    assert registry.resolve_policy_type("zero_action") is ZeroActionPolicy
    with patch("importlib.import_module", side_effect=AssertionError("Unexpected selector import")):
        assert (
            registry.resolve_policy_type("isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy")
            is ZeroActionPolicy
        )
        with pytest.raises(AssertionError, match="one registered policy"):
            registry.resolve_policy_type("unregistered_policy_package.CustomPolicy")
    with patch.object(PolicyRegistry, "get_policy_cfg_type", return_value=TypedChildCfg):
        policy = build("isaaclab_arena.policy.zero_action_policy.ZeroActionPolicy", {"count": "3"})
        assert type(policy.policies["child"]) is ZeroActionPolicy
        assert policy.policies["child"].config.count == 3
        policy.close()
        for params in ({"count": "invalid"}, {"unknown": 1}, {"defaults": [], "unknown": 1}):
            with pytest.raises(ValueError, match="child policy 'child'"):
                build("zero_action", params)
    for selector in (
        "multi_robot",
        "isaaclab_arena.policy.multi_robot_policy.MultiRobotPolicy",
    ):
        with pytest.raises(AssertionError, match="Nested"):
            build(selector, {})
    for selector in ("rsl_rl", "isaaclab_arena.policy.rsl_rl_action_policy.RslRlActionPolicy"):
        with pytest.raises(AssertionError, match="full environment wrapper"):
            build(selector, {"checkpoint_path": "unused.pt"})
    return True


def test_child_parameters_use_typed_schema_and_class_resolution():
    assert run_function_with_persistent_simulation_app(_test_child_configuration)
