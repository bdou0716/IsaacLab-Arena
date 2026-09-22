# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Batch robots sharing a registered policy and assemble their action columns."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.hydra.typed_config import compose_typed_config
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.utils.observation_bindings import ObservationBinding


@dataclass
class MultiRobotPolicyCfg(PolicyCfg):
    """Configure named policies and assign robot instance keys to them."""

    policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Policy instance names mapped to registered type names and primitive parameters."""

    assignments: dict[str, str] = field(default_factory=dict)
    """Robot scene keys mapped to policy instance names."""


@dataclass(frozen=True)
class _PolicyBinding:
    """Hold a fully validated mapping between one environment and its child policies."""

    env: Any
    num_envs: int
    layouts: dict[str, list[tuple[str, slice]]]
    views: dict[str, tuple[list[str], _RobotEnvView]]
    observations: tuple[ObservationBinding, ...]


@register_policy
class MultiRobotPolicy(PolicyBase[MultiRobotPolicyCfg]):
    """Call each inner policy once using robot-major rows."""

    name = "multi_robot"

    def __init__(self, config: MultiRobotPolicyCfg):
        from isaaclab_arena.policy.rsl_rl_action_policy import RslRlActionPolicy

        super().__init__(config)
        assert config.assignments, "Assign at least one robot to an inner policy"
        assert set(config.assignments.values()) == set(config.policies), "Every policy must have assigned robots"
        registry = PolicyRegistry()
        self.policies = {}
        with ExitStack() as cleanup:
            for index, (name, definition) in enumerate(config.policies.items()):
                assert (
                    set(definition) <= {"type", "params"} and "type" in definition
                ), "Inner policy needs type and params"
                params = definition.get("params", {})
                assert isinstance(params, dict) and _primitive(
                    params
                ), "Inner policy parameters must be primitive values"
                policy_type = registry.resolve_policy_type(definition["type"])
                assert not issubclass(policy_type, MultiRobotPolicy), "Nested multi-robot policies are not supported"
                assert not issubclass(
                    policy_type, RslRlActionPolicy
                ), "RSL-RL policies require a full environment wrapper and cannot use a robot view"
                try:
                    child_cfg = compose_typed_config(
                        registry.get_policy_cfg_type(policy_type), params, f"arena_multi_robot_child_{index}"
                    )
                except (HydraException, OmegaConfBaseException, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid parameters for child policy '{name}': {exc}") from exc
                self.policies[name] = policy_type(child_cfg)
                cleanup.callback(self.policies[name].close)
            cleanup.pop_all()
        self._binding: _PolicyBinding | None = None

    def _bind(self, env) -> _PolicyBinding:
        """Resolve action columns and construct each inner policy's environment view."""
        manager = env.unwrapped.action_manager
        layouts = {key: [] for key in self.config.assignments}
        column = 0
        for name, width in zip(manager.active_terms, manager.action_term_dim):
            term = manager.get_term(name)
            owner = term.cfg.asset_name
            assert owner in layouts, f"Action term '{name}' has no policy assignment for '{owner}'"
            layouts[owner].append((name, slice(column, column + width)))
            column += width
        assert column == env.action_space.shape[-1], "Action manager widths must cover the action space"
        assert all(layouts.values()), "Every assigned robot must own at least one action term"
        observations = tuple(env.unwrapped.cfg.observation_bindings)
        assert observations, "The environment must declare observation bindings for composite policies"
        assert len({binding.source for binding in observations}) == len(
            observations
        ), "Observation bindings must be unique"
        for binding in observations:
            assert (
                binding.robot_key is None or binding.robot_key in layouts
            ), f"Observation '{binding.source}' has no policy assignment for '{binding.robot_key}'"
        views = {}
        for name in self.policies:
            keys = [key for key, assigned in self.config.assignments.items() if assigned == name]
            manager_view = _RobotActionManager(keys, layouts)
            views[name] = (keys, _RobotEnvView(env.unwrapped, manager_view, len(keys)))
        return _PolicyBinding(env, env.unwrapped.num_envs, layouts, views, observations)

    def get_action(self, env, observation):
        """Batch each policy's robot observations and scatter its returned actions."""
        if self._binding is None:
            self._binding = self._bind(env)
        binding = self._binding
        assert env is binding.env, "Construct a new composite policy when the environment is rebuilt"
        output = torch.empty(env.action_space.shape, device=env.unwrapped.device)
        for name, policy in self.policies.items():
            keys, view = binding.views[name]
            robot_observations = [_robot_observation(observation, key, binding.observations) for key in keys]
            observations = _stack_observations(robot_observations, binding.num_envs)
            actions = policy.get_action(view, observations)
            expected = (view.num_envs, view.action_manager.total_action_dim)
            assert (
                isinstance(actions, torch.Tensor) and tuple(actions.shape) == expected
            ), f"Policy '{name}' must return actions shaped {expected}"
            for key, rows in zip(keys, actions.split(binding.num_envs)):
                term_actions = rows.split(view.action_manager.action_term_dim, dim=-1)
                for (_, columns), values in zip(binding.layouts[key], term_actions):
                    output[:, columns] = values
        return output

    def reset(self, env_ids=None):
        """Expand environment indices to the corresponding rows of every shared policy."""
        if env_ids is None:
            for policy in self.policies.values():
                policy.reset(None)
            return
        binding = self._binding
        assert binding is not None, "Indexed reset requires the first action call to establish row counts"
        assert env_ids.ndim == 1, "Reset indices must be a one-dimensional tensor"
        assert bool(((env_ids >= 0) & (env_ids < binding.num_envs)).all()), "Reset indices are out of range"
        for name, policy in self.policies.items():
            count = len(binding.views[name][0])
            policy.reset(torch.cat([env_ids + index * binding.num_envs for index in range(count)]))

    def set_task_description(self, task_description):
        """Send the mission description to every inner policy."""
        super().set_task_description(task_description)
        for policy in self.policies.values():
            policy.set_task_description(task_description)
        return task_description

    def close(self):
        """Close each inner policy once."""
        with ExitStack() as cleanup:
            for policy in reversed(self.policies.values()):
                cleanup.callback(policy.close)

    @property
    def is_remote(self):
        """Report whether any inner policy uses a remote service."""
        return any(policy.is_remote for policy in self.policies.values())

    def has_length(self):
        """Require inner policies to agree whether they replay recorded actions."""
        lengths = {policy.has_length() for policy in self.policies.values()}
        assert len(lengths) == 1, "Inner policies must agree whether they have a recorded length"
        return lengths.pop()

    def length(self):
        """Return the common recording length, rejecting disagreement."""
        lengths = {policy.length() for policy in self.policies.values()}
        assert len(lengths) == 1, "Inner policies must agree on recorded length"
        return lengths.pop()


class _RobotEnvView:
    """Expose only the robot batch and action interface used by inner policies."""

    def __init__(self, env, action_manager, robot_count):
        self.num_envs = env.num_envs * robot_count
        self.device = env.device
        self.action_manager = action_manager
        lows = [
            np.concatenate([env.single_action_space.low[columns] for _, columns in layout])
            for layout in action_manager._layouts
        ]
        highs = [
            np.concatenate([env.single_action_space.high[columns] for _, columns in layout])
            for layout in action_manager._layouts
        ]
        assert all(
            np.array_equal(value, lows[0]) for value in lows
        ), "Shared action spaces must have equal lower bounds"
        assert all(
            np.array_equal(value, highs[0]) for value in highs
        ), "Shared action spaces must have equal upper bounds"
        self.single_action_space = gym.spaces.Box(lows[0], highs[0])
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.unwrapped = self


class _RobotActionManager:
    """Present corresponding action terms stacked over robots, then environments."""

    def __init__(self, keys, layouts):
        self._layouts = [layouts[key] for key in keys]
        self.active_terms = [_strip_prefix(name, keys[0]) for name, _ in self._layouts[0]]
        self.action_term_dim = [columns.stop - columns.start for _, columns in self._layouts[0]]
        for key, layout in zip(keys, self._layouts):
            assert [
                _strip_prefix(name, key) for name, _ in layout
            ] == self.active_terms, "Shared policies need matching action terms"
            assert [
                columns.stop - columns.start for _, columns in layout
            ] == self.action_term_dim, "Shared policies need matching action widths"
        self.total_action_dim = sum(self.action_term_dim)


def _strip_prefix(name, key):
    """Restore the term name expected by a single-robot policy."""
    return name.removeprefix(f"{key}_") if key != "robot" else name


def _robot_observation(observation, key, bindings):
    """Select owned and shared observations using declared source and local names."""
    sources = {
        (name, term)
        for name, value in observation.items()
        for term in (value if name == "camera_obs" and isinstance(value, dict) else [None])
    }
    assert sources == {
        binding.source for binding in bindings
    }, "Observation bindings must match runtime observations; update bindings when changing observation groups"
    result = {}
    whole_groups = set()
    for binding in bindings:
        if binding.robot_key not in (None, key):
            continue
        value = observation[binding.source_group]
        if binding.source_term is None:
            assert binding.local_group not in result, f"Duplicate local observation group '{binding.local_group}'"
            result[binding.local_group] = value
            whole_groups.add(binding.local_group)
        else:
            assert binding.local_group not in whole_groups, f"Duplicate local observation group '{binding.local_group}'"
            group = result.setdefault(binding.local_group, {})
            assert (
                isinstance(group, dict) and binding.local_term not in group
            ), f"Duplicate local observation term '{binding.local_group}/{binding.local_term}'"
            group[binding.local_term] = value[binding.source_term]
    assert result, f"No observation groups found for robot '{key}'"
    return result


def _stack_observations(values, num_envs):
    """Stack matching nested observation dictionaries in robot-major row order."""
    first = values[0]
    if isinstance(first, dict):
        assert all(
            isinstance(value, dict) and set(value) == set(first) for value in values
        ), "Shared policies need matching observation groups and terms"
        return {name: _stack_observations([value[name] for value in values], num_envs) for name in first}
    assert all(
        isinstance(value, torch.Tensor) and value.shape[1:] == first.shape[1:] for value in values
    ), "Shared policies need matching observation tensor shapes"
    assert all(value.shape[0] == num_envs for value in values), "Observation rows must match the environment count"
    return torch.cat(values, dim=0)


def _primitive(value):
    """Check whether policy parameters contain only serializable scalar containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_primitive(child) for child in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _primitive(child) for key, child in value.items())
    return False
