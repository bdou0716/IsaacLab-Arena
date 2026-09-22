# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gr00tRemoteClosedloopPolicy.

Exercises the real obs/action translation pipeline and mocks only the native PolicyClient -- the
wire boundary to a remote GR00T server -- so no server is required.

Verifies:
- the obs dict sent to the server has the expected language/video/state structure;
- the action dict returned by the server is converted to a tensor of the
  expected (num_envs, action_chunk_length, action_dim) shape;
- ``reset`` propagates to the client;
- a failing ping during construction raises ``ConnectionError``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import sys
import torch
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from isaaclab_arena.assets.registries import PolicyRegistry
from isaaclab_arena.policy import action_scheduling
from isaaclab_arena.policy.multi_robot_policy import MultiRobotPolicy, MultiRobotPolicyCfg
from isaaclab_arena_gr00t.policy import gr00t_remote_closedloop_policy as gr00t_policy
from isaaclab_arena_gr00t.tests.utils.constants import TestConstants as Gr00tTestConstants

pytestmark = pytest.mark.gr00t_policy

NUM_ENVS = 2
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
NUM_CHANNELS = 3
NUM_SIM_JOINTS = 43  # G1 43-DoF
ACTION_HORIZON = 50
ACTION_CHUNK_LENGTH = 50
EXPECTED_ACTION_DIM = 50  # 43 joints + 3 nav + 1 base height + 3 torso rpy

# Joint group sizes from gr00t_43dof_joint_space.yaml (must match the YAML).
POLICY_GROUP_SIZES = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "left_hand": 7,
    "right_arm": 7,
    "right_hand": 7,
}


# ----------------------------- fixtures ------------------------------ #


@pytest.fixture
def policy_config_yaml():
    """Return the g1 locomanip test config used for Arena-side obs/action translation."""
    return (
        Gr00tTestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_gr00t_closedloop_config.yaml"
    )


@pytest.fixture
def synthetic_observation():
    """Env-shaped torch obs: one ego camera + 43 joint positions per env."""
    rgb = torch.randint(0, 255, (NUM_ENVS, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS), dtype=torch.uint8)
    joint_pos = torch.randn(NUM_ENVS, NUM_SIM_JOINTS, dtype=torch.float32)
    return {
        "camera_obs": {"robot_head_cam_rgb": rgb},
        "policy": {"robot_joint_pos": joint_pos},
    }


def _make_action_response(num_envs: int, horizon: int) -> dict[str, np.ndarray]:
    """Synthetic GR00T-server response in the format ``build_gr00t_action_np`` expects."""
    response: dict[str, np.ndarray] = {}
    for group, size in POLICY_GROUP_SIZES.items():
        response[group] = np.random.randn(num_envs, horizon, size).astype(np.float32)
    response["navigate_command"] = np.random.randn(num_envs, horizon, 3).astype(np.float32)
    response["base_height_command"] = np.random.randn(num_envs, horizon, 1).astype(np.float32)
    return response


class _FakePolicyClient:
    """Minimal stand-in for ``gr00t.policy.server_client.PolicyClient``."""

    def __init__(self, *args, ping_ok: bool = True, **kwargs):
        self.init_kwargs = kwargs
        self._ping_ok = ping_ok
        self.last_observation: dict[str, Any] | None = None
        self.get_action_calls = 0
        self.reset_called = False

    def ping(self) -> bool:
        return self._ping_ok

    def get_action(self, observation: dict[str, Any]):
        self.last_observation = observation
        self.get_action_calls += 1
        # Match the real PolicyClient return signature: (action_dict, latency_or_meta).
        num_envs = len(observation["language"]["annotation.human.task_description"])
        return _make_action_response(num_envs, ACTION_HORIZON), None

    def reset(self):
        self.reset_called = True


@pytest.fixture
def fake_client_factory(monkeypatch):
    """Install a fake native PolicyClient and expose each constructed client."""
    created: list[_FakePolicyClient] = []

    def factory(ping_ok: bool = True):
        def _ctor(*args, **kwargs):
            client = _FakePolicyClient(*args, ping_ok=ping_ok, **kwargs)
            created.append(client)
            return client

        server_client_module = ModuleType("gr00t.policy.server_client")
        server_client_module.PolicyClient = _ctor
        monkeypatch.setitem(sys.modules, "gr00t.policy.server_client", server_client_module)
        return created

    return factory


def _build_policy(policy_config_yaml: str, scheduler: str = "chunk"):
    cfg = gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=policy_config_yaml,
        policy_device="cpu",  # keep the test portable; remote policy does no GPU compute
        num_envs=NUM_ENVS,
        remote_host="unused",
        remote_port=0,
        remote_api_token=None,
        scheduler=scheduler,
    )
    return gr00t_policy.Gr00tRemoteClosedloopPolicy(cfg)


def _environment(num_envs=NUM_ENVS):
    """Supply the runtime batch size consumed by the policy interface."""
    env = SimpleNamespace(num_envs=num_envs)
    env.unwrapped = env
    return env


# ------------------------------- tests ------------------------------- #


def test_gr00t_client_imports():
    """Import the real lightweight GR00T client API."""
    from gr00t.policy.server_client import MsgSerializer, PolicyClient

    assert MsgSerializer is not None
    assert PolicyClient is not None


def test_observation_sent_to_server_has_expected_structure(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """The dict passed to ``PolicyClient.get_action`` matches GR00T's expected
    language/video/state layout for the g1 sim WBC modality config."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    policy.get_action(_environment(), synthetic_observation)

    assert len(clients) == 1
    sent = clients[0].last_observation
    assert sent is not None and clients[0].get_action_calls == 1

    # language: one [task] entry per env
    assert "language" in sent
    lang = sent["language"]["annotation.human.task_description"]
    assert lang == [["pick up the brown box"]] * NUM_ENVS

    # video: one ego view, shaped (N, 1, H, W, C) and uint8
    assert "video" in sent and "ego_view" in sent["video"]
    ego = sent["video"]["ego_view"]
    assert ego.shape == (NUM_ENVS, 1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, NUM_CHANNELS)
    assert ego.dtype == np.uint8

    # state: one (N, 1, dof_in_group) entry per modality state key
    expected_state_keys = {"left_arm", "right_arm", "left_hand", "right_hand", "waist"}
    assert set(sent["state"].keys()) == expected_state_keys
    for key in expected_state_keys:
        arr = sent["state"][key]
        assert arr.ndim == 3
        assert arr.shape[0] == NUM_ENVS
        assert arr.shape[1] == 1
        assert arr.shape[2] == POLICY_GROUP_SIZES[key]


def test_action_response_is_translated_to_correct_tensor_shape(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """Server's grouped joint dict is reassembled into a (N, chunk, action_dim) tensor."""
    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    policy.set_task_description("pick up the brown box")

    policy.get_action(_environment(), synthetic_observation)
    action = policy._get_action_chunk(synthetic_observation, ["robot_head_cam_rgb"])

    assert isinstance(action, torch.Tensor)
    # _get_action_chunk asserts action_tensor.shape[1] >= action_chunk_length, but
    # the full horizon is returned; pin both dims explicitly.
    assert action.shape == (NUM_ENVS, ACTION_HORIZON, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_reset_propagates_to_client(policy_config_yaml, fake_client_factory):
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml)
    assert clients[0].reset_called is False
    policy.reset()
    assert clients[0].reset_called is True


def test_construction_fails_when_server_unreachable(policy_config_yaml, fake_client_factory):
    fake_client_factory(ping_ok=False)
    with pytest.raises(ConnectionError):
        _build_policy(policy_config_yaml)


def test_registration_builds_typed_config_and_scheduler(policy_config_yaml, synthetic_observation, fake_client_factory):
    """The registered typed config retains scheduler selection."""
    fake_client_factory(ping_ok=True)
    assert (
        PolicyRegistry().get_policy_cfg_type(gr00t_policy.Gr00tRemoteClosedloopPolicy)
        is gr00t_policy.Gr00tRemoteClosedloopPolicyCfg
    )
    policy = gr00t_policy.Gr00tRemoteClosedloopPolicy(
        gr00t_policy.Gr00tRemoteClosedloopPolicyCfg(
            policy_config_yaml_path=policy_config_yaml,
            policy_device="cpu",
            num_envs=NUM_ENVS,
            remote_host="unused",
            remote_port=0,
            scheduler="synced_batch",
        )
    )

    assert isinstance(policy.config, gr00t_policy.Gr00tRemoteClosedloopPolicyCfg)
    assert policy._chunking_state is None
    policy.set_task_description("pick up the brown box")
    policy.get_action(_environment(), synthetic_observation)
    assert isinstance(policy._chunking_state, action_scheduling.SyncedBatchActionScheduler)


# ---------------- scheduler-switch tests ---------------- #


@pytest.mark.parametrize("scheduler_cls_name", ["ActionChunkScheduler", "SyncedBatchActionScheduler"])
def test_get_action_returns_correct_shape_for_each_scheduler(
    policy_config_yaml, synthetic_observation, fake_client_factory, scheduler_cls_name
):
    """Both ActionChunkScheduler and SyncedBatchActionScheduler should drive the policy
    end-to-end and produce one (num_envs, action_dim) action per get_action call."""
    scheduler_cls = getattr(action_scheduling, scheduler_cls_name)
    scheduler = "synced_batch" if scheduler_cls_name == "SyncedBatchActionScheduler" else "chunk"

    fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler=scheduler)
    policy.set_task_description("pick up the brown box")

    action = policy.get_action(env=_environment(), observation=synthetic_observation)

    assert isinstance(policy._chunking_state, scheduler_cls)
    assert isinstance(action, torch.Tensor)
    assert action.shape == (NUM_ENVS, EXPECTED_ACTION_DIM)
    assert action.device.type == "cpu"


def test_synced_batch_holds_joint_position_for_env_after_partial_reset(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """After resetting a single env, SyncedBatchActionScheduler should hold that env on
    the joint-position-derived hold_action while the others continue stepping their chunk."""
    clients = fake_client_factory(ping_ok=True)
    policy = _build_policy(policy_config_yaml, scheduler="synced_batch")
    policy.set_task_description("pick up the brown box")

    # Step 1: every env needs a chunk → exactly one fetch.
    policy.get_action(env=_environment(), observation=synthetic_observation)
    assert clients[0].get_action_calls == 1

    # Reset env 1 only; env 0 still has its chunk.
    policy.reset(env_ids=torch.tensor([1]))

    action = policy.get_action(env=_environment(), observation=synthetic_observation)
    # No new chunk fetched (env 0 not yet exhausted, so .all() is False).
    assert clients[0].get_action_calls == 1
    expected_hold = policy._extract_hold_action(synthetic_observation)
    torch.testing.assert_close(action[1], expected_hold[1])


def test_shared_policy_sizes_remote_state_from_its_robot_batch(
    policy_config_yaml, synthetic_observation, fake_client_factory
):
    """Two robots share one controller without repeating their batch size in configuration."""
    clients = fake_client_factory(ping_ok=True)
    policy = MultiRobotPolicy(
        MultiRobotPolicyCfg(
            policies={
                "shared": {
                    "type": "gr00t_remote_closedloop",
                    "params": {
                        "policy_config_yaml_path": policy_config_yaml,
                        "policy_device": "cpu",
                        "remote_host": "unused",
                        "remote_port": 0,
                    },
                }
            },
            assignments={"left": "shared", "right": "shared"},
        )
    )
    env = _environment()
    env.device = "cpu"
    from isaaclab_arena.utils.observation_bindings import ObservationBinding

    env.cfg = SimpleNamespace(
        observation_bindings=[
            binding
            for key in ("left", "right")
            for binding in (
                ObservationBinding(f"{key}_policy", None, key, "policy"),
                ObservationBinding("camera_obs", f"{key}_robot_head_cam_rgb", key, "camera_obs", "robot_head_cam_rgb"),
            )
        ]
    )
    env.single_action_space = gym.spaces.Box(-1.0, 1.0, (2 * EXPECTED_ACTION_DIM,))
    env.action_space = gym.vector.utils.batch_space(env.single_action_space, NUM_ENVS)
    terms = {f"{key}_action": SimpleNamespace(cfg=SimpleNamespace(asset_name=key)) for key in ("left", "right")}
    env.action_manager = SimpleNamespace(
        active_terms=list(terms),
        action_term_dim=[EXPECTED_ACTION_DIM, EXPECTED_ACTION_DIM],
        get_term=terms.__getitem__,
    )
    observations = {f"{key}_policy": synthetic_observation["policy"] for key in ("left", "right")}
    observations["camera_obs"] = {
        f"{key}_robot_head_cam_rgb": synthetic_observation["camera_obs"]["robot_head_cam_rgb"]
        for key in ("left", "right")
    }
    child = policy.policies["shared"]
    try:
        policy.set_task_description("pick up the brown box")
        policy.reset()
        assert clients[0].reset_called
        action = policy.get_action(env, observations)
        assert action.shape == (NUM_ENVS, 2 * EXPECTED_ACTION_DIM)
        assert child.num_envs == 2 * NUM_ENVS
        assert clients[0].get_action_calls == 1
        policy.reset(torch.tensor([1]))
        assert policy.get_action(env, observations).shape == action.shape
        with pytest.raises(AssertionError, match="batch size changes"):
            child.get_action(_environment(1), synthetic_observation)
    finally:
        policy.close()
    with pytest.raises(AssertionError, match="closed"):
        child.get_action(_environment(2 * NUM_ENVS), synthetic_observation)
