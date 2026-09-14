# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task success, reporting, and reset share the progress tracker's lifecycle."""

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _controlled_predicate(env, predicate_name):
    env.predicate_calls[predicate_name] += 1
    return env.predicate_results[predicate_name]


def _make_environment_and_manager(predicate_names):
    import torch

    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTrackingRecorderCfg
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessFromProgress
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={},
        extras={},
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        predicate_results={name: torch.ones(2, dtype=torch.bool) for name in predicate_names},
        predicate_calls={name: 0 for name in predicate_names},
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(num_envs=2, device="cpu"),
    )
    objectives = [
        ProgressObjective(
            name="pick_and_place",
            sequence=[partial(_controlled_predicate, predicate_name=name) for name in predicate_names],
        )
    ]
    recorder_cfg = ProgressTrackingRecorderCfg()
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    manager = TerminationManager(
        {"success": TerminationTermCfg(func=TaskSuccessFromProgress, params={"progress_objectives": objectives})},
        env,
    )
    env.termination_manager = manager
    return env, manager, recorder


def _test_success_advances_once_and_reporting_is_passive(simulation_app):
    from isaaclab_arena.recording.progress_terms import record_progress_results

    env, manager, recorder = _make_environment_and_manager(["settle", "lift", "place"])

    for step_number, completed_predicate in enumerate(["settle", "lift", "place"], start=1):
        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == [step_number == 3, step_number == 3]
        assert env.predicate_calls[completed_predicate] == 1

        for _ in range(2):
            assert recorder.record_post_step() == (None, None)
            progress = env.extras["progress_tracking"]
            assert [len(events) for events in progress["events"]] == [step_number, step_number]
            assert [state.all_complete for state in progress["states"]] == [step_number == 3, step_number == 3]
            assert manager.get_term("success").tolist() == [step_number == 3, step_number == 3]

        assert sum(env.predicate_calls.values()) == step_number

    # The episode recorder sees the final predicate on the same step as success.
    recorded_progress = record_progress_results(env, env_id=0)["progress"]
    assert recorded_progress["all_complete"]
    assert recorded_progress["overall_score"] == 1.0
    assert [event["step"] for event in recorded_progress["events"]] == [1, 2, 3]
    return True


def _test_manager_reset_clears_only_selected_progress_and_rest_poses(simulation_app):
    import torch

    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state

    env, manager, recorder = _make_environment_and_manager(["settle", "place"])
    resting_positions = torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.3]])
    env.object_initial_rest_pose_recorder.record("object", resting_positions, torch.tensor([True, True]))
    for _ in range(2):
        env.episode_length_buf += 1
        manager.compute()
    assert manager.get_term("success").all()

    manager.reset(env_ids=torch.tensor([0]))
    env.episode_length_buf[0] = 0
    recorder.record_post_step()
    progress = env.extras["progress_tracking"]
    assert [state.all_complete for state in progress["states"]] == [False, True]
    assert [len(events) for events in progress["events"]] == [0, 2]
    positions, has_settled = get_object_initial_rest_state(env, "object")
    assert has_settled.tolist() == [False, True]
    assert torch.isnan(positions[0]).all()
    torch.testing.assert_close(positions[1], resting_positions[1])

    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [False, True]
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").all()

    # Isaac Lab translates a full reset into slice(None) for class terms.
    manager.reset()
    env.episode_length_buf.zero_()
    recorder.record_post_step()
    progress = env.extras["progress_tracking"]
    assert not any(state.all_complete for state in progress["states"])
    assert progress["events"] == [[], []]
    positions, has_settled = get_object_initial_rest_state(env, "object")
    assert not has_settled.any()
    assert torch.isnan(positions).all()
    return True


def _test_builder_installs_success_only_for_progress_objectives(simulation_app):
    from isaaclab.envs.mdp import time_out
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessFromProgress
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class _ProgressTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(
                success=[
                    ProgressObjective(name="done", sequence=[partial(_controlled_predicate, predicate_name="done")])
                ],
                failures={
                    "object_dropped": TerminationTermCfg(
                        func=_controlled_predicate, params={"predicate_name": "object_dropped"}
                    )
                },
                timeout_s=12.0,
            )

    for task in [NoTask(), _ProgressTask()]:
        description = IsaacLabArenaEnvironment(name="progress_success_builder", scene=Scene(), task=task)
        builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, device="cpu"))
        env_cfg, _ = builder.compose_manager_cfg()
        success_term = getattr(env_cfg.terminations, "success", None)
        if isinstance(task, _ProgressTask):
            assert isinstance(success_term, TerminationTermCfg)
            assert success_term.func is TaskSuccessFromProgress
            assert len(success_term.params["progress_objectives"]) == 1
            assert env_cfg.terminations.object_dropped.func is _controlled_predicate
            # The task configuration is authoritative, not the constructor's default episode length.
            assert env_cfg.episode_length_s == 12.0
        else:
            assert success_term is None
            assert env_cfg.episode_length_s == task.episode_length_s
        assert env_cfg.terminations.time_out.func is time_out
        assert env_cfg.terminations.time_out.time_out
    return True


def _test_builder_rejects_success_owned_by_other_components(simulation_app):
    from unittest.mock import patch

    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    for component_name in ["scene", "embodiment"]:
        description = IsaacLabArenaEnvironment(
            name="duplicate_success_builder", scene=Scene(), task=NoTask(), embodiment=NoEmbodiment()
        )
        component = getattr(description, component_name)
        legacy_termination_cfg = SimpleNamespace(success=TerminationTermCfg(func=_controlled_predicate))
        builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
        with patch.object(component, "get_termination_cfg", return_value=legacy_termination_cfg):
            with pytest.raises(AssertionError, match="success"):
                builder.compose_manager_cfg()
    return True


def _test_task_termination_config_validation(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    first_config = TaskTerminationCfg(timeout_s=10.0)
    second_config = TaskTerminationCfg(timeout_s=20.0)
    first_config.failures["object_dropped"] = TerminationTermCfg(func=_controlled_predicate)
    assert second_config.failures == {}
    assert first_config.success == second_config.success == []

    for invalid_timeout in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(AssertionError, match="timeout_s"):
            TaskTerminationCfg(timeout_s=invalid_timeout)
    for reserved_name in ["success", "time_out"]:
        with pytest.raises(AssertionError, match="reserved"):
            TaskTerminationCfg(timeout_s=10.0, failures={reserved_name: TerminationTermCfg(func=_controlled_predicate)})
    with pytest.raises(AssertionError, match="timeout_s"):
        TaskTerminationCfg(
            timeout_s=10.0, failures={"truncated": TerminationTermCfg(func=_controlled_predicate, time_out=True)}
        )
    with pytest.raises(AssertionError, match="ProgressObjective"):
        TaskTerminationCfg(timeout_s=10.0, success=[TerminationTermCfg(func=_controlled_predicate)])
    return True


def _test_builder_rejects_task_without_unified_termination_config(simulation_app):
    from unittest.mock import patch

    import pytest

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask

    task = NoTask()
    description = IsaacLabArenaEnvironment(name="invalid_task_config", scene=Scene(), task=task)
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
    with patch.object(task, "get_termination_cfg", return_value=SimpleNamespace()):
        with pytest.raises(AssertionError, match="TaskTerminationCfg"):
            builder.compose_manager_cfg()
    return True


def test_success_advances_once_and_reporting_is_passive():
    assert run_function_with_persistent_simulation_app(_test_success_advances_once_and_reporting_is_passive)


def test_manager_reset_clears_only_selected_progress_and_rest_poses():
    assert run_function_with_persistent_simulation_app(_test_manager_reset_clears_only_selected_progress_and_rest_poses)


def test_builder_installs_success_only_for_progress_objectives():
    assert run_function_with_persistent_simulation_app(_test_builder_installs_success_only_for_progress_objectives)


def test_builder_rejects_success_owned_by_other_components():
    assert run_function_with_persistent_simulation_app(_test_builder_rejects_success_owned_by_other_components)


def test_task_termination_config_validation():
    assert run_function_with_persistent_simulation_app(_test_task_termination_config_validation)


def test_builder_rejects_task_without_unified_termination_config():
    assert run_function_with_persistent_simulation_app(_test_builder_rejects_task_without_unified_termination_config)
