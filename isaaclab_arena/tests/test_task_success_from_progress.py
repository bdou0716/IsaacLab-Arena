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
            predicate_groups={
                "task_success": [partial(_controlled_predicate, predicate_name=name) for name in predicate_names]
            },
        )
    ]
    # Isaac Lab constructs recorders before the termination manager that owns progress.
    recorder_cfg = ProgressTrackingRecorderCfg(progress_objectives=objectives, advance_tracker=False)
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    assert not hasattr(env, "_progress_tracker")
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


def _test_placement_cannot_bypass_lift(simulation_app):
    env, manager, recorder = _make_environment_and_manager(["settle", "lift", "place"])
    env.predicate_results["lift"][0] = False
    for _ in range(4):
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        assert not manager.get_term("success")[0]
    assert manager.get_term("success").tolist() == [False, True]
    assert [len(events) for events in env.extras["progress_tracking"]["events"]] == [1, 3]

    env.predicate_results["lift"][0] = True
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [False, True]
    env.episode_length_buf += 1
    manager.compute()
    recorder.record_post_step()
    assert manager.get_term("success").all()
    assert [event.step for event in env.extras["progress_tracking"]["events"][0]] == [1, 5, 6]
    return True


def _test_manager_reset_clears_only_selected_progress_and_rest_poses(simulation_app):
    import torch

    from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state

    for reset_ids, reset_mask in [
        (torch.tensor([0]), [True, False]),
        (slice(1, 2), [False, True]),
        (None, [True, True]),
        (slice(None), [True, True]),
    ]:
        env, manager, recorder = _make_environment_and_manager(["settle", "place"])
        resting_positions = torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.3]])
        env.object_initial_rest_pose_recorder.record("object", resting_positions, torch.tensor([True, True]))
        for _ in range(2):
            env.episode_length_buf += 1
            manager.compute()
        assert manager.get_term("success").all()

        # A full manager reset passes slice(None) to its class terms.
        manager.reset(env_ids=reset_ids)
        env.episode_length_buf[reset_mask] = 0
        recorder.record_post_step()
        progress = env.extras["progress_tracking"]
        positions, has_settled = get_object_initial_rest_state(env, "object")
        for environment_index, was_reset in enumerate(reset_mask):
            assert progress["states"][environment_index].all_complete == (not was_reset)
            assert progress["states"][environment_index].overall_score == (0.0 if was_reset else 1.0)
            assert len(progress["events"][environment_index]) == (0 if was_reset else 2)
            assert bool(has_settled[environment_index]) == (not was_reset)
            if was_reset:
                assert torch.isnan(positions[environment_index]).all()
            else:
                torch.testing.assert_close(positions[environment_index], resting_positions[environment_index])

        new_resting_positions = resting_positions + 0.5
        env.object_initial_rest_pose_recorder.record("object", new_resting_positions, torch.tensor([True, True]))
        positions, has_settled = get_object_initial_rest_state(env, "object")
        assert has_settled.all()
        for environment_index, was_reset in enumerate(reset_mask):
            expected_position = (new_resting_positions if was_reset else resting_positions)[environment_index]
            torch.testing.assert_close(positions[environment_index], expected_position)

        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == [not was_reset for was_reset in reset_mask]
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        assert manager.get_term("success").all()
        event_steps = [[event.step for event in events] for events in env.extras["progress_tracking"]["events"]]
        assert event_steps == [[1, 2], [1, 2]]
    return True


def _test_success_results_remain_stable_after_updates_and_reset(simulation_app):
    env, manager, _ = _make_environment_and_manager(["settle", "place"])
    success_cfg = manager.get_term_cfg("success")
    env.episode_length_buf += 1
    first_result = success_cfg.func(env, **success_cfg.params)
    assert first_result.tolist() == [False, False]
    env.episode_length_buf += 1
    completed_result = success_cfg.func(env, **success_cfg.params)
    assert completed_result.tolist() == [True, True]
    assert first_result.tolist() == [False, False]

    success_cfg.func.reset(env_ids=[0])
    assert completed_result.tolist() == [True, True]
    env.episode_length_buf += 1
    after_partial_reset = success_cfg.func(env, **success_cfg.params)
    assert after_partial_reset.tolist() == [False, True]
    success_cfg.func.reset()
    env.episode_length_buf += 1
    after_full_reset = success_cfg.func(env, **success_cfg.params)
    assert after_full_reset.tolist() == [False, False]
    assert first_result.tolist() == [False, False]
    assert completed_result.tolist() == [True, True]
    assert after_partial_reset.tolist() == [False, True]
    return True


def _test_success_requires_objectives_and_one_owner(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.task_success import TaskSuccessFromProgress

    env, manager, _ = _make_environment_and_manager(["place"])
    empty_cfg = TerminationTermCfg(func=TaskSuccessFromProgress, params={"progress_objectives": []})
    with pytest.raises(AssertionError, match="at least one progress objective"):
        TaskSuccessFromProgress(empty_cfg, env)
    with pytest.raises(AssertionError, match="Only one root term"):
        TaskSuccessFromProgress(manager.get_term_cfg("success"), env)
    return True


def _test_builder_limits_progress_owned_success_to_root_pick_and_place(simulation_app):
    from unittest.mock import Mock, patch

    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.task_success import TaskSuccessFromProgress
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.utils.configclass import make_configclass

    class _LegacyProgressTask(NoTask):
        def get_progress_objectives(self):
            return [
                ProgressObjective(name="place", predicate_groups=partial(_controlled_predicate, predicate_name="place"))
            ]

        def get_termination_cfg(self):
            success = TerminationTermCfg(func=_controlled_predicate, params={"predicate_name": "place"})
            return make_configclass("LegacyTerminationCfg", [("success", TerminationTermCfg, success)])()

    pick_up_object = SimpleNamespace(
        name="object",
        get_contact_sensor_cfg=Mock(return_value=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object")),
    )
    pick_and_place = PickAndPlaceTask(
        pick_up_object,
        SimpleNamespace(name="destination"),
        SimpleNamespace(object_min_z=-0.1),
        episode_length_s=12.0,
    )
    with (
        patch.object(pick_and_place, "get_viewer_cfg", return_value=ViewerCfg()),
        patch.object(pick_and_place, "get_metrics", return_value=[]),
    ):
        for task in [
            pick_and_place,
            _LegacyProgressTask(),
            CompositeTaskBase([pick_and_place]),
            SequentialTaskBase([pick_and_place]),
            NoTask(),
        ]:
            description = IsaacLabArenaEnvironment(name="progress_success_builder", scene=Scene(), task=task)
            builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, device="cpu"))
            env_cfg, _ = builder.compose_manager_cfg()
            assert env_cfg.episode_length_s == task.get_episode_length_s()
            if task is pick_and_place:
                success = env_cfg.terminations.success
                assert success.func is TaskSuccessFromProgress
                objectives = success.params["progress_objectives"]
                assert len(objectives) == 1
                assert [predicate.func for predicate in objectives[0].predicate_groups] == [
                    objects_settled,
                    object_is_above_height,
                    object_on_destination,
                ]
                assert not env_cfg.recorders.progress_tracking.advance_tracker
                assert getattr(env_cfg.events, "reset_progress_objectives", None) is None
                original_terminations = pick_and_place.get_termination_cfg()
                assert env_cfg.terminations.object_dropped.func is original_terminations.object_dropped.func
                assert env_cfg.terminations.time_out.time_out
                assert original_terminations.success.func is object_on_destination
            elif isinstance(task, _LegacyProgressTask | CompositeTaskBase):
                assert env_cfg.terminations.success.func is task.get_termination_cfg().success.func
                assert env_cfg.recorders.progress_tracking.advance_tracker
                assert env_cfg.events.reset_progress_objectives.mode == "reset"
            else:
                assert getattr(env_cfg.terminations, "success", None) is None
                assert getattr(env_cfg.recorders, "progress_tracking", None) is None
                assert getattr(env_cfg.events, "reset_progress_objectives", None) is None
    return True


def test_success_advances_once_and_reporting_is_passive():
    assert run_function_with_persistent_simulation_app(_test_success_advances_once_and_reporting_is_passive)


def test_placement_cannot_bypass_lift():
    assert run_function_with_persistent_simulation_app(_test_placement_cannot_bypass_lift)


def test_manager_reset_clears_only_selected_progress_and_rest_poses():
    assert run_function_with_persistent_simulation_app(_test_manager_reset_clears_only_selected_progress_and_rest_poses)


def test_success_results_remain_stable_after_updates_and_reset():
    assert run_function_with_persistent_simulation_app(_test_success_results_remain_stable_after_updates_and_reset)


def test_success_requires_objectives_and_one_owner():
    assert run_function_with_persistent_simulation_app(_test_success_requires_objectives_and_one_owner)


def test_builder_limits_progress_owned_success_to_root_pick_and_place():
    assert run_function_with_persistent_simulation_app(
        _test_builder_limits_progress_owned_success_to_root_pick_and_place
    )
