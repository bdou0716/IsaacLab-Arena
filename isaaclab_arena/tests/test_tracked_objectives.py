# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recorded events must not change task success, ordering, or timeout behavior."""

from functools import partial

from isaaclab_arena.tests.test_task_success_from_progress import (
    _controlled_predicate,
    _make_environment,
    _make_environment_and_manager,
)
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _objective(name):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    return ProgressObjective(
        name=name,
        predicate_sequence=[partial(_controlled_predicate, predicate_name=name)],
    )


def _test_tracked_events_do_not_gate_success_and_reset_independently(simulation_app):
    from dataclasses import replace

    import pytest

    from isaaclab_arena.recording.progress_terms import record_progress_results

    env, manager, recorder = _make_environment_and_manager(
        ["arrived", "found", "fallen"],
        success_objectives=[replace(_objective("arrived"), score=0.5)],
        tracked_objectives=[_objective("found"), _objective("fallen")],
    )
    env.predicate_results["arrived"][:] = False
    env.predicate_results["fallen"][:] = False
    env.episode_length_buf[:] = 1
    assert manager.compute().tolist() == [False, False]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert not record["all_complete"]
    assert record["overall_score"] == pytest.approx(1.0 / 2.5)
    assert [event["objective"] for event in record["events"]] == ["found"]

    env.predicate_results["arrived"][:] = True
    env.episode_length_buf[:] = 2
    assert manager.compute().tolist() == [True, True]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["all_complete"]
    assert record["overall_score"] == pytest.approx(1.5 / 2.5)
    assert set(record["objectives"]) == {"arrived", "found", "fallen"}
    assert [event["objective"] for event in record["events"]] == ["found", "arrived"]
    assert env.predicate_calls["found"] == 1
    assert env.predicate_calls["fallen"] == 2
    assert record["has_success_criteria"]
    assert {name: objective["role"] for name, objective in record["objectives"].items()} == {
        "arrived": "success",
        "found": "tracked",
        "fallen": "tracked",
    }

    env.predicate_results["fallen"][0] = True
    env.episode_length_buf += 1
    manager.compute()
    recorder.record_post_step()
    states = env.progress_tracker.get_state()
    assert [state.all_complete for state in states] == [True, True]
    assert states[0].overall_score == pytest.approx(1.0)
    assert states[1].overall_score == pytest.approx(1.5 / 2.5)

    manager.reset(env_ids=[0])
    states = env.progress_tracker.get_state()
    assert not states[0].progress_objectives["found"].is_complete
    assert states[1].progress_objectives["found"].is_complete
    assert env.progress_tracker.get_events()[0] == []
    assert len(env.progress_tracker.get_events()[1]) == 2
    env.predicate_results["arrived"][0] = False
    env.predicate_results["found"][0] = False
    env.episode_length_buf[0] = 1
    assert manager.compute().tolist() == [False, True]
    return True


def _test_tracked_only_builder_records_until_timeout(simulation_app):
    import torch
    from types import SimpleNamespace

    import pytest
    from isaaclab.managers import TerminationManager

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg
    from isaaclab_arena.recording.common_terms import record_core_episode_results
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class TrackedOnlyTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=1.0, tracked=[_objective("found")])

    definition = IsaacLabArenaEnvironment(name="tracked_only", scene=Scene(), task=TrackedOnlyTask())
    cfg, _ = ArenaEnvBuilder(
        definition, ArenaEnvBuilderCfg(num_envs=2, device="cpu", solve_relations=False)
    ).compose_manager_cfg()
    assert cfg.episode_length_s == 1.0
    assert set(cfg.terminations.to_dict()) == {"progress_tracking", "time_out"}

    env = _make_environment(["found"])
    env.max_episode_length = 2
    env.cfg = SimpleNamespace(seed=7)
    env.get_episode_index = lambda env_id: 0
    env.get_language_instruction = lambda: None
    manager = TerminationManager(cfg.terminations, env)
    env.termination_manager = manager
    recorder_cfg = cfg.recorders.progress_tracking
    recorder = recorder_cfg.class_type(recorder_cfg, env)

    assert not env.progress_tracker.has_success_criteria
    env.episode_length_buf[:] = 1
    assert not manager.compute().any()
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["overall_score"] == 1.0
    assert not record["all_complete"]
    assert not record["has_success_criteria"]
    assert record["objectives"]["found"]["role"] == "tracked"
    assert [event["objective"] for event in record["events"]] == ["found"]
    assert record_core_episode_results(env, env_id=0)["success"] is None

    success_recorder_cfg = SuccessRecorderCfg()
    success_recorder = success_recorder_cfg.class_type(success_recorder_cfg, env)
    with pytest.raises(AssertionError, match="requires task success objectives"):
        success_recorder.record_pre_reset([0, 1])

    env.episode_length_buf[:] = torch.tensor([2, 1])
    assert manager.compute().tolist() == [True, False]
    assert manager.time_outs.tolist() == [True, False]
    assert not manager.terminated.any()
    assert "success" not in manager.active_terms
    assert env.predicate_calls["found"] == 1

    manager.reset(env_ids=[0])
    assert env.progress_tracker.get_events()[0] == []
    assert len(env.progress_tracker.get_events()[1]) == 1
    states = env.progress_tracker.get_state()
    assert not states[0].progress_objectives["found"].is_complete
    assert states[1].progress_objectives["found"].is_complete
    return True


def _test_composite_tracked_events_ignore_order_and_final_conditions(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class ObservedTask(NoTask):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=2.0, success=[_objective(self.name)], tracked=[_objective("event")])

    cfg = CompositeTaskBase(
        [ObservedTask("first"), ObservedTask("second")],
        subtasks_are_sequential=True,
        desired_subtask_success_state=[False, True],
    ).get_termination_cfg()
    assert [objective.name for objective in cfg.tracked] == ["subtask_0/event", "subtask_1/event"]
    assert [objective.parent_subtask_idx for objective in cfg.tracked] == [0, 1]
    env, manager, _ = _make_environment_and_manager(
        ["first", "second", "event"],
        success_objectives=cfg.success,
        tracked_objectives=cfg.tracked,
        subtasks_are_sequential=cfg.subtasks_are_sequential,
        desired_subtask_success_state=cfg.desired_subtask_success_state,
    )
    env.predicate_results["first"][:] = False
    env.predicate_results["event"][1] = False
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["second"] == 0
    assert env.progress_tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
    assert {event.progress_objective for event in env.progress_tracker.get_events()[0]} == {
        "subtask_0/event",
        "subtask_1/event",
    }
    assert env.progress_tracker.get_events()[1] == []

    env.predicate_results["first"][:] = True
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["second"] == 0
    env.predicate_results["first"][:] = False
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
    # The recorded event is true in one environment and false in the other.
    # Neither value changes the requested false final condition for the first task.
    return True


def _test_objective_names_are_unique_across_both_lists(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    with pytest.raises(AssertionError, match="names must be unique"):
        ProgressTracker([_objective("same")], 2, "cpu", tracked_objectives=[_objective("same")])
    return True


def test_tracked_events_do_not_gate_success_and_reset_independently():
    assert run_function_with_persistent_simulation_app(_test_tracked_events_do_not_gate_success_and_reset_independently)


def test_tracked_only_builder_records_until_timeout():
    assert run_function_with_persistent_simulation_app(_test_tracked_only_builder_records_until_timeout)


def test_composite_tracked_events_ignore_order_and_final_conditions():
    assert run_function_with_persistent_simulation_app(_test_composite_tracked_events_ignore_order_and_final_conditions)


def test_objective_names_are_unique_across_both_lists():
    assert run_function_with_persistent_simulation_app(_test_objective_names_are_unique_across_both_lists)


def _test_shared_stateless_predicate_runs_once_for_both_roles(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    shared = partial(_controlled_predicate, predicate_name="shared")
    env, manager, _ = _make_environment_and_manager(
        ["shared"],
        success_objectives=[ProgressObjective(name="success", predicate_sequence=[shared])],
        tracked_objectives=[ProgressObjective(name="tracked", predicate_sequence=[shared])],
    )
    env.predicate_results["shared"][1] = False
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["shared"] == 1
    assert [len(events) for events in env.progress_tracker.get_events()] == [2, 0]
    manager.reset(env_ids=[0])
    env.predicate_results["shared"][:] = True
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["shared"] == 2
    assert manager.get_term("success").tolist() == [True, True]
    assert [len(events) for events in env.progress_tracker.get_events()] == [2, 2]
    return True


def _test_reused_temporal_requirement_has_independent_role_counters(simulation_app):
    import torch

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _make_environment(["gate", "shared"])
    requirement = TrueForConsecutiveStepsCfg(
        predicate=partial(_controlled_predicate, predicate_name="shared"), required_steps=2
    )
    success = ProgressObjective(
        name="success",
        predicate_sequence=[partial(_controlled_predicate, predicate_name="gate"), requirement],
    )
    tracked = ProgressObjective(name="tracked", predicate_sequence=[requirement])
    tracker = ProgressTracker([success], 2, "cpu", env=env, tracked_objectives=[tracked])
    env.predicate_results["gate"][:] = False
    for step in (1, 2):
        tracker.step(env, torch.full((2,), step, dtype=torch.long))
    assert not tracker.is_complete().any()
    assert tracker.get_state()[0].progress_objectives["tracked"].is_complete
    env.predicate_results["gate"][:] = True
    for step in (3, 4):
        tracker.step(env, torch.full((2,), step, dtype=torch.long))
    assert not tracker.is_complete().any()
    tracker.step(env, torch.tensor([5, 5]))
    assert tracker.is_complete().all()

    tracker.reset([0])
    tracker.step(env, torch.tensor([0, 5]))
    assert tracker.is_complete().tolist() == [False, True]
    assert not tracker.get_state()[0].progress_objectives["tracked"].is_complete
    assert tracker.get_state()[1].progress_objectives["tracked"].is_complete
    tracker.step(env, torch.tensor([1, 5]))
    assert tracker.get_state()[0].progress_objectives["tracked"].is_complete
    assert not tracker.is_complete()[0]
    return True


def test_shared_stateless_predicate_runs_once_for_both_roles():
    assert run_function_with_persistent_simulation_app(_test_shared_stateless_predicate_runs_once_for_both_roles)


def test_reused_temporal_requirement_has_independent_role_counters():
    assert run_function_with_persistent_simulation_app(_test_reused_temporal_requirement_has_independent_role_counters)


def _test_temporal_roles_share_only_the_instantaneous_check(simulation_app):
    import torch

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _make_environment(["shared"])
    requirement = TrueForConsecutiveStepsCfg(
        predicate=partial(_controlled_predicate, predicate_name="shared"), required_steps=2
    )
    tracker = ProgressTracker(
        [ProgressObjective(name="required", predicate_sequence=[requirement])],
        2,
        "cpu",
        env=env,
        tracked_objectives=[ProgressObjective(name="observed", predicate_sequence=[requirement])],
    )
    tracker.step(env, torch.tensor([1, 1]))
    tracker.step(env, torch.tensor([1, 1]))
    assert env.predicate_calls["shared"] == 1
    assert not tracker.is_complete().any()
    tracker.step(env, torch.tensor([2, 1]))
    assert tracker.is_complete().tolist() == [True, False]
    assert [len(events) for events in tracker.get_events()] == [2, 0]
    tracker.step(env, torch.tensor([2, 2]))
    assert tracker.is_complete().all()
    assert [len(events) for events in tracker.get_events()] == [2, 2]

    tracker.reset([0])
    tracker.step(env, torch.tensor([0, 2]))
    assert tracker.is_complete().tolist() == [False, True]
    tracker.step(env, torch.tensor([2, 2]))  # An unobserved step breaks the streak.
    assert tracker.is_complete().tolist() == [False, True]
    tracker.step(env, torch.tensor([3, 2]))
    assert tracker.is_complete().all()
    return True


def _test_read_only_managed_predicate_can_use_distinct_wrappers(simulation_app):
    from isaaclab.managers import ManagerTermBase, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    class ReadOnlyPredicate(ManagerTermBase):
        def __call__(self, env):
            return env.predicate_results["shared"]

    env = _make_environment(["shared"])
    shared = ReadOnlyPredicate(TerminationTermCfg(func=ReadOnlyPredicate), env)
    tracker = ProgressTracker(
        [ProgressObjective(name="required", predicate_sequence=[partial(shared)])],
        2,
        "cpu",
        env=env,
        tracked_objectives=[ProgressObjective(name="observed", predicate_sequence=[partial(shared)])],
    )
    tracker.step(env)
    assert tracker.is_complete().all()
    assert [len(events) for events in tracker.get_events()] == [2, 2]
    return True


def _test_settling_observers_cannot_record_success_reference_early(simulation_app):
    import torch
    from types import SimpleNamespace

    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_settling import objects_settled
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    instantaneous = partial(objects_settled, object_names=["object"])
    configured = TerminationTermCfg(func=objects_settled, params={"object_names": ["object"]})
    observers = (
        instantaneous,
        configured,
        TrueForConsecutiveStepsCfg(predicate=instantaneous, required_steps=2),
        TrueForConsecutiveStepsCfg(predicate=configured, required_steps=2),
    )
    for observer in observers:
        env = _make_environment(["gate"])
        env.predicate_results["gate"][:] = False
        positions = torch.zeros((2, 3))
        velocities = torch.zeros((2, 3))
        env.scene = SimpleNamespace(deformable_objects={})
        env.arena_world = SimpleNamespace(
            get_position_w=lambda name: positions,
            get_root_linear_velocity_w=lambda name: velocities,
            get_root_angular_velocity_w=lambda name: velocities,
        )
        success = ProgressObjective(
            name="required",
            predicate_sequence=[
                partial(_controlled_predicate, predicate_name="gate"),
                instantaneous,
            ],
        )
        with pytest.raises(AssertionError, match="'observer'.*initial rest poses"):
            ProgressTracker(
                [success],
                2,
                "cpu",
                env=env,
                tracked_objectives=[ProgressObjective(name="observer", predicate_sequence=[observer])],
            )
        _, recorded = env.object_initial_rest_pose_recorder.get("object")
        assert not recorded.any()

        tracker = ProgressTracker([success], 2, "cpu", env=env)
        tracker.step(env)
        _, recorded = env.object_initial_rest_pose_recorder.get("object")
        assert not recorded.any()
        positions[:, 2] = torch.tensor([2.0, 3.0])
        env.predicate_results["gate"][:] = True
        tracker.step(env)
        tracker.step(env)
        recorded_positions, recorded = env.object_initial_rest_pose_recorder.get("object")
        assert recorded.all()
        torch.testing.assert_close(recorded_positions, positions)
    return True


def _test_recording_continues_without_automatic_success_termination(simulation_app):
    from types import SimpleNamespace

    from isaaclab.managers import TerminationManager

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.metrics.success_rate import SuccessRecorderCfg
    from isaaclab_arena.recording.common_terms import record_core_episode_results
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class ObservedTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(
                timeout_s=None,
                success=[_objective("arrived")],
                tracked=[_objective("found")],
            )

    definition = IsaacLabArenaEnvironment(name="record_after_success", scene=Scene(), task=ObservedTask())
    cfg, _ = ArenaEnvBuilder(
        definition, ArenaEnvBuilderCfg(num_envs=2, device="cpu", solve_relations=False)
    ).compose_manager_cfg()
    cfg.terminations.success = None

    env = _make_environment(["arrived", "found"])
    env.cfg = SimpleNamespace(seed=7)
    env.get_episode_index = lambda env_id: 0
    env.get_language_instruction = lambda: None
    env.predicate_results["found"][:] = False
    manager = TerminationManager(cfg.terminations, env)
    env.termination_manager = manager
    recorder_cfg = cfg.recorders.progress_tracking
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    success_recorder_cfg = SuccessRecorderCfg()
    success_recorder = success_recorder_cfg.class_type(success_recorder_cfg, env)
    assert success_recorder.record_pre_reset([0, 1]) == (None, None)

    env.episode_length_buf += 1
    assert not manager.compute().any()
    recorder.record_post_step()
    assert record_core_episode_results(env, env_id=0)["success"] is True
    assert record_progress_results(env, env_id=0)["progress"]["overall_score"] == 0.5

    env.predicate_results["found"][:] = True
    env.episode_length_buf += 1
    assert not manager.compute().any()
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["overall_score"] == 1.0
    assert [event["objective"] for event in record["events"]] == ["arrived", "found"]
    assert env.predicate_calls == {"arrived": 1, "found": 2}
    assert success_recorder.record_pre_reset([0, 1])[1].tolist() == [True, True]

    manager.reset(env_ids=[0])
    assert env.progress_tracker.is_complete().tolist() == [False, True]
    assert env.progress_tracker.get_events()[0] == []
    assert len(env.progress_tracker.get_events()[1]) == 2
    return True


def _test_composite_children_require_success_objectives(simulation_app):
    import pytest

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class TrackedOnlyTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=1.0, tracked=[_objective("found")])

    with pytest.raises(AssertionError, match="must define success objectives"):
        CompositeTaskBase([TrackedOnlyTask()]).get_termination_cfg()
    return True


def test_temporal_roles_share_only_the_instantaneous_check():
    assert run_function_with_persistent_simulation_app(_test_temporal_roles_share_only_the_instantaneous_check)


def test_read_only_managed_predicate_can_use_distinct_wrappers():
    assert run_function_with_persistent_simulation_app(_test_read_only_managed_predicate_can_use_distinct_wrappers)


def test_settling_observers_cannot_record_success_reference_early():
    assert run_function_with_persistent_simulation_app(_test_settling_observers_cannot_record_success_reference_early)


def test_recording_continues_without_automatic_success_termination():
    assert run_function_with_persistent_simulation_app(_test_recording_continues_without_automatic_success_termination)


def test_composite_children_require_success_objectives():
    assert run_function_with_persistent_simulation_app(_test_composite_children_require_success_objectives)
