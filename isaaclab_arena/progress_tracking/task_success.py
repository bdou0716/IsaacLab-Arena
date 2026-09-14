# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Connect task progress to Isaac Lab's termination evaluation and reset lifecycle."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
from isaaclab_arena.tasks.predicates.object_settling import reset_rest_pose_recorder


class ProgressBasedSuccessTerm(ManagerTermBase):
    """Connect task progress to Isaac Lab's termination and reset lifecycle.

    ArenaEnvBuilder installs this term automatically. It owns the tracker,
    advances it during termination evaluation, and reports same-step success.
    Isaac Lab's episode resets are forwarded to the tracker.
    """

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        # Isaac Lab validates required __call__ parameters before constructing this term.
        progress_objectives: list[ProgressObjective] = cfg.params["progress_objectives"]
        assert progress_objectives, "Task success requires at least one progress objective."
        assert env._progress_tracker is None, "Only one root term may own task progress."
        self._progress_tracker = ProgressTracker(progress_objectives, num_envs=env.num_envs, device=env.device)
        self._environment_ids = torch.arange(env.num_envs, device=env.device)
        env._progress_tracker = self._progress_tracker

    def __call__(self, env, progress_objectives: list[ProgressObjective]) -> torch.Tensor:
        """Advance each active stage once and return task completion per environment."""
        self._progress_tracker.step(env, step_index=env.episode_length_buf)
        return self._progress_tracker.is_complete()

    def reset(self, env_ids=None) -> None:
        """Clear progress and initial resting positions for the restarting environments."""
        selected_env_ids = self._environment_ids if env_ids is None else self._environment_ids[env_ids]
        self._progress_tracker.reset(selected_env_ids)
        reset_rest_pose_recorder(self._env, selected_env_ids)
