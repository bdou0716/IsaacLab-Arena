# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode-level completion metrics for gear insertion."""

from __future__ import annotations

import logging
import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg

logger = logging.getLogger(__name__)

_GEAR_GATE_NAMES = ("xy", "z", "upright", "support", "velocity")


def _terminal_diagnostics(success_predicate, env_ids, gear_names: tuple[str, ...]) -> list[dict[str, dict[str, bool]]]:
    """Build per-episode completion and gate diagnostics for each gear."""

    assert success_predicate.results.ndim == 2, "Gear insertion success results must have shape (num_gears, num_envs)."
    assert success_predicate.results.shape[0] == len(gear_names), (
        f"Gear insertion success exposes {success_predicate.results.shape[0]} gears, but {len(gear_names)} names were"
        " provided."
    )
    assert len(success_predicate.predicates) == len(gear_names), (
        f"Gear insertion success configures {len(success_predicate.predicates)} gear predicates, but "
        f"{len(gear_names)} names were provided."
    )

    per_gear_completion = success_predicate.results[:, env_ids].transpose(0, 1).tolist()
    per_gear_gates = []
    for predicate_cfg in success_predicate.predicates:
        gear_predicate = predicate_cfg.func
        assert hasattr(gear_predicate, "results"), "Gear insertion child predicate does not expose gate results."
        assert gear_predicate.results.ndim == 2, "Gear gate results must have shape (num_gates, num_envs)."
        assert gear_predicate.results.shape == (len(_GEAR_GATE_NAMES), success_predicate.results.shape[1]), (
            f"Gear gate results have shape {tuple(gear_predicate.results.shape)}; expected "
            f"({len(_GEAR_GATE_NAMES)}, {success_predicate.results.shape[1]})."
        )
        per_gear_gates.append(gear_predicate.results[:, env_ids].transpose(0, 1).tolist())

    episodes = []
    for env_index, completion in enumerate(per_gear_completion):
        episode = {}
        for gear_index, gear_name in enumerate(gear_names):
            episode[gear_name] = {
                "success": bool(completion[gear_index]),
                **dict(zip(_GEAR_GATE_NAMES, per_gear_gates[gear_index][env_index], strict=True)),
            }
        episodes.append(episode)
    return episodes


class GearInsertionFractionRecorder(RecorderTerm):
    """Record the terminal fraction of gears satisfying the success criteria."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name
        self.gear_names = tuple(cfg.gear_names)
        self.first_reset = True

    def record_pre_reset(self, env_ids):
        if self.first_reset:
            assert len(env_ids) == self._env.num_envs
            self.first_reset = False
            return None, None

        progress_tracker = self._env.progress_tracker
        assert progress_tracker is not None, "Gear insertion diagnostics require task success tracking."
        success_predicate = progress_tracker.get_predicate("gear_insertion")
        if not hasattr(success_predicate, "results"):
            raise TypeError("gear insertion success predicate does not expose per-gear completion")
        per_gear = success_predicate.results[:, env_ids].transpose(0, 1)
        assert per_gear.ndim == 2 and per_gear.shape[1] == len(self.gear_names), (
            f"Gear completion results have shape {tuple(per_gear.shape)}; expected "
            f"(num_episodes, {len(self.gear_names)})."
        )
        logger.warning(
            "terminal per-gear diagnostics: %s",
            _terminal_diagnostics(success_predicate, env_ids, self.gear_names),
        )
        fractions = per_gear.to(torch.float32).mean(dim=-1)
        return self.name, fractions


@configclass
class GearInsertionFractionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = GearInsertionFractionRecorder
    name: str = "gear_insertion_fraction"
    gear_names: tuple[str, ...] = MISSING


def compute_gear_insertion_fraction(recorded_metric_data: list[np.ndarray]) -> float:
    """Average terminal per-episode gear completion fractions."""

    if not recorded_metric_data:
        return 0.0
    values = np.concatenate(recorded_metric_data)
    if values.ndim != 1:
        raise ValueError("gear insertion fraction samples must be one-dimensional")
    return float(np.mean(values))


class GearInsertionFractionMetric(MetricBase):
    """Report the fraction of individually seated gears at episode termination."""

    name = "gear_insertion_fraction"
    recorder_term_name = "gear_insertion_fraction"

    def __init__(self, gear_names: tuple[str, ...]):
        self.gear_names = gear_names

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return GearInsertionFractionRecorderCfg(name=self.recorder_term_name, gear_names=self.gear_names)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(
            compute_metric_func=compute_gear_insertion_fraction,
            params={},
            recorder_term_name=self.recorder_term_name,
        )
