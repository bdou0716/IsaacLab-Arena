# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Preserve observation ownership while robot configurations are combined."""

from dataclasses import dataclass, fields
from typing import Any

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg


@dataclass(frozen=True)
class ObservationBinding:
    """Map a composed observation to its owner and original controller name."""

    source_group: str
    source_term: str | None
    robot_key: str | None
    """Runtime scene key, or None for observations shared with every robot."""
    local_group: str
    local_term: str | None = None

    @property
    def source(self) -> tuple[str, str | None]:
        """Identify a whole group or one term in the shared camera group."""
        return self.source_group, self.source_term


def observation_bindings(
    cfg: Any, robot_key: str | None = None, instance_key: str | None = None
) -> list[ObservationBinding]:
    """Describe one known contributor's already named observation configuration."""
    if cfg is None:
        return []
    prefix = f"{instance_key}_" if instance_key is not None else ""
    bindings = []
    for group_field in fields(cfg):
        group = getattr(cfg, group_field.name)
        if not isinstance(group, ObservationGroupCfg):
            continue
        if group_field.name == "camera_obs":
            assert not group.concatenate_terms, "Shared robot camera observations must retain named terms"
            for term_field in fields(group):
                if isinstance(getattr(group, term_field.name), ObservationTermCfg):
                    bindings.append(
                        ObservationBinding(
                            group_field.name,
                            term_field.name,
                            robot_key,
                            group_field.name,
                            term_field.name.removeprefix(prefix),
                        )
                    )
        else:
            bindings.append(
                ObservationBinding(
                    group_field.name,
                    None,
                    robot_key,
                    group_field.name.removeprefix(prefix),
                )
            )
    return bindings


def override_observation_bindings(
    defaults: list[ObservationBinding], overrides: list[ObservationBinding]
) -> list[ObservationBinding]:
    """Replace declared ownership only for observations present in this contribution."""
    bindings = {binding.source: binding for binding in defaults}
    assert len({binding.source for binding in overrides}) == len(overrides), "Observation bindings must be unique"
    for binding in overrides:
        assert binding.source in bindings, f"Observation binding refers to an absent observation: {binding.source}"
        assert (binding.source_term is None) == (
            binding.local_term is None
        ), "Whole groups and individual terms must keep their observation structure"
        bindings[binding.source] = binding
    return list(bindings.values())
