# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose configuration values through their declared Hydra schemas."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def hydra_context() -> AbstractContextManager[None]:
    """Initialize Hydra only when no caller-owned composition context exists."""
    if GlobalHydra.instance().is_initialized():
        return nullcontext()
    return initialize(version_base=None, config_path=None)


def compose_typed_config(cfg_type: type, values: dict[str, Any], name: str) -> Any:
    """Validate and convert values using the selected configuration schema."""
    if "defaults" in values:
        raise ValueError("Configuration values cannot override the schema defaults")
    with hydra_context():
        store = ConfigStore.instance()
        schema_name = f"{name}_schema"
        store.store(name=schema_name, node=cfg_type)
        store.store(name=name, node={"defaults": [schema_name, "_self_"], **values})
        return OmegaConf.to_object(compose(config_name=name))
