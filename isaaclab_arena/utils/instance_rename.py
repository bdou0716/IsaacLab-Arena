# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Copy and namespace one robot's configuration without changing its source.

Scene entities, manager terms, robot prim paths, entity references, frame sensors,
and camera bindings use the instance key. Camera observations retain their shared
outer group. Validation checks explicit references, omitted entity defaults, and
literal manager lookups in inspectable Python functions. It cannot certify arbitrary
helper calls or dynamically constructed names; embodiment authors must avoid them.
"""

import ast
import copy
import inspect
import keyword
import textwrap
import torch
from dataclasses import fields, is_dataclass
from typing import Any, Literal

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg

from isaaclab_arena.utils.configclass import make_configclass


def validate_instance_key(instance_key: str) -> None:
    """Validate an instance name before it becomes an asset identifier."""
    assert isinstance(instance_key, str) and instance_key.isidentifier(), "Instance key must be an identifier"
    assert not keyword.iskeyword(instance_key), "Instance key must not be a Python keyword"
    assert instance_key.isascii() and instance_key.islower(), "Instance keys must use lowercase ASCII identifiers"
    assert instance_key != "robot", "Instance key must differ from the unkeyed robot scene name"


def instance_scene_name(instance_key: str | None, name: str) -> str:
    """Return the runtime scene name for a robot-owned entity."""
    if instance_key is None:
        return name
    return instance_key if name == "robot" else f"{instance_key}_{name}"


def _is_robot_prim_path(value: str) -> bool:
    return value == "{ENV_REGEX_NS}/Robot" or value.startswith("{ENV_REGEX_NS}/Robot/")


def robot_last_action(env, action_names: tuple[str, ...]) -> torch.Tensor:
    """Return raw action terms in the caller-supplied action-name order."""
    return torch.cat([env.action_manager.get_term(name).raw_actions for name in action_names], dim=-1)


def scope_last_action(cfg: Any, action_names: tuple[str, ...]) -> Any:
    """Copy observations and bind default action observations to one robot."""
    if cfg is None:
        return None
    copied = copy.deepcopy(cfg)
    for group_field in fields(copied):
        group = getattr(copied, group_field.name)
        if not isinstance(group, ObservationGroupCfg):
            continue
        for term_field in fields(group):
            _bind_last_action(getattr(group, term_field.name), action_names)
    return copied


def _bind_last_action(term: Any, action_names: tuple[str, ...]) -> bool:
    """Bind an unconfigured last-action observation to its robot's action terms."""
    if isinstance(term, ObservationTermCfg) and term.func is mdp.last_action and not term.params:
        term.func = robot_last_action
        term.params = {"action_names": action_names}
        return True
    return False


def rename_instance_cfg(
    cfg: Any,
    instance_key: str,
    scene_names: tuple[str, ...],
    action_names: tuple[str, ...],
    kind: Literal[
        "scene", "actions", "observations", "events", "rewards", "curriculum", "commands", "recorders", "terminations"
    ],
) -> Any:
    """Return an independent configuration with this robot's names rewritten.

    Args:
        cfg: An outer scene or manager configuration instance, or None.
        instance_key: Unique identifier of the robot.
        scene_names: Original scene and camera field names.
        action_names: Original action term names in concatenation order.
        kind: Scene, observations, or another manager's name.

    Returns:
        A new configuration with copied values and names scoped to the instance.
    """
    validate_instance_key(instance_key)
    assert instance_key not in scene_names, "Instance key must differ from every original scene field"
    if cfg is None:
        return None
    scene_map = {name: instance_scene_name(instance_key, name) for name in scene_names}
    action_map = {name: f"{instance_key}_{name}" for name in action_names}
    copied = copy.deepcopy(cfg)
    renamed_fields = []
    for field in fields(copied):
        value = getattr(copied, field.name)
        location = f"{instance_key}:{kind}.{field.name}"
        if kind == "observations" and field.name == "camera_obs":
            camera_fields = []
            for camera_field in fields(value):
                term = getattr(value, camera_field.name)
                name = (
                    f"{instance_key}_{camera_field.name}" if isinstance(term, ObservationTermCfg) else camera_field.name
                )
                camera_fields.append(
                    (name, camera_field.type, _rewrite(term, scene_map, action_map, f"{location}.{name}", instance_key))
                )
            group = make_configclass("InstanceCameraObsCfg", camera_fields, bases=(ObservationGroupCfg,))()
            renamed_fields.append(("camera_obs", type(group), group))
            continue
        name = scene_map[field.name] if kind == "scene" else f"{instance_key}_{field.name}"
        value = _rewrite(value, scene_map, action_map, location, instance_key)
        renamed_fields.append((name, field.type, value))
    names = [name for name, _, _ in renamed_fields]
    assert len(names) == len(set(names)), f"{kind}: instance names collide"
    renamed = make_configclass(f"{type(cfg).__name__}Instance", renamed_fields)()
    _assert_no_old_references(renamed, scene_map, kind)
    return renamed


def _rewrite(value, scene_map, action_map, location, instance_key, attribute=""):
    """Rewrite configured entity names while preserving physical link names."""
    if isinstance(value, str):
        if attribute == "prim_path":
            key = scene_map.get("robot")
            if key is not None and _is_robot_prim_path(value):
                return f"{{ENV_REGEX_NS}}/{key[0].upper()}{key[1:]}" + value[len("{ENV_REGEX_NS}/Robot") :]
        if attribute in {
            "asset_name",
            "frame_transformer_name",
            "camera_name",
            "sensor_name",
            "sensor_names",
            "scene_writes",
            "write_pose_list",
        }:
            return scene_map.get(value, value)
        if attribute == "action_name":
            return action_map.get(value, value)
        return value
    if isinstance(value, SceneEntityCfg):
        value.name = scene_map.get(value.name, value.name)
        return value
    if _bind_last_action(value, tuple(action_map.values())):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        if isinstance(value, FrameTransformerCfg.FrameCfg):
            target_name = value.name or value.prim_path.rstrip("/").rsplit("/", 1)[-1]
            value.name = f"{instance_key}_{target_name}"
        if hasattr(value, "func") and hasattr(value, "params"):
            _validate_callable(value.func, value.params, scene_map, action_map, location)
        for field in fields(value):
            child = getattr(value, field.name)
            object.__setattr__(
                value,
                field.name,
                _rewrite(child, scene_map, action_map, f"{location}.{field.name}", instance_key, field.name),
            )
        return value
    if isinstance(value, dict):
        return {
            key: _rewrite(child, scene_map, action_map, f"{location}.{key}", instance_key, key)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_rewrite(child, scene_map, action_map, location, instance_key, attribute) for child in value)
    return value


def _default_scene_names(value):
    """Find entity defaults in supported configuration containers."""
    if isinstance(value, SceneEntityCfg):
        yield value.name
    elif isinstance(value, dict):
        for child in value.values():
            yield from _default_scene_names(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _default_scene_names(child)


def _validate_callable(func, params, scene_map, action_map, location):
    """Reject implicit entity defaults and literal scene or action lookups."""
    assert not isinstance(func, str), f"{location}: keyed terms require a callable, not a string reference"
    callable_body = func.__call__ if inspect.isclass(func) else inspect.unwrap(func)
    bodies = (func.__init__, callable_body) if inspect.isclass(func) else (callable_body,)
    for body in bodies:
        if body is object.__init__:
            continue
        try:
            signature = inspect.signature(body)
        except (TypeError, ValueError) as error:
            raise AssertionError(f"{location}: cannot inspect callable entity defaults") from error
        for name, parameter in signature.parameters.items():
            if body is callable_body and name in params:
                continue
            for entity_name in _default_scene_names(parameter.default):
                assert entity_name not in scene_map, (
                    f"{location}: pass '{name}' explicitly; its default references '{entity_name}'. "
                    "Managed constructors must read entity bindings from their configuration."
                )
        try:
            source = ast.parse(textwrap.dedent(inspect.getsource(body)))
        except (OSError, TypeError, IndentationError, SyntaxError) as error:
            raise AssertionError(f"{location}: cannot inspect keyed term body {body}") from error
        for node in ast.walk(source):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_term":
                arguments = list(node.args[:1]) + [item.value for item in node.keywords if item.arg == "name"]
                for argument in arguments:
                    if isinstance(argument, ast.Constant):
                        name = argument.value
                        assert (
                            name not in action_map
                        ), f"{location}: literal action-term lookup '{name}' cannot be renamed"
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "scene":
                if isinstance(node.slice, ast.Constant):
                    name = node.slice.value
                    assert name not in scene_map, f"{location}: literal scene lookup '{name}' cannot be renamed"


def _assert_no_old_references(value, scene_map, location):
    """Reject stale entity names after rewriting a configuration."""
    if isinstance(value, str):
        assert value not in scene_map, f"{location}: unrenamed scene reference '{value}'"
    elif isinstance(value, SceneEntityCfg):
        assert value.name not in scene_map, f"{location}: unrenamed scene reference '{value.name}'"
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            child = getattr(value, field.name)
            if field.name == "prim_path" and isinstance(child, str):
                assert not _is_robot_prim_path(child), f"{location}: unrenamed robot prim path"
            _assert_no_old_references(child, scene_map, f"{location}.{field.name}")
    elif isinstance(value, dict):
        for name, child in value.items():
            _assert_no_old_references(child, scene_map, f"{location}.{name}")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_old_references(child, scene_map, location)
