# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare configurations without depending on partially bound callback identity."""

from functools import partial


def comparable_configuration(value):
    """Compare nested configurations and partially bound functions by their contents."""
    if isinstance(value, partial):
        return (partial, value.func, comparable_configuration(value.args), comparable_configuration(value.keywords))
    if isinstance(value, dict):
        return {key: comparable_configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(comparable_configuration(item) for item in value)
    if not isinstance(value, type) and hasattr(value, "to_dict"):
        return comparable_configuration(value.to_dict())
    return value
