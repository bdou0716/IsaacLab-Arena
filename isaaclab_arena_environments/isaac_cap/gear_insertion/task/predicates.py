# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gear-specific insertion predicates."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from pxr import Usd


class GearIsSupported(ManagerTermBase):
    """Check that one gear bottom remains near the plate support surface."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.plate_asset_cfg: SceneEntityCfg = cfg.params["plate_asset_cfg"]
        self.gear_asset_cfg: SceneEntityCfg = cfg.params["gear_asset_cfg"]
        plate_asset = env.scene[self.plate_asset_cfg.name]
        gear_asset = env.scene[self.gear_asset_cfg.name]
        self.plate_collision_corners = self._collision_corners(
            plate_asset,
            env.device,
            collision_prim_name="platform",
            enabled_only=True,
        )
        self.gear_collision_corners = self._collision_corners(gear_asset, env.device, enabled_only=True)

    def __call__(
        self,
        env: ManagerBasedEnv,
        plate_asset_cfg: SceneEntityCfg,
        gear_asset_cfg: SceneEntityCfg,
        support_z_threshold: float,
    ) -> torch.Tensor:
        assert plate_asset_cfg.name == self.plate_asset_cfg.name
        assert gear_asset_cfg.name == self.gear_asset_cfg.name
        T_W_P = env.arena_world.get_pose_w(plate_asset_cfg.name)
        T_W_G = env.arena_world.get_pose_w(gear_asset_cfg.name)
        _, plate_top_z = self._world_collision_z_bounds(
            self.plate_collision_corners,
            T_W_P[:, :3],
            T_W_P[:, 3:],
        )
        gear_bottom_z, _ = self._world_collision_z_bounds(
            self.gear_collision_corners,
            T_W_G[:, :3],
            T_W_G[:, 3:],
        )
        support_error = torch.abs(gear_bottom_z - plate_top_z)
        return support_error <= support_z_threshold

    @staticmethod
    def _collision_corners(
        asset: RigidObject,
        device: str,
        collision_prim_name: str | None = None,
        enabled_only: bool = False,
    ) -> torch.Tensor:
        from pxr import Usd, UsdGeom, UsdPhysics

        root_prims = sim_utils.find_matching_prims(asset.cfg.prim_path)
        assert root_prims, f"{asset.cfg.prim_path} has no matching prims"
        root_prim = root_prims[0]
        rigid_prim = GearIsSupported._rigid_body_prim(root_prim)
        assert rigid_prim is not None, f"{asset.cfg.prim_path} has no rigid-body prim"

        bbox_cache = UsdGeom.BBoxCache(
            0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
            useExtentsHint=True,
        )
        corners = []
        for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            collision_prim = prim
            while collision_prim != root_prim and not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_prim = collision_prim.GetParent()
            if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if enabled_only and UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if collision_prim_name is not None and collision_prim.GetName() != collision_prim_name:
                continue
            local_box = bbox_cache.ComputeRelativeBound(prim, rigid_prim).ComputeAlignedBox()
            box_min = local_box.GetMin()
            box_max = local_box.GetMax()
            corners.extend(
                [x, y, z]
                for x in (box_min[0], box_max[0])
                for y in (box_min[1], box_max[1])
                for z in (box_min[2], box_max[2])
            )
        assert corners, f"{asset.cfg.prim_path} has no collision geometry"
        return torch.tensor(corners, device=device, dtype=torch.float32)

    @staticmethod
    def _rigid_body_prim(root_prim: Usd.Prim) -> Usd.Prim | None:
        from pxr import Usd, UsdPhysics

        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim
        return None

    @staticmethod
    def _world_collision_z_bounds(
        local_corners: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = root_pos.shape[0]
        num_corners = local_corners.shape[0]
        corners = local_corners.unsqueeze(0).expand(num_envs, num_corners, 3).reshape(-1, 3)
        quats = root_quat.unsqueeze(1).expand(num_envs, num_corners, 4).reshape(-1, 4)
        positions = root_pos.unsqueeze(1).expand(num_envs, num_corners, 3).reshape(-1, 3)
        world_z = (positions + math_utils.quat_apply(quats, corners))[:, 2].reshape(num_envs, num_corners)
        return world_z.min(dim=1).values, world_z.max(dim=1).values
