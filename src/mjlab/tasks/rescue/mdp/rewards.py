"""Reward terms for saved-state rescue policy training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.rescue.mdp.events import get_rescue_context

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reference_joint_position_similarity_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  reset_event_name: str = "reset_from_failure_dataset",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward similarity to dataset reference joint positions."""
  raise NotImplementedError


def reference_joint_velocity_similarity_exp(
  env: "ManagerBasedRlEnv",
  std: float,
  reset_event_name: str = "reset_from_failure_dataset",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward similarity to dataset reference joint velocities."""
  raise NotImplementedError


def rescue_reference_command(
  env: "ManagerBasedRlEnv",
  reset_event_name: str = "reset_from_failure_dataset",
) -> torch.Tensor:
  """Expose the dataset reference command for diagnostics."""
  return get_rescue_context(env, reset_event_name).reference_command
