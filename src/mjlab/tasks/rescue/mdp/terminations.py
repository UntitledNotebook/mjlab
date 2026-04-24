"""Termination terms for rescue policy training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class StableForWindow:
  """Successful recovery when the robot remains stable for a fixed window."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    self._env = env
    self.stable_step_count = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    recovery_window_s: float,
    upright_threshold: float,
    root_height_range: tuple[float, float],
    max_body_ang_vel: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    """Return true when an environment has recovered for the configured window."""
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset stable-window counters."""
    self.stable_step_count[env_ids] = 0


def hard_unrecoverable(
  env: ManagerBasedRlEnv,
  max_tilt: float,
  root_height_range: tuple[float, float],
  grace_period_s: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate states that are outside the recoverable envelope."""
  raise NotImplementedError
