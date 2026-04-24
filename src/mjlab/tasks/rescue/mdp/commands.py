"""Command terms for rescue policy training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.rescue.mdp.events import get_rescue_context

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RescueReferenceCommand(CommandTerm):
  """Frozen command sourced from the saved failure dataset."""

  cfg: RescueReferenceCommandCfg

  def __init__(self, cfg: "RescueReferenceCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    """Update command metrics."""
    raise NotImplementedError

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Freeze command to the dataset reference sampled at reset."""
    context = get_rescue_context(self._env, self.cfg.reset_event_name)
    self._command[env_ids] = context.reference_command[env_ids]

  def _update_command(self) -> None:
    """Keep the dataset reference command frozen during the episode."""
    return


@dataclass(kw_only=True)
class RescueReferenceCommandCfg(CommandTermCfg):
  """Configuration for dataset-backed rescue velocity commands."""

  reset_event_name: str = "reset_from_failure_dataset"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False

  def build(self, env: ManagerBasedRlEnv) -> RescueReferenceCommand:
    """Build the command term."""
    return RescueReferenceCommand(self, env)
