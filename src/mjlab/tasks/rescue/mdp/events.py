"""Reset events and episode context for saved-state rescue training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.rescue.mdp.dataset import RescueFailureDataset

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@dataclass
class RescueEpisodeContext:
  """Per-environment references sampled by the rescue dataset reset event."""

  sample_ids: torch.Tensor
  reference_command: torch.Tensor
  reference_joint_pos: torch.Tensor
  reference_joint_vel: torch.Tensor


class ResetFromFailureDataset:
  """Reset robot state from a saved rescue failure dataset."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    self._env = env
    self.dataset_path = Path(cfg.params["dataset_path"])
    self.asset_cfg: SceneEntityCfg = cfg.params.get(
      "asset_cfg", SceneEntityCfg("robot")
    )
    self.context: RescueEpisodeContext | None = None
    self.dataset: RescueFailureDataset | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    dataset_path: str | Path,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    """Sample a saved failure state and write it into the simulation."""
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset any event-local bookkeeping after environment resets."""
    del env_ids


def get_rescue_context(
  env: ManagerBasedRlEnv,
  event_name: str = "reset_from_failure_dataset",
) -> RescueEpisodeContext:
  """Return typed reset-time rescue references from the dataset reset event."""
  term_cfg = env.event_manager.get_term_cfg(event_name)
  if not isinstance(term_cfg.func, ResetFromFailureDataset):
    raise TypeError(
      f"Event '{event_name}' must be ResetFromFailureDataset, "
      f"got {type(term_cfg.func).__name__}."
    )
  if term_cfg.func.context is None:
    raise RuntimeError(
      f"Event '{event_name}' has no rescue episode context. "
      "Call env.reset() before reading rescue references."
    )
  return term_cfg.func.context
