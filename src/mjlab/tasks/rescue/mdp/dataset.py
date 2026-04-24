"""Saved failure dataset contracts for rescue policy training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch


@dataclass(frozen=True)
class RescueFailureSampleBatch:
  """Batch of sampled rescue initial states and reference targets."""

  sample_ids: torch.Tensor
  root_pose: torch.Tensor
  root_velocity: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  reference_command: torch.Tensor
  reference_joint_pos: torch.Tensor
  reference_joint_vel: torch.Tensor


@dataclass(frozen=True)
class RescueFailureDataset:
  """Loaded saved failure dataset.

  Implementation is intentionally left for the P2 dataset milestone. The class
  exists now so environment configs, reset events, command terms, and rewards can
  depend on a stable interface.
  """

  path: Path

  REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
    "schema_version",
    "source_task_id",
    "joint_names",
    "root_pose",
    "root_velocity",
    "root_velocity_frame",
    "joint_pos",
    "joint_vel",
    "reference_command",
    "reference_joint_pos",
    "reference_joint_vel",
  )
  ROOT_VELOCITY_FRAME: ClassVar[str] = "world_root_link"

  @classmethod
  def load(cls, path: str | Path) -> "RescueFailureDataset":
    """Load and validate a rescue failure dataset."""
    raise NotImplementedError

  def validate_joint_names(self, joint_names: tuple[str, ...]) -> None:
    """Validate that dataset joint order matches the robot joint order."""
    raise NotImplementedError

  def sample(
    self,
    env_ids: torch.Tensor,
    *,
    device: str | torch.device,
  ) -> RescueFailureSampleBatch:
    """Sample reset states and references for the requested environments."""
    raise NotImplementedError
