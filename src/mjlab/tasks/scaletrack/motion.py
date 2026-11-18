"""ScaleTrack's 50 Hz motion library, reference state and tracking modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import FileMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from .g1 import BODY_NAMES, JOINT_NAMES, TRACKED_BODIES, TRACKING_MODES

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _motion_entries(path: Path) -> dict[str, tuple[Path, float]]:
  if path.suffix == ".npz":
    return {path.stem: (path, 1.0)}
  if path.suffix not in (".yaml", ".yml"):
    raise ValueError(f"Expected a ScaleTrack NPZ or YAML manifest, got {path}")
  with path.open() as stream:
    manifest = yaml.safe_load(stream)
  if not isinstance(manifest, dict) or not manifest:
    raise ValueError(f"{path}: expected a nonempty mapping of motion names to clips")
  entries = {}
  parents = path.resolve().parents
  for name, entry in manifest.items():
    if isinstance(entry, dict):
      if entry.keys() - {"path", "weight"}:
        raise ValueError(f"{path}: clip {name!r} accepts only 'path' and 'weight'")
      filename, weight = entry.get("path"), entry.get("weight", 1.0)
    else:
      filename, weight = entry, 1.0
    if not isinstance(filename, str):
      raise ValueError(f"{path}: expected an NPZ path for clip {name!r}")
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
      raise ValueError(f"{path}: weight for clip {name!r} must be finite and >= 0")
    # ScaleTrack's example manifest stores paths relative to its repository root.
    # User-generated manifests can also be relative to the YAML or the working dir.
    candidates = (parent / filename for parent in parents)
    clip_path = next((p for p in candidates if p.is_file()), Path(filename))
    entries[name] = (clip_path, weight)
  return entries


class MotionLibrary:
  def __init__(self, path: str, device: str):
    entries = _motion_entries(Path(path))
    self.names = list(entries)
    weights = [weight for _, weight in entries.values()]
    max_weight = max(weights)
    if max_weight == 0:
      raise ValueError(f"{path}: at least one clip must have a positive weight")
    self._sampling_weights: torch.Tensor | None = None
    if min(weights) < max_weight:
      self._sampling_weights = torch.tensor(
        [w / max_weight for w in weights], dtype=torch.float64, device=device
      )
    body_ids = [BODY_NAMES.index(name) for name in TRACKED_BODIES]
    fields = {
      "joint_pos": (len(JOINT_NAMES),),
      "joint_vel": (len(JOINT_NAMES),),
      "body_pos_w": (len(BODY_NAMES), 3),
      "body_quat_w": (len(BODY_NAMES), 4),
      "body_lin_vel_w": (len(BODY_NAMES), 3),
      "body_ang_vel_w": (len(BODY_NAMES), 3),
    }
    clips: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    lengths = []
    for p, _ in entries.values():
      with np.load(p) as data:
        length = len(data["joint_pos"])
        if data["fps"].item() != 50 or length == 0:
          raise ValueError(f"{p} must contain a nonempty 50 Hz ScaleTrack motion")
        for key, shape in fields.items():
          values = data[key]
          if values.shape != (length, *shape):
            raise ValueError(
              f"{p}: expected {key} shape {(length, *shape)}, got {values.shape}"
            )
          if key.startswith("body_"):
            values = values[:, body_ids]
          clips[key].append(values)
        lengths.append(length)
    self.data = {
      key: torch.as_tensor(np.concatenate(values), dtype=torch.float32, device=device)
      for key, values in clips.items()
    }
    self.lengths = torch.tensor(lengths, device=device)
    self.starts = self.lengths.cumsum(dim=0) - self.lengths

  def sample_clip_ids(self, count: int) -> torch.Tensor:
    """Draw training clips with replacement."""
    # Keep seeded uniform draws. Early resets can also request zero new clips.
    if self._sampling_weights is None or count == 0:
      return torch.randint(len(self.names), (count,), device=self.lengths.device)
    return torch.multinomial(self._sampling_weights, count, replacement=True)


@dataclass(kw_only=True)
class ScaleTrackMotionCfg(FileMotionCommandCfg):
  play: bool = False
  mode_index: int | None = None
  """Fixed mode. Training samples one of the eight modes per episode."""
  local_tracking: bool = False
  far_frame_range: tuple[int, int] = (5, 33)
  """Interval for the sixth actor frame, in steps (upper bound exclusive)."""

  def build(self, env: ManagerBasedRlEnv) -> ScaleTrackMotion:
    return ScaleTrackMotion(self, env)


class ScaleTrackMotion(CommandTerm):
  cfg: ScaleTrackMotionCfg

  def __init__(self, cfg: ScaleTrackMotionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene["robot"]
    self.library = MotionLibrary(cfg.motion_file, self.device)
    self.joint_ids = torch.tensor(
      self.robot.find_joints(JOINT_NAMES, preserve_order=True)[0], device=self.device
    )
    self.body_ids = torch.tensor(
      self.robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0], device=self.device
    )
    self.mode_table = torch.zeros(
      len(TRACKING_MODES), len(TRACKED_BODIES), device=self.device
    )
    for i, indexes in enumerate(TRACKING_MODES):
      self.mode_table[i, list(indexes)] = 1
    if cfg.mode_index is not None and not 0 <= cfg.mode_index < len(TRACKING_MODES):
      raise ValueError("ScaleTrack mode_index must be between 0 and 7")
    if cfg.local_tracking and (
      cfg.mode_index is None or 0 not in TRACKING_MODES[cfg.mode_index]
    ):
      raise ValueError("Local tracking requires a fixed mode that includes the pelvis")
    self.mode = self.mode_table[-1].repeat(self.num_envs, 1)
    self.motion_ids = torch.arange(self.num_envs, device=self.device) % len(
      self.library.names
    )
    self.time_steps = torch.zeros_like(self.motion_ids)
    self.started = torch.zeros_like(self.motion_ids, dtype=torch.bool)
    self.actor_offsets = torch.tensor([0, 1, 2, 3, 4, 5], device=self.device).repeat(
      self.num_envs, 1
    )
    self.critic_offsets = torch.tensor(
      [0, 1, 2, 4, 8, 16, 32], device=self.device
    ).expand(self.num_envs, -1)
    self.metrics["error_body_pos_g"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def frame(self) -> torch.Tensor:
    return self.library.starts[self.motion_ids] + self.time_steps

  @property
  def last_frame(self) -> torch.Tensor:
    return self.library.lengths[self.motion_ids] - 1

  def future_frames(self, offsets: torch.Tensor) -> torch.Tensor:
    local = torch.minimum(self.time_steps[:, None] + offsets, self.last_frame[:, None])
    return self.library.starts[self.motion_ids, None] + local

  @property
  def command(self) -> torch.Tensor:
    return self.library.data["joint_pos"][self.frame]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.library.data["body_pos_w"][self.frame] + self._env.scene.env_origins[:, None]
    )

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_ids]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_ids]

  def _update_metrics(self) -> None:
    self.metrics["error_body_pos_g"] = (
      (self.body_pos_w - self.robot_body_pos_w).norm(dim=-1).mean(dim=-1)
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    ended = self.started[env_ids] & (
      self.time_steps[env_ids] >= self.last_frame[env_ids]
    )
    if self.cfg.play:
      # Spread viewers across a manifest, then advance each to the next completed clip.
      self.motion_ids[env_ids] = (self.motion_ids[env_ids] + ended) % len(
        self.library.names
      )
      self.time_steps[env_ids] = 0
    else:
      # After a failure, resume one frame later in the same motion. A fresh
      # episode with a new clip samples its starting phase uniformly.
      new_clip = env_ids[ended | ~self.started[env_ids]]
      self.time_steps[env_ids] += 1
      self.motion_ids[new_clip] = self.library.sample_clip_ids(len(new_clip))
      self.time_steps[new_clip] = (
        torch.rand(len(new_clip), device=self.device) * self.last_frame[new_clip]
      ).long()
    self.started[env_ids] = True
    self.actor_offsets[env_ids, -1] = torch.randint(
      *self.cfg.far_frame_range, (count,), device=self.device
    )
    mode_ids = self.cfg.mode_index
    if mode_ids is None:
      mode_ids = torch.randint(len(TRACKING_MODES), (count,), device=self.device)
    self.mode[env_ids] = self.mode_table[mode_ids]

    frame = self.frame[env_ids]
    data = self.library.data
    pos = data["body_pos_w"][frame, 0] + self._env.scene.env_origins[env_ids]
    quat = data["body_quat_w"][frame, 0]
    vel = torch.cat(
      (data["body_lin_vel_w"][frame, 0], data["body_ang_vel_w"][frame, 0]), dim=-1
    )
    joints = data["joint_pos"][frame]
    if not self.cfg.play:
      jitter = torch.empty(count, 6, device=self.device).uniform_(-1, 1)
      jitter *= jitter.new_tensor((0.05, 0.05, 0.01, 0.1, 0.1, 0.2))
      pos += jitter[:, :3]
      quat = quat_mul(quat_from_euler_xyz(*jitter[:, 3:].unbind(-1)), quat)
      vel += torch.empty_like(vel).uniform_(-1, 1) * vel.new_tensor(
        [high for _, high in VELOCITY_RANGE.values()]
      )
      limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
      joints = (joints + torch.empty_like(joints).uniform_(-0.1, 0.1)).clamp(
        limits[..., 0], limits[..., 1]
      )
    self.robot.write_joint_state_to_sim(
      joints, data["joint_vel"][frame], joint_ids=self.joint_ids, env_ids=env_ids
    )
    self.robot.write_root_link_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids)
    self.robot.write_root_com_velocity_to_sim(vel, env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Isaac's explicit reset observes frame zero. Its post-step command advances
    # one frame, including for automatically reset environments.
    if env_ids is None:
      self.time_steps = torch.minimum(self.time_steps + 1, self.last_frame)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    positions = self.body_pos_w
    if self.cfg.local_tracking:
      positions = positions + (self.robot_body_pos_w[:, :1] - positions[:, :1])
    for i in visualizer.get_env_indices(self.num_envs):
      points = positions[i].cpu().numpy()
      for j in self.mode[i].nonzero().flatten().tolist():
        visualizer.add_sphere(
          center=points[j],
          radius=0.025,
          color=(1.0, 0.45, 0.1, 0.8),
          label=f"reference_{i}_{TRACKED_BODIES[j]}",
        )
