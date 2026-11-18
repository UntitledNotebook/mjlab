"""Pelvis-frame observations and global tracking objectives used by ScaleTrack."""

from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .motion import ScaleTrackMotion


def motion(env: ManagerBasedRlEnv) -> ScaleTrackMotion:
  return cast(ScaleTrackMotion, env.command_manager.get_term("motion"))


def _tan_norm(quat: torch.Tensor) -> torch.Tensor:
  # The checkpoint encodes the rotated X and Z axes, concatenated in that order.
  rot = matrix_from_quat(quat)
  return torch.cat((rot[..., 0], rot[..., 2]), dim=-1)


def future_geometry(env: ManagerBasedRlEnv, privileged: bool = False) -> torch.Tensor:
  cmd = motion(env)
  frames = cmd.future_frames(cmd.critic_offsets if privileged else cmd.actor_offsets)
  ref = cmd.library.data
  pos = ref["body_pos_w"][frames] + env.scene.env_origins[:, None, None]
  quat = ref["body_quat_w"][frames]
  robot_pos = cmd.robot_body_pos_w[:, None]
  robot_quat = cmd.robot_body_quat_w[:, None].expand_as(quat)
  root_pos = (
    cmd.body_pos_w[:, None, :1] if cmd.cfg.local_tracking else robot_pos[:, :, :1]
  )
  root_quat = robot_quat[:, :, :1].expand_as(quat)
  root_inv = quat_inv(root_quat)
  rel_rot = quat_mul(quat, quat_conjugate(robot_quat))
  features = [
    quat_apply(root_inv, pos - root_pos),
    quat_apply(root_inv, pos - robot_pos),
    _tan_norm(quat_mul(root_inv, quat)),
    _tan_norm(quat_mul(quat_mul(root_inv, rel_rot), root_quat)),
  ]
  if privileged:
    robot = cmd.robot.data
    features.extend(
      (
        quat_apply(
          root_inv,
          ref["body_lin_vel_w"][frames]
          - robot.body_com_lin_vel_w[:, None, cmd.body_ids],
        ),
        quat_apply(
          root_inv,
          ref["body_ang_vel_w"][frames]
          - robot.body_com_ang_vel_w[:, None, cmd.body_ids],
        ),
      )
    )
  return torch.cat([feature.flatten(2) for feature in features], dim=-1)


def timestamps(env: ManagerBasedRlEnv, privileged: bool = False) -> torch.Tensor:
  cmd = motion(env)
  offsets = cmd.critic_offsets if privileged else cmd.actor_offsets
  # Timestamp is the requested frame offset, even when the pose clamps at clip end.
  return offsets.unsqueeze(-1).float()


def tracking_mode(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = motion(env)
  return cmd.mode[:, None].expand(-1, cmd.actor_offsets.shape[1], -1)


def critic_body_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = motion(env)
  pos = cmd.robot_body_pos_w
  quat = cmd.robot_body_quat_w
  heading_inv = quat_inv(yaw_quat(quat[:, 0])).unsqueeze(1).expand_as(quat)
  return torch.cat(
    (
      pos[:, 0, 2:3],
      quat_apply(heading_inv[:, 1:], pos[:, 1:] - pos[:, :1]).flatten(1),
      _tan_norm(quat_mul(heading_inv, quat)).flatten(1),
      quat_apply(
        heading_inv, cmd.robot.data.body_com_lin_vel_w[:, cmd.body_ids]
      ).flatten(1),
      quat_apply(
        heading_inv, cmd.robot.data.body_com_ang_vel_w[:, cmd.body_ids]
      ).flatten(1),
    ),
    dim=1,
  )


def global_height(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  cmd = motion(env)
  error = cmd.body_pos_w[:, 0, 2] - cmd.robot_body_pos_w[:, 0, 2]
  return torch.exp(-error.square() / std**2)


def global_position(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  cmd = motion(env)
  error = (cmd.body_pos_w - cmd.robot_body_pos_w).square().sum(-1)
  return torch.exp(-error.mean(-1) / std**2)


def global_rotation(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  cmd = motion(env)
  error = quat_error_magnitude(
    cmd.library.data["body_quat_w"][cmd.frame], cmd.robot_body_quat_w
  )
  return torch.exp(-error.square().mean(-1) / std**2)


def global_velocity(env: ManagerBasedRlEnv, std: float, angular: bool) -> torch.Tensor:
  cmd = motion(env)
  robot = cmd.robot.data
  if angular:
    error = (
      cmd.library.data["body_ang_vel_w"][cmd.frame]
      - robot.body_com_ang_vel_w[:, cmd.body_ids]
    )
  else:
    error = (
      cmd.library.data["body_lin_vel_w"][cmd.frame]
      - robot.body_com_lin_vel_w[:, cmd.body_ids]
    )
  return torch.exp(-error.square().sum(-1).mean(-1) / std**2)


def motion_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  cmd = motion(env)
  return cmd.time_steps >= cmd.last_frame


def tracking_failed(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
  cmd = motion(env)
  return (cmd.body_pos_w - cmd.robot_body_pos_w).norm(dim=-1).amax(-1) > threshold


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_hand_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  added_mass_range: tuple[float, float],
) -> None:
  """ScaleTrack adds mass at fixed hand geometry and rescales the inertia."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  hands = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
  nominal = env.sim.get_default_field("body_mass")[hands]
  added = torch.empty((len(env_ids), len(hands)), device=env.device)
  mass = nominal + added.uniform_(*added_mass_range)
  env.sim.model.body_mass[env_ids[:, None], hands] = mass
  inertia = env.sim.get_default_field("body_inertia")[hands]
  env.sim.model.body_inertia[env_ids[:, None], hands] = inertia * (
    mass / nominal
  ).unsqueeze(-1)
