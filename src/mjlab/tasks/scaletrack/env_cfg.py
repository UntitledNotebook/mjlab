"""ScaleBFM's flat-ground G1 tracking task."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg

from . import mdp
from .g1 import ACTION_SCALE, JOINT_NAMES, robot_cfg
from .motion import ScaleTrackMotionCfg


def scaletrack_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  cfg.sim.njmax = 512
  cfg.scene.entities = {"robot": robot_cfg()}
  cfg.viewer.body_name = "pelvis"
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=JOINT_NAMES,
    preserve_order=True,
    scale=ACTION_SCALE,
    use_default_offset=True,
  )
  cfg.commands["motion"] = ScaleTrackMotionCfg(
    motion_file="",
    play=play,
    mode_index=7 if play else None,
    resampling_time_range=(1e9, 1e9),
    debug_vis=True,
  )
  joints = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  joint_pos = ObservationTermCfg(
    func=common.joint_pos_rel, params={"asset_cfg": joints, "biased": True}
  )
  joint_vel = ObservationTermCfg(
    func=common.joint_vel_rel, params={"asset_cfg": joints}, scale=0.05
  )
  prop_terms = {
    "gravity": ObservationTermCfg(func=common.projected_gravity),
    "angular_velocity": ObservationTermCfg(func=common.base_ang_vel),
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
  }
  for term, amplitude in zip(prop_terms.values(), (0.05, 0.2, 0.01, 0.5), strict=True):
    term.noise = UniformNoiseCfg(n_min=-amplitude, n_max=amplitude)
  history = 3
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=prop_terms,
      history_length=history,
      flatten_history_dim=False,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "body_state": ObservationTermCfg(func=mdp.critic_body_state),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
      },
      history_length=history,
      flatten_history_dim=False,
    ),
    "action": ObservationGroupCfg(
      terms={"action": ObservationTermCfg(func=common.last_action)},
      history_length=history,
      flatten_history_dim=False,
    ),
    "actor_task": ObservationGroupCfg(
      terms={
        "geometry": ObservationTermCfg(
          func=mdp.future_geometry, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        ),
        "timestamp": ObservationTermCfg(func=mdp.timestamps),
        "mode": ObservationTermCfg(func=mdp.tracking_mode),
      },
      enable_corruption=not play,
    ),
    "critic_task": ObservationGroupCfg(
      terms={
        "geometry": ObservationTermCfg(
          func=mdp.future_geometry, params={"privileged": True}
        ),
        "timestamp": ObservationTermCfg(
          func=mdp.timestamps, params={"privileged": True}
        ),
      }
    ),
  }
  cfg.rewards = {
    "motion_body_height": RewardTermCfg(
      func=mdp.global_height, weight=0.5, params={"std": 0.3}
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.global_position, weight=1.0, params={"std": 0.3}
    ),
    "motion_body_rot": RewardTermCfg(
      func=mdp.global_rotation, weight=1.0, params={"std": 0.4}
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.global_velocity, weight=1.0, params={"std": 1.0, "angular": False}
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.global_velocity, weight=1.0, params={"std": 3.14, "angular": True}
    ),
    "action_rate_l2": cfg.rewards["action_rate_l2"],
    "joint_limit": cfg.rewards["joint_limit"],
    "survival": RewardTermCfg(func=common.is_alive, weight=1.0),
  }
  cfg.terminations = {
    "motion_end": TerminationTermCfg(func=mdp.motion_end, time_out=True),
  }
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.events = {}
  else:
    cfg.terminations["time_out"] = TerminationTermCfg(
      func=common.time_out, time_out=True
    )
    cfg.terminations["body_pos"] = TerminationTermCfg(
      func=mdp.tracking_failed, params={"threshold": 0.5}
    )
    cfg.terminations["nan_term"] = TerminationTermCfg(
      func=common.nan_detection, time_out=False
    )
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
      "robot", body_names="torso_link"
    )
    friction = cfg.events.pop("foot_friction")
    friction.params["asset_cfg"] = SceneEntityCfg("robot", geom_names=".*_collision")
    friction.params["shared_random"] = False
    cfg.events["friction"] = friction
    cfg.events["hand_mass"] = EventTermCfg(
      func=mdp.randomize_hand_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_wrist_yaw_link"),
        "added_mass_range": (0.0, 1.0),
      },
    )
  return cfg
