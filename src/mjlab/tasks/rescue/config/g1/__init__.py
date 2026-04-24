"""Unitree G1 rescue task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_foot_slip_rescue_env_cfg
from .rl_cfg import unitree_g1_rescue_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Rescue-Flat-Unitree-G1-FootSlip",
  env_cfg=unitree_g1_foot_slip_rescue_env_cfg(),
  play_env_cfg=unitree_g1_foot_slip_rescue_env_cfg(play=True),
  rl_cfg=unitree_g1_rescue_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
