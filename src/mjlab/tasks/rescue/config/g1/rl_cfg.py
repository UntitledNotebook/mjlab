"""RL configuration for Unitree G1 rescue task."""

from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_rescue_ppo_runner_cfg():
  """Create PPO runner configuration for Unitree G1 foot-slip rescue training."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_rescue_foot_slip"
  return cfg
