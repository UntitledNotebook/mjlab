"""ScaleTrack whole-body reference motion tracking."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import scaletrack_env_cfg
from .rl import ScaleTrackRunner, ScaleTrackRunnerCfg

register_mjlab_task(
  task_id="Mjlab-ScaleTrack-Flat-Unitree-G1",
  env_cfg=scaletrack_env_cfg(),
  play_env_cfg=scaletrack_env_cfg(play=True),
  rl_cfg=ScaleTrackRunnerCfg(),
  runner_cls=ScaleTrackRunner,
)
