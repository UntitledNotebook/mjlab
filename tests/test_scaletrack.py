"""ScaleTrack data layout and pretrained controller regressions."""

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.scaletrack.env_cfg import scaletrack_env_cfg
from mjlab.tasks.scaletrack.g1 import JOINT_NAMES, TRACKED_BODIES
from mjlab.tasks.scaletrack.motion import MotionLibrary, ScaleTrackMotionCfg
from mjlab.tasks.scaletrack.policy import HumanoidTransformer
from mjlab.tasks.scaletrack.rl import ScaleTrackRunner, ScaleTrackRunnerCfg


@pytest.fixture
def motion_folder(tmp_path: Path) -> Path:
  folder = tmp_path / "source" / "example"
  folder.mkdir(parents=True)
  joints = np.arange(29, dtype=np.float32)[None]
  bodies = np.broadcast_to(np.arange(30)[None, :, None], (1, 30, 3))
  for name, offset in (("a", 0), ("b", 100)):
    np.savez(
      folder / f"{name}.npz",
      fps=50,
      joint_pos=joints + offset,
      joint_vel=joints,
      body_pos_w=bodies + offset,
      body_quat_w=np.zeros((1, 30, 4)),
      body_lin_vel_w=bodies,
      body_ang_vel_w=bodies,
    )
  return folder


def test_motion_library_reads_isaac_body_order(motion_folder: Path) -> None:
  manifest = motion_folder / "motions.yaml"
  manifest.write_text("first: source/example/a.npz\nsecond: b.npz\n")
  lib = MotionLibrary(str(manifest), "cpu")
  assert lib.names == ["first", "second"]
  assert lib.starts.tolist() == [0, 1]
  assert lib.data["joint_pos"][1].tolist() == list(range(100, 129))
  expected = torch.tensor([0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29])
  torch.testing.assert_close(lib.data["body_pos_w"][0, :, 0], expected.float())
  torch.testing.assert_close(lib.data["body_pos_w"][1, :, 0], expected.float() + 100)


def test_weighted_clip_draws(motion_folder: Path) -> None:
  manifest = motion_folder / "motions.yaml"
  manifest.write_text(
    "first: a.npz\n"
    "second: {path: source/example/b.npz, weight: 0.5}\n"
    "third: {path: a.npz}\n"
    "disabled: {path: b.npz, weight: 0}\n"
  )
  lib = MotionLibrary(str(manifest), "cpu")
  with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(42)
    draws = lib.sample_clip_ids(10_000)
  counts = draws.bincount(minlength=4)
  assert counts[3].item() == 0
  assert (counts[:3] / len(draws)).tolist() == pytest.approx([0.4, 0.2, 0.4], abs=0.02)
  assert lib.sample_clip_ids(0).shape == (0,)  # Early resets keep their clip.


def test_unweighted_clip_draws(motion_folder: Path) -> None:
  manifest = motion_folder / "motions.yaml"
  manifest.write_text("first: a.npz\nsecond: b.npz\n")
  lib = MotionLibrary(str(manifest), "cpu")
  with torch.random.fork_rng(devices=[]):
    expected = torch.randint(2, (32,))
  torch.testing.assert_close(lib.sample_clip_ids(32), expected)
  single = MotionLibrary(str(motion_folder / "a.npz"), "cpu")
  assert single.sample_clip_ids(3).tolist() == [0, 0, 0]


@pytest.mark.parametrize(
  ("weight", "error"),
  [
    ("-1", "weight for clip 'first'"),
    (".nan", "weight for clip 'first'"),
    (".inf", "weight for clip 'first'"),
    ("high", "weight for clip 'first'"),
    ("0", "at least one clip must have a positive weight"),
  ],
)
def test_invalid_clip_weight(motion_folder: Path, weight: str, error: str) -> None:
  manifest = motion_folder / "motions.yaml"
  manifest.write_text(f"first: {{path: a.npz, weight: {weight}}}\n")
  with pytest.raises(ValueError, match=error):
    MotionLibrary(str(manifest), "cpu")


def test_actor_masks_untracked_links_and_uses_two_past_actions() -> None:
  model = HumanoidTransformer(
    (3, 64), (3, 29), (6, 267), 29, 32, 2, 32, 1, mask_targets=True
  ).eval()
  torch.nn.init.normal_(model.projection_head.weight)
  prop = torch.randn(1, 3, 64)
  action = torch.randn(1, 3, 29)
  task = torch.randn(1, 6, 267)
  task[..., -14:] = 0
  task[..., -14] = 1  # Pelvis-only tracking.

  def predict():
    return model(torch.cat([x.flatten(1) for x in (prop, action, task)], -1))

  with torch.inference_mode():
    original = predict()
    action[:, 0] += 10
    for start, width in ((0, 3), (42, 3), (84, 6), (168, 6)):
      task[..., start + width : start + 14 * width] += 10
    torch.testing.assert_close(predict(), original)
    action[:, 1] += 10
    assert not torch.allclose(predict(), original)


@pytest.mark.slow
@pytest.mark.skipif(
  "SCALETRACK_ROOT" not in os.environ,
  reason="requires the published ScaleTrack XL assets",
)
def test_xl_checkpoint_on_original_motions() -> None:
  """Set SCALETRACK_ROOT to the ScaleTrack directory containing logs/ and source/."""
  root = Path(os.environ["SCALETRACK_ROOT"])
  cfg = scaletrack_env_cfg(play=True)
  cfg.scene.num_envs = 3
  cmd_cfg = cfg.commands["motion"]
  assert isinstance(cmd_cfg, ScaleTrackMotionCfg)
  cmd_cfg.motion_file = str(root / "source/scaletrack/data/example/example.yaml")
  cmd_cfg.far_frame_range = (16, 17)
  cfg.seed = 42
  device = get_test_device()
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device))
  try:
    checkpoint = (
      root
      / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl"
      / "model_22200.pt"
    )
    runner = ScaleTrackRunner(env, asdict(ScaleTrackRunnerCfg()), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location="cpu")
    policy = runner.get_inference_policy(device)
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    body_ids = robot.find_bodies(TRACKED_BODIES, preserve_order=True)[0]
    files = ("weishuai-136", "weishuai-205", "230711-panic_run_away_180_R_001__A424")
    reference = []
    for name in files:
      with np.load(root / f"source/scaletrack/data/example/{name}.npz") as data:
        # Indices in the original packaged motions, independent of the loader.
        ids = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
        reference.append(torch.tensor(data["body_pos_w"][:, ids], device=device))
    errors = [[] for _ in files]
    model = env.unwrapped.sim.mj_model
    ctrl_for_joint = {
      model.joint(int(joint)).name: i
      for i, (joint, _) in enumerate(model.actuator_trnid)
    }
    ctrl_ids = [ctrl_for_joint[f"robot/{name}"] for name in JOINT_NAMES]
    with torch.inference_mode():
      for frame in range(1, max(len(ref) for ref in reference)):
        obs, _, dones, _ = env.step(policy(obs))
        if frame == 1:
          # Actual Isaac Lab joint targets for weishuai-136, frame 0, far frame 16.
          golden = torch.tensor(
            [
              -0.353927,
              -0.158161,
              -0.055335,
              0.024630,
              -0.078840,
              0.058838,
              -0.275444,
              -0.319844,
              0.030914,
              0.572979,
              0.348509,
              -0.165607,
              -0.142752,
              0.069460,
              -0.012701,
              0.763137,
              -0.764563,
              -0.024246,
              -0.014049,
              0.665330,
              -0.903532,
              0.802822,
              0.738004,
              -0.192135,
              0.094154,
              -0.083314,
              0.011471,
              0.066393,
              0.058241,
            ],
            device=device,
          )
          torch.testing.assert_close(
            env.unwrapped.sim.data.ctrl[0, ctrl_ids], golden, atol=2e-4, rtol=0
          )
        pos = (
          robot.data.body_link_pos_w[:, body_ids]
          - env.unwrapped.scene.env_origins[:, None]
        )
        for i, ref in enumerate(reference):
          if frame < len(ref):
            assert not dones[i]
            errors[i].append((pos[i] - ref[frame]).norm(dim=-1))
    # Original Isaac Lab mean link errors, in metres, on the same nominal robot.
    for name, samples, baseline in zip(
      files, errors, (0.0539, 0.0451, 0.0967), strict=True
    ):
      distances = torch.stack(samples)
      print(f"{name}: mean link error {distances.mean().item():.4f} m")
      assert distances.mean() < baseline + 0.03
      assert distances.max() < 0.5  # Original task's tracking-failure threshold.
  finally:
    env.close()
