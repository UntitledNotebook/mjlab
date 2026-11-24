"""ScaleTrack XL policy configuration and checkpoint import."""

from dataclasses import dataclass, field

from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlPpoAlgorithmCfg

from .mdp import motion


@dataclass
class TransformerCfg:
  class_name: str = "mjlab.tasks.scaletrack.policy:ScaleTrackModel"
  embedding_dim: int = 384
  num_heads: int = 6
  ff_dim: int = 384
  num_layers: int = 6
  init_std: float | None = None


@dataclass
class ScaleTrackRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  experiment_name: str = "g1_bfm_tracking_exp"
  num_steps_per_env: int = 64
  max_iterations: int = 50000
  save_interval: int = 1000
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("actor", "action", "actor_task"),
      "critic": ("critic", "action", "critic_task"),
    }
  )
  actor: TransformerCfg = field(default_factory=lambda: TransformerCfg(init_std=0.8))
  critic: TransformerCfg = field(default_factory=TransformerCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      num_learning_epochs=2,
      num_mini_batches=32,
      learning_rate=2e-5,
    )
  )


class ScaleTrackRunner(MjlabOnPolicyRunner):
  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    motion(self.env.unwrapped).schedule_pool_refresh(self.cfg["num_steps_per_env"])
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def _load_checkpoint(
    self, loaded_dict: dict, load_cfg: dict | None, strict: bool
  ) -> dict:
    if "model_state_dict" in loaded_dict:
      weights = loaded_dict.pop("model_state_dict")
      for network in ("actor", "critic"):
        state = {
          "mlp." + key.removeprefix(network + "."): value
          for key, value in weights.items()
          if key.startswith(network + ".")
        }
        for suffix in ("weight", "bias"):
          state["mlp.task_projection." + suffix] = weights[
            f"{network}_task_embedder.task_projection.{suffix}"
          ]
        loaded_dict[network + "_state_dict"] = state
      loaded_dict["actor_state_dict"]["distribution.std_param"] = weights["std"]
      # ScaleBFM trains with two optimizers. Its weights can initialize native PPO.
      if load_cfg is None:
        load_cfg = {"actor": True, "critic": True, "iteration": True}
        print("Imported ScaleBFM weights. Native PPO starts with a fresh optimizer.")
      elif load_cfg.get("optimizer"):
        raise ValueError("ScaleBFM's separate optimizers require a weights-only import")
    return super()._load_checkpoint(loaded_dict, load_cfg, strict)
