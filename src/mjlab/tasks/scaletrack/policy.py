"""ScaleBFM humanoid transformer for mjlab's RSL-RL actor and critic."""

import math

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from .g1 import TRACKED_BODIES


class RMSNorm(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = x.norm(p=2, dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
    return self.scale * x / (rms + 1e-8)


class RotaryEncoding(nn.Module):
  inv_freq: torch.Tensor

  def __init__(self, dim: int):
    super().__init__()
    self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    angles = torch.outer(torch.arange(x.shape[1], device=x.device), self.inv_freq)
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class SwiGLU(nn.Module):
  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.w = nn.Linear(dim, hidden_dim, bias=False)
    self.v = nn.Linear(dim, hidden_dim, bias=False)
    self.output = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.output(F.silu(self.w(x)) * self.v(x))


class TransformerBlock(nn.Module):
  def __init__(self, dim: int, heads: int, ff_dim: int):
    super().__init__()
    self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
    self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
    self.feed_forward = SwiGLU(dim, ff_dim)
    self.rmsnorm1 = RMSNorm(dim)
    self.rmsnorm2 = RMSNorm(dim)
    self.rmsnorm3 = RMSNorm(dim)
    self.cond_norm = RMSNorm(dim)
    self.rope = RotaryEncoding(dim)

  def forward(self, x: torch.Tensor, task: torch.Tensor, mask: torch.Tensor):
    values = self.rmsnorm1(x)
    queries = self.rope(values)
    x = (
      x
      + self.self_attention(
        queries, queries, values, attn_mask=mask, need_weights=False
      )[0]
    )
    task = self.cond_norm(task)
    x = x + self.cross_attention(self.rmsnorm2(x), task, task, need_weights=False)[0]
    return x + self.feed_forward(self.rmsnorm3(x))


class HumanoidTransformer(nn.Module):
  attention_mask: torch.Tensor

  def __init__(
    self,
    prop_shape: tuple[int, int],
    action_shape: tuple[int, int],
    task_shape: tuple[int, int],
    output_dim: int,
    embedding_dim: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int,
    mask_targets: bool,
  ):
    super().__init__()
    self.prop_shape, self.action_shape, self.task_shape = (
      prop_shape,
      action_shape,
      task_shape,
    )
    self.input_sizes = (
      math.prod(prop_shape),
      math.prod(action_shape),
      math.prod(task_shape),
    )
    self.mask_targets = mask_targets
    self.num_bodies = len(TRACKED_BODIES)
    self.prop_projection = nn.Linear(prop_shape[1], embedding_dim)
    self.action_projection = nn.Linear(action_shape[1], embedding_dim)
    self.task_projection = nn.Linear(task_shape[1], embedding_dim)
    self.empty_embedding = nn.Parameter(torch.empty(1, 1, embedding_dim))
    nn.init.trunc_normal_(self.empty_embedding, std=0.02)
    self.transformer_blocks = nn.ModuleList(
      TransformerBlock(embedding_dim, num_heads, ff_dim) for _ in range(num_layers)
    )
    self.final_norm = RMSNorm(embedding_dim)
    self.projection_head = nn.Linear(embedding_dim, output_dim)
    nn.init.zeros_(self.projection_head.weight)
    nn.init.zeros_(self.projection_head.bias)
    context = 2 * prop_shape[0]
    mask = torch.zeros(context, context, dtype=torch.bool)
    mask[:-1, -1] = True
    self.register_buffer("attention_mask", mask, persistent=False)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    prop, action, task = obs.split(self.input_sizes, dim=-1)
    prop = self.prop_projection(prop.reshape(-1, *self.prop_shape))
    action = self.action_projection(action.reshape(-1, *self.action_shape)[:, 1:])
    task = task.reshape(-1, *self.task_shape)
    if self.mask_targets:
      mode = task[..., -self.num_bodies :]
      masks = [mode.repeat_interleave(width, dim=-1) for width in (3, 3, 6, 6)]
      geometry = task[..., : -(self.num_bodies + 1)] * torch.cat(masks, -1)
      task = torch.cat((geometry, task[..., -(self.num_bodies + 1) :]), -1)
    task = self.task_projection(task)
    # Chronological tokens: s0, a0, s1, a1, s2, query. The action in the
    # observation at t is the action that produced state t.
    action_query = torch.cat(
      (action, self.empty_embedding.expand(prop.shape[0], -1, -1)), 1
    )
    x = torch.stack((prop, action_query), dim=2).flatten(1, 2)
    for block in self.transformer_blocks:
      x = block(x, task, self.attention_mask)
    return self.projection_head(self.final_norm(x[:, -1]))


class ScaleTrackModel(MLPModel):
  """Use RSL-RL's distribution, PPO and exporters with a transformer head."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    embedding_dim: int = 384,
    num_heads: int = 6,
    ff_dim: int = 384,
    num_layers: int = 6,
    init_std: float | None = None,
  ):
    # MLPModel.__init__ builds a dense MLP and rejects sequence observations.
    nn.Module.__init__(self)
    self.obs_groups = obs_groups[obs_set]
    prop, action, task = (obs[key] for key in self.obs_groups)
    self.obs_dim = sum(t.shape[1] * t.shape[2] for t in (prop, action, task))
    self.obs_normalization = False
    self.obs_normalizer = nn.Identity()
    self.distribution = (
      GaussianDistribution(output_dim, init_std=init_std)
      if init_std is not None
      else None
    )
    self.add_module(
      "mlp",
      HumanoidTransformer(
        (prop.shape[1], prop.shape[2]),
        (action.shape[1], action.shape[2]),
        (task.shape[1], task.shape[2]),
        output_dim,
        embedding_dim,
        num_heads,
        ff_dim,
        num_layers,
        mask_targets=obs_set == "actor",
      ),
    )

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    return torch.cat([obs[key].flatten(1) for key in self.obs_groups], dim=-1)
