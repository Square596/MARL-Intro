#!/usr/bin/env python3
"""Train TorchRL PPO on the custom Jumanji Cleaner seminar env."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp

# Initialize JAX before importing Torch so PyTorch's bundled cuDNN does not
# shadow the CUDA libraries used by the JAX CUDA plugin.
_JAX_DEVICES = jax.devices()
jnp.asarray(0, dtype=jnp.int8).block_until_ready()

import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential, set_interaction_type
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs import JumanjiWrapper, TransformedEnv
from torchrl.envs.transforms import RewardSum
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from custom_cleaner import CustomCleaner, FixedObstacleGenerator

DIRTY = 0
CLEAN = 1
WALL = 2


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    height: int = 8
    width: int = 8
    num_agents: int = 3
    num_envs: int = 64
    n_iters: int = 300
    frames_per_batch: int = 4096
    ppo_epochs: int = 4
    minibatch_size: int = 512
    hidden_size: int = 256
    depth: int = 2
    encoder: Literal["cnn", "mlp"] = "cnn"
    obs_clean_channel: bool = False
    obs_other_channel: bool = False
    penalty_per_timestep: float = 0.1
    lr: float = 5.0e-4
    gamma: float = 0.995
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 1.0e-4
    critic_coeff: float = 1.0
    max_grad_norm: float = 1.0
    diversity_coeff: float = 0.0
    diversity_pairs: Literal["all", "adjacent", "sampled"] = "all"
    diversity_source_grad: bool = False
    diversity_sampled_pairs: int = 8
    diversity_bonus_max: float = 3.0
    policy_centralized: bool = False
    critic_centralized: bool = True
    share_policy_params: bool = False
    share_critic_params: bool = True
    ac_architecture: Literal["separate", "shared-trunk"] = "separate"
    output_dir: str = ".artifacts/cleaner_ppo"
    eval_rollout_length: int = 64
    eval_frequency: int = 1
    early_stop_mode_success: float = 0.95
    early_stop_sample_success: float = 0.85
    render_frequency: int = 0
    check_specs: bool = True


class PerAgentMaskedCategorical(MaskedCategorical):
    """Masked categorical policy that keeps one log-prob per agent."""

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value)

    def entropy(self) -> torch.Tensor:
        return super().entropy()


class CleanerObservationBuilder(nn.Module):
    """Create flat and spatial multi-agent observations from Jumanji Cleaner keys."""

    def __init__(
        self,
        height: int,
        width: int,
        num_agents: int,
        obs_clean_channel: bool = False,
        obs_other_channel: bool = False,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_agents = num_agents
        self.obs_clean_channel = obs_clean_channel
        self.obs_other_channel = obs_other_channel
        self.actor_spatial_channels = 4 + int(obs_clean_channel) + int(obs_other_channel)
        self.critic_spatial_channels = 3 + int(obs_clean_channel)
        self.local_obs_dim = height * width + 1 + 2 + 4
        self.global_obs_dim = height * width + num_agents * 2 + 1

    def forward(
        self,
        grid: torch.Tensor,
        agents_locations: torch.Tensor,
        step_count: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leading_shape = grid.shape[:-2]
        normalizer = torch.tensor(
            [max(self.height - 1, 1), max(self.width - 1, 1)],
            device=grid.device,
            dtype=torch.float32,
        )
        grid_features = grid.float().flatten(-2, -1) / float(WALL)
        own_locations = agents_locations.float() / normalizer
        all_locations = own_locations.flatten(-2, -1)
        step = (step_count.float() / float(self.height * self.width)).reshape(*leading_shape, 1)

        global_observation = torch.cat([grid_features, all_locations, step], dim=-1)
        shared_local = torch.cat([grid_features, step], dim=-1)
        shared_local = shared_local.unsqueeze(-2).expand(
            *leading_shape,
            self.num_agents,
            shared_local.shape[-1],
        )
        agent_observation = torch.cat(
            [shared_local, own_locations, action_mask.float()],
            dim=-1,
        )

        dirty = (grid == DIRTY).float()
        clean = (grid == CLEAN).float()
        wall = (grid == WALL).float()
        agent_maps = self._agent_position_maps(agents_locations, grid.device, grid.dtype)
        own_maps = agent_maps
        all_agents_map = agent_maps.sum(dim=-3, keepdim=True)
        other_maps = all_agents_map - own_maps

        actor_base_channels = [dirty, wall]
        if self.obs_clean_channel:
            actor_base_channels.append(clean)
        actor_base = torch.stack(actor_base_channels, dim=-3)
        actor_base_per_agent = actor_base.unsqueeze(-4).expand(
            *leading_shape,
            self.num_agents,
            len(actor_base_channels),
            self.height,
            self.width,
        )
        actor_agent_channels = [own_maps.unsqueeze(-3), all_agents_map.expand_as(own_maps).unsqueeze(-3)]
        if self.obs_other_channel:
            actor_agent_channels.append(other_maps.unsqueeze(-3))
        spatial_observation = torch.cat([actor_base_per_agent, *actor_agent_channels], dim=-3)

        critic_channels = [dirty, wall, all_agents_map.squeeze(-3)]
        if self.obs_clean_channel:
            critic_channels.append(clean)
        global_spatial_observation = torch.stack(critic_channels, dim=-3)
        return (
            agent_observation,
            spatial_observation,
            action_mask.bool(),
            global_observation,
            global_spatial_observation,
            step,
        )

    def _agent_position_maps(
        self,
        agents_locations: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        flat_shape = agents_locations.shape[:-2]
        flat_locations = agents_locations.reshape(-1, self.num_agents, 2).long()
        rows = flat_locations[..., 0].clamp(0, self.height - 1)
        cols = flat_locations[..., 1].clamp(0, self.width - 1)
        flat_indices = rows * self.width + cols
        maps = torch.nn.functional.one_hot(flat_indices, num_classes=self.height * self.width)
        maps = maps.to(device=device, dtype=torch.float32)
        return maps.reshape(*flat_shape, self.num_agents, self.height, self.width)


class SpatialCNNEncoder(nn.Module):
    """Small spatial encoder for Cleaner grids."""

    def __init__(self, in_channels: int, height: int, width: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        leading_shape = observation.shape[:-3]
        flat = observation.reshape(-1, *observation.shape[-3:])
        encoded = self.net(flat)
        return encoded.reshape(*leading_shape, encoded.shape[-1])


class CleanerMLPPolicyNet(nn.Module):
    """Legacy flat-observation multi-agent policy."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        obs_dim = config.height * config.width + 1 + 2 + 4
        self.net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=4,
            n_agents=config.num_agents,
            centralized=config.policy_centralized,
            share_params=config.share_policy_params,
            depth=config.depth,
            num_cells=config.hidden_size,
            activation_class=nn.Tanh,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class CleanerMLPCriticNet(nn.Module):
    """Legacy flat-observation multi-agent critic."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        obs_dim = config.height * config.width + 1 + 2 + 4
        self.net = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=config.num_agents,
            centralized=config.critic_centralized,
            share_params=config.share_critic_params,
            depth=config.depth,
            num_cells=config.hidden_size,
            activation_class=nn.Tanh,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        per_agent_value = self.net(observation)
        return per_agent_value.mean(dim=-2)


class CleanerCNNPolicyNet(nn.Module):
    """Decentralized CNN policy over per-agent spatial observations."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.num_agents = config.num_agents
        self.share_params = config.share_policy_params
        side_dim = 1 + 4
        in_channels = 4 + int(config.obs_clean_channel) + int(config.obs_other_channel)
        if self.share_params:
            self.encoders = nn.ModuleList([
                SpatialCNNEncoder(in_channels, config.height, config.width, config.hidden_size)
            ])
            self.heads = nn.ModuleList([nn.Linear(config.hidden_size + side_dim, 4)])
        else:
            self.encoders = nn.ModuleList(
                [SpatialCNNEncoder(in_channels, config.height, config.width, config.hidden_size) for _ in range(config.num_agents)]
            )
            self.heads = nn.ModuleList(
                [nn.Linear(config.hidden_size + side_dim, 4) for _ in range(config.num_agents)]
            )

    def forward(
        self,
        spatial_observation: torch.Tensor,
        step_feature: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = []
        step_per_agent = step_feature.unsqueeze(-2).expand(*spatial_observation.shape[:-4], self.num_agents, 1)
        side = torch.cat([step_per_agent, action_mask.float()], dim=-1)
        for agent_idx in range(self.num_agents):
            module_idx = 0 if self.share_params else agent_idx
            embedding = self.encoders[module_idx](spatial_observation[..., agent_idx, :, :, :])
            logits.append(self.heads[module_idx](torch.cat([embedding, side[..., agent_idx, :]], dim=-1)))
        return torch.stack(logits, dim=-2)


class CleanerCNNCriticNet(nn.Module):
    """CNN critic: centralized by default, decentralized fallback for ablations."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.centralized = config.critic_centralized
        self.num_agents = config.num_agents
        self.share_params = config.share_critic_params
        side_dim = 1
        actor_in_channels = 4 + int(config.obs_clean_channel) + int(config.obs_other_channel)
        critic_in_channels = 3 + int(config.obs_clean_channel)
        if self.centralized:
            self.encoder = SpatialCNNEncoder(critic_in_channels, config.height, config.width, config.hidden_size)
            self.head = nn.Linear(config.hidden_size + side_dim, 1)
        elif self.share_params:
            self.encoders = nn.ModuleList([
                SpatialCNNEncoder(actor_in_channels, config.height, config.width, config.hidden_size)
            ])
            self.heads = nn.ModuleList([nn.Linear(config.hidden_size + side_dim, 1)])
        else:
            self.encoders = nn.ModuleList(
                [SpatialCNNEncoder(actor_in_channels, config.height, config.width, config.hidden_size) for _ in range(config.num_agents)]
            )
            self.heads = nn.ModuleList(
                [nn.Linear(config.hidden_size + side_dim, 1) for _ in range(config.num_agents)]
            )

    def forward(
        self,
        spatial_observation: torch.Tensor,
        global_spatial_observation: torch.Tensor,
        step_feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.centralized:
            embedding = self.encoder(global_spatial_observation)
            return self.head(torch.cat([embedding, step_feature], dim=-1))
        values = []
        for agent_idx in range(self.num_agents):
            module_idx = 0 if self.share_params else agent_idx
            embedding = self.encoders[module_idx](spatial_observation[..., agent_idx, :, :, :])
            values.append(self.heads[module_idx](torch.cat([embedding, step_feature], dim=-1)))
        return torch.stack(values, dim=-2).mean(dim=-2)


class SharedCleanerBackbone(nn.Module):
    """Legacy shared actor-critic trunk for flat MLP ablations."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        if config.encoder != "mlp":
            raise ValueError("shared-trunk is only supported with --encoder mlp")
        if config.policy_centralized != config.critic_centralized:
            raise ValueError("shared-trunk requires policy_centralized == critic_centralized")
        if config.share_policy_params != config.share_critic_params:
            raise ValueError("shared-trunk requires share_policy_params == share_critic_params")
        obs_dim = config.height * config.width + 1 + 2 + 4
        trunk_depth = max(config.depth - 1, 1)
        self.trunk = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=config.hidden_size,
            n_agents=config.num_agents,
            centralized=config.policy_centralized,
            share_params=config.share_policy_params,
            depth=trunk_depth,
            num_cells=config.hidden_size,
            activation_class=nn.Tanh,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.trunk(observation)


class SharedTrunkPolicyHead(nn.Module):
    def __init__(self, backbone: SharedCleanerBackbone, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 4)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(observation))


class SharedTrunkCriticHead(nn.Module):
    def __init__(self, backbone: SharedCleanerBackbone, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(observation)).mean(dim=-2)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--height", type=int, default=TrainConfig.height)
    parser.add_argument("--width", type=int, default=TrainConfig.width)
    parser.add_argument("--num-agents", type=int, default=TrainConfig.num_agents)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--n-iters", type=int, default=TrainConfig.n_iters)
    parser.add_argument("--frames-per-batch", type=int, default=TrainConfig.frames_per_batch)
    parser.add_argument("--ppo-epochs", type=int, default=TrainConfig.ppo_epochs)
    parser.add_argument("--minibatch-size", type=int, default=TrainConfig.minibatch_size)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--depth", type=int, default=TrainConfig.depth)
    parser.add_argument("--encoder", choices=["cnn", "mlp"], default=TrainConfig.encoder)
    parser.add_argument("--obs-clean-channel", type=str_to_bool, default=TrainConfig.obs_clean_channel)
    parser.add_argument("--obs-other-channel", type=str_to_bool, default=TrainConfig.obs_other_channel)
    parser.add_argument("--penalty-per-timestep", type=float, default=TrainConfig.penalty_per_timestep)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--lmbda", type=float, default=TrainConfig.lmbda)
    parser.add_argument("--clip-epsilon", type=float, default=TrainConfig.clip_epsilon)
    parser.add_argument("--entropy-coeff", type=float, default=TrainConfig.entropy_coeff)
    parser.add_argument("--critic-coeff", type=float, default=TrainConfig.critic_coeff)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--diversity-coeff", type=float, default=TrainConfig.diversity_coeff)
    parser.add_argument("--madpo-lite-coeff", type=float, default=None, help="Deprecated alias for --diversity-coeff.")
    parser.add_argument("--diversity-pairs", choices=["all", "adjacent", "sampled"], default=TrainConfig.diversity_pairs)
    parser.add_argument("--diversity-source-grad", type=str_to_bool, default=TrainConfig.diversity_source_grad)
    parser.add_argument("--diversity-sampled-pairs", type=int, default=TrainConfig.diversity_sampled_pairs)
    parser.add_argument("--diversity-bonus-max", type=float, default=TrainConfig.diversity_bonus_max, help="Clip the CE diversity bonus used in the loss; set <=0 to disable.")
    parser.add_argument("--policy-centralized", type=str_to_bool, default=TrainConfig.policy_centralized)
    parser.add_argument("--critic-centralized", type=str_to_bool, default=TrainConfig.critic_centralized)
    parser.add_argument("--share-policy-params", type=str_to_bool, default=TrainConfig.share_policy_params)
    parser.add_argument("--share-critic-params", type=str_to_bool, default=TrainConfig.share_critic_params)
    parser.add_argument("--ac-architecture", choices=["separate", "shared-trunk"], default=TrainConfig.ac_architecture)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--eval-rollout-length", type=int, default=TrainConfig.eval_rollout_length)
    parser.add_argument("--eval-frequency", type=int, default=TrainConfig.eval_frequency)
    parser.add_argument("--early-stop-mode-success", type=float, default=TrainConfig.early_stop_mode_success)
    parser.add_argument("--early-stop-sample-success", type=float, default=TrainConfig.early_stop_sample_success)
    parser.add_argument("--render-frequency", type=int, default=TrainConfig.render_frequency)
    parser.add_argument("--skip-spec-check", action="store_true")
    args = parser.parse_args()
    diversity_coeff = args.diversity_coeff if args.madpo_lite_coeff is None else args.madpo_lite_coeff
    return TrainConfig(
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_agents=args.num_agents,
        num_envs=args.num_envs,
        n_iters=args.n_iters,
        frames_per_batch=args.frames_per_batch,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        encoder=args.encoder,
        obs_clean_channel=args.obs_clean_channel,
        obs_other_channel=args.obs_other_channel,
        penalty_per_timestep=args.penalty_per_timestep,
        lr=args.lr,
        gamma=args.gamma,
        lmbda=args.lmbda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        max_grad_norm=args.max_grad_norm,
        diversity_coeff=diversity_coeff,
        diversity_pairs=args.diversity_pairs,
        diversity_source_grad=args.diversity_source_grad,
        diversity_sampled_pairs=args.diversity_sampled_pairs,
        diversity_bonus_max=args.diversity_bonus_max,
        policy_centralized=args.policy_centralized,
        critic_centralized=args.critic_centralized,
        share_policy_params=args.share_policy_params,
        share_critic_params=args.share_critic_params,
        ac_architecture=args.ac_architecture,
        output_dir=args.output_dir,
        eval_rollout_length=args.eval_rollout_length,
        eval_frequency=args.eval_frequency,
        early_stop_mode_success=args.early_stop_mode_success,
        early_stop_sample_success=args.early_stop_sample_success,
        render_frequency=args.render_frequency,
        check_specs=not args.skip_spec_check,
    )


def require_cuda() -> torch.device:
    print(f"jax devices: {_JAX_DEVICES}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"torch cuda version: {torch.version.cuda}")
    if not any(device.platform == "gpu" for device in _JAX_DEVICES):
        raise RuntimeError("JAX did not report a CUDA/GPU device")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch is not CUDA-enabled")
    return torch.device("cuda")


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device: torch.device) -> float:
    cuda_sync(device)
    return time.perf_counter()


def build_env(config: TrainConfig, device: torch.device, seed: int | None = None) -> TransformedEnv:
    generator = FixedObstacleGenerator(
        height=config.height,
        width=config.width,
        num_agents=config.num_agents,
    )
    cleaner = CustomCleaner(
        generator=generator,
        time_limit=config.height * config.width,
        penalty_per_timestep=config.penalty_per_timestep,
    )
    base_env = JumanjiWrapper(
        cleaner,
        batch_size=[config.num_envs],
        categorical_action_encoding=True,
        jit=True,
        device=device,
    )
    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=["episode_reward"]),
    )
    env.set_seed(config.seed if seed is None else seed)
    return env


def make_observation_module(config: TrainConfig, device: torch.device) -> TensorDictModule:
    return TensorDictModule(
        CleanerObservationBuilder(
            config.height,
            config.width,
            config.num_agents,
            obs_clean_channel=config.obs_clean_channel,
            obs_other_channel=config.obs_other_channel,
        ).to(device),
        in_keys=["grid", "agents_locations", "step_count", "action_mask"],
        out_keys=[
            ("agents", "observation"),
            ("agents", "spatial_observation"),
            ("agents", "action_mask"),
            "global_observation",
            "global_spatial_observation",
            "step_feature",
        ],
    )


def build_modules(
    config: TrainConfig,
    device: torch.device,
) -> tuple[ProbabilisticActor, TensorDictModule, TensorDictSequential]:
    actor_obs = make_observation_module(config, device)
    critic_obs = make_observation_module(config, device)

    if config.encoder == "mlp":
        if config.ac_architecture == "shared-trunk":
            backbone = SharedCleanerBackbone(config).to(device)
            policy_net = SharedTrunkPolicyHead(backbone, config.hidden_size).to(device)
            critic_net = SharedTrunkCriticHead(backbone, config.hidden_size).to(device)
        else:
            policy_net = CleanerMLPPolicyNet(config).to(device)
            critic_net = CleanerMLPCriticNet(config).to(device)
        policy_logits = TensorDictModule(
            policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )
        critic_value = TensorDictModule(
            critic_net,
            in_keys=[("agents", "observation")],
            out_keys=["state_value"],
        )
    else:
        if config.ac_architecture == "shared-trunk":
            raise ValueError("shared-trunk is only supported with --encoder mlp")
        policy_net = CleanerCNNPolicyNet(config).to(device)
        critic_net = CleanerCNNCriticNet(config).to(device)
        policy_logits = TensorDictModule(
            policy_net,
            in_keys=[("agents", "spatial_observation"), "step_feature", ("agents", "action_mask")],
            out_keys=[("agents", "logits")],
        )
        critic_value = TensorDictModule(
            critic_net,
            in_keys=[("agents", "spatial_observation"), "global_spatial_observation", "step_feature"],
            out_keys=["state_value"],
        )

    actor_module = TensorDictSequential(actor_obs, policy_logits)
    actor = ProbabilisticActor(
        actor_module,
        in_keys={"logits": ("agents", "logits"), "mask": ("agents", "action_mask")},
        out_keys=["action"],
        distribution_class=PerAgentMaskedCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    critic = TensorDictSequential(critic_obs, critic_value)
    return actor, critic, actor_module


def pair_indices(num_agents: int, mode: str, max_sampled_pairs: int, device: torch.device) -> list[tuple[int, int]]:
    if num_agents < 2:
        return []
    if mode == "adjacent":
        return [(idx, idx + 1) for idx in range(num_agents - 1)]
    pairs = [(i, j) for i in range(num_agents) for j in range(num_agents) if i != j]
    if mode == "sampled" and len(pairs) > max_sampled_pairs:
        perm = torch.randperm(len(pairs), device=device)[:max_sampled_pairs].cpu().tolist()
        return [pairs[idx] for idx in perm]
    return pairs


def masked_cross_entropy_pair(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    detach_source: bool,
) -> torch.Tensor:
    common_mask = source_mask & target_mask
    fallback_mask = source_mask | target_mask
    has_common = common_mask.any(dim=-1, keepdim=True)
    mask = torch.where(has_common, common_mask, fallback_mask)
    source_log_probs = torch.log_softmax(source_logits.masked_fill(~mask, -1.0e9), dim=-1)
    target_log_probs = torch.log_softmax(target_logits.masked_fill(~mask, -1.0e9), dim=-1)
    source_probs = source_log_probs.exp().masked_fill(~mask, 0.0)
    if detach_source:
        source_probs = source_probs.detach()
    return -(source_probs * target_log_probs).sum(dim=-1).mean()


def compute_diversity_bonus(
    actor_module: TensorDictSequential,
    minibatch: TensorDict,
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not config.diversity_coeff or config.num_agents < 2:
        zero = torch.zeros((), device=device)
        return zero, zero
    td = minibatch.select("grid", "agents_locations", "step_count", "action_mask").clone()
    actor_module(td)
    logits = td[("agents", "logits")]
    mask = td[("agents", "action_mask")]
    values = []
    for source, target in pair_indices(
        config.num_agents,
        config.diversity_pairs,
        config.diversity_sampled_pairs,
        device,
    ):
        values.append(
            masked_cross_entropy_pair(
                logits[..., source, :],
                logits[..., target, :],
                mask[..., source, :],
                mask[..., target, :],
                detach_source=not config.diversity_source_grad,
            )
        )
    if not values:
        zero = torch.zeros((), device=device)
        return zero, zero
    raw_bonus = torch.stack(values).mean()
    if config.diversity_bonus_max > 0.0:
        clipped_values = [value.clamp(max=config.diversity_bonus_max) for value in values]
        return torch.stack(clipped_values).mean(), raw_bonus
    return raw_bonus, raw_bonus


def normalize_value_estimator_shapes(batch: TensorDict, num_agents: int) -> None:
    advantage = batch.get("advantage", None)
    if advantage is not None:
        if advantage.ndim == len(batch.batch_size):
            advantage = advantage.unsqueeze(-1)
        if advantage.shape[-1] == 1:
            advantage = advantage.expand(*advantage.shape[:-1], num_agents)
        if advantage.ndim == len(batch.batch_size) + 1:
            advantage = advantage.unsqueeze(-1)
        batch.set("advantage", advantage)
    value_target = batch.get("value_target", None)
    if value_target is not None and value_target.ndim == len(batch.batch_size):
        batch.set("value_target", value_target.unsqueeze(-1))


def flatten_batch(batch: TensorDict) -> TensorDict:
    return batch.reshape(-1)


def scalar_from_loss(losses: TensorDict, key: str) -> float:
    value = losses.get(key)
    if value is None:
        return 0.0
    return float(value.detach().mean().cpu())


def completed_episode_return_mean(batch: TensorDict) -> tuple[float, int]:
    dones = batch.get(("next", "done")).squeeze(-1).bool()
    episode_rewards = batch.get(("next", "episode_reward")).squeeze(-1)
    completed = episode_rewards[dones]
    if completed.numel() == 0:
        return float("nan"), 0
    return float(completed.mean().detach().cpu()), int(completed.numel())


def dirty_counts_from_grid(grid: torch.Tensor) -> torch.Tensor:
    return (grid == DIRTY).sum(dim=(-1, -2))


def summarize_eval_rollout(rollout: TensorDict, config: TrainConfig, prefix: str) -> dict[str, float]:
    rewards = rollout.get(("next", "reward")).sum(dim=1).squeeze(-1)
    next_grid = rollout.get(("next", "grid"))
    dirty_counts = dirty_counts_from_grid(next_grid)
    solved_by_step = dirty_counts == 0
    solved = solved_by_step.any(dim=1)
    first_solved_step = solved_by_step.float().argmax(dim=1).float() + 1.0
    timeout_step = torch.full_like(first_solved_step, float(config.eval_rollout_length))
    completion_step = torch.where(solved, first_solved_step, timeout_step)
    success = solved.float()
    success_completion_step = completion_step[solved].mean() if solved.any() else torch.tensor(float("nan"), device=completion_step.device)
    non_wall_counts = (next_grid != WALL).sum(dim=(-1, -2)).clamp_min(1)
    final_dirty_fraction = dirty_counts[:, -1].float() / non_wall_counts[:, -1].float()
    return {
        f"{prefix}_return_mean": float(rewards.mean().detach().cpu()),
        f"{prefix}_success_rate": float(success.mean().detach().cpu()),
        f"{prefix}_final_dirty_fraction": float(final_dirty_fraction.mean().detach().cpu()),
        f"{prefix}_completion_step_mean": float(completion_step.mean().detach().cpu()),
        f"{prefix}_completion_step_success_mean": float(success_completion_step.detach().cpu()),
    }


def evaluate_policy(
    env: JumanjiWrapper,
    actor: ProbabilisticActor,
    rollout_length: int,
    config: TrainConfig,
) -> dict[str, float | TensorDict]:
    with torch.no_grad(), set_interaction_type(InteractionType.MODE):
        mode_rollout = env.rollout(rollout_length, policy=actor)
    with torch.no_grad(), set_interaction_type(InteractionType.RANDOM):
        sample_rollout = env.rollout(rollout_length, policy=actor)
    values: dict[str, float | TensorDict] = {
        **summarize_eval_rollout(mode_rollout, config, "eval_mode"),
        **summarize_eval_rollout(sample_rollout, config, "eval_sample"),
        "rollout": mode_rollout.detach().cpu(),
    }
    values["eval_return_mean"] = values["eval_mode_return_mean"]
    values["eval_success_rate"] = values["eval_mode_success_rate"]
    values["eval_final_dirty_fraction"] = values["eval_mode_final_dirty_fraction"]
    values["eval_completion_step_mean"] = values["eval_mode_completion_step_mean"]
    values["eval_completion_step_success_mean"] = values["eval_mode_completion_step_success_mean"]
    return values


def eval_score(values: dict[str, float | TensorDict]) -> tuple[float, float, float]:
    return (
        float(values["eval_sample_success_rate"]),
        -float(values["eval_sample_final_dirty_fraction"]),
        float(values["eval_sample_return_mean"]),
    )


def save_best_checkpoint(
    output_dir: Path,
    iteration: int,
    frames: int,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    config: TrainConfig,
    eval_values: dict[str, float | TensorDict],
) -> None:
    checkpoint_path = output_dir / "best_policy.pt"
    torch.save(
        {
            "iteration": iteration,
            "frames": frames,
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": asdict(config),
            "eval": {key: value for key, value in eval_values.items() if key != "rollout"},
        },
        checkpoint_path,
    )
    (output_dir / "best_policy.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "frames": frames,
                "checkpoint": str(checkpoint_path),
                "eval": {key: value for key, value in eval_values.items() if key != "rollout"},
            },
            indent=2,
        )
        + "\n"
    )


def render_rollout_gif(rollout: TensorDict, output_path: Path, config: TrainConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_env = 0
    grids = rollout.get(("next", "grid"))[first_env]
    locations = rollout.get(("next", "agents_locations"))[first_env]
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.matplotlib.colors.ListedColormap(["#8b8b8b", "#f8f8f8", "#222222"])

    def draw(frame_idx: int) -> list[Any]:
        ax.clear()
        ax.imshow(grids[frame_idx].numpy(), cmap=cmap, vmin=0, vmax=2)
        loc = locations[frame_idx].numpy()
        ax.scatter(loc[:, 1], loc[:, 0], c="#d62728", s=80, edgecolors="white", linewidths=1.0)
        for agent_idx, (row, col) in enumerate(loc):
            ax.text(col, row, str(agent_idx), color="white", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"step {frame_idx + 1}")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=grids.shape[0], interval=180, blit=False)
    ani.save(output_path, writer=animation.PillowWriter(fps=5))
    plt.close(fig)


def write_metrics(output_dir: Path, metrics: list[dict[str, float]]) -> None:
    if not metrics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    frames = [row["frames"] for row in metrics]
    axes[0].plot(frames, [row["episode_return_mean"] for row in metrics], label="train completed episode return")
    axes[0].plot(frames, [row["collector_window_return_mean"] for row in metrics], label="collector window return", alpha=0.35)
    axes[0].plot(frames, [row["eval_mode_return_mean"] for row in metrics], label="eval mode")
    axes[0].plot(frames, [row["eval_sample_return_mean"] for row in metrics], label="eval sample")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(frames, [row["eval_mode_success_rate"] for row in metrics], label="mode success rate")
    axes[1].plot(frames, [row["eval_sample_success_rate"] for row in metrics], label="sample success rate")
    axes[1].plot(frames, [row["eval_mode_final_dirty_fraction"] for row in metrics], label="mode dirty fraction")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[2].plot(frames, [row["eval_mode_completion_step_mean"] for row in metrics], label="mode completion step")
    axes[2].plot(frames, [row["eval_sample_completion_step_mean"] for row in metrics], label="sample completion step")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Completion step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.suptitle("CustomCleaner PPO training")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close(fig)


def validate_config(config: TrainConfig) -> None:
    if config.frames_per_batch % config.num_envs != 0:
        raise ValueError("frames_per_batch must be divisible by num_envs for this seminar script")
    if config.minibatch_size > config.frames_per_batch:
        raise ValueError("minibatch_size must be <= frames_per_batch")
    if config.depth < 1:
        raise ValueError("depth must be >= 1")
    if config.eval_frequency < 1:
        raise ValueError("eval_frequency must be >= 1")
    if not 0.0 <= config.early_stop_mode_success <= 1.0:
        raise ValueError("early_stop_mode_success must be between 0 and 1")
    if not 0.0 <= config.early_stop_sample_success <= 1.0:
        raise ValueError("early_stop_sample_success must be between 0 and 1")
    if config.diversity_sampled_pairs < 1:
        raise ValueError("diversity_sampled_pairs must be >= 1")
    if config.diversity_bonus_max < 0.0:
        raise ValueError("diversity_bonus_max must be >= 0; use 0 to disable clipping")


def main() -> None:
    config = parse_args()
    validate_config(config)
    torch.manual_seed(config.seed)
    device = require_cuda()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    env = build_env(config, device, seed=config.seed)
    eval_env = build_env(config, device, seed=config.seed + 10_000)
    if config.check_specs:
        check_env_specs(env)
    actor, critic, actor_module = build_modules(config, device)
    print("action_spec:", env.action_spec)
    print("reward_spec:", env.reward_spec)
    print(
        "architecture:",
        json.dumps(
            {
                "encoder": config.encoder,
                "obs_clean_channel": config.obs_clean_channel,
                "obs_other_channel": config.obs_other_channel,
                "penalty_per_timestep": config.penalty_per_timestep,
                "policy_centralized": config.policy_centralized,
                "critic_centralized": config.critic_centralized,
                "share_policy_params": config.share_policy_params,
                "share_critic_params": config.share_critic_params,
                "ac_architecture": config.ac_architecture,
                "depth": config.depth,
                "hidden_size": config.hidden_size,
            },
            sort_keys=True,
        ),
    )

    collector = Collector(
        env,
        actor,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.frames_per_batch * config.n_iters,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_coeff,
        critic_coeff=config.critic_coeff,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward="reward",
        action="action",
        sample_log_prob="sample_log_prob",
        value="state_value",
        done="done",
        terminated="terminated",
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

    metrics: list[dict[str, float]] = []
    best_score: tuple[float, float, float] | None = None
    wall_start = time.perf_counter()
    collector_iter = iter(collector)
    for iteration in range(1, config.n_iters + 1):
        t0 = now(device)
        batch = next(collector_iter).to(device)
        collection_sec = now(device) - t0

        rewards = batch.get(("next", "reward"))
        episode_return_mean, completed_episodes = completed_episode_return_mean(batch)

        t0 = now(device)
        with torch.no_grad():
            loss_module.value_estimator(batch)
            normalize_value_estimator_shapes(batch, config.num_agents)
        gae_sec = now(device) - t0

        flat_batch = flatten_batch(batch.detach())
        last_losses: dict[str, float] = {}
        diversity_bonus_value = 0.0
        diversity_bonus_raw_value = 0.0
        grad_norm_value = 0.0
        loss_forward_sec = 0.0
        diversity_sec = 0.0
        backward_sec = 0.0
        optimizer_sec = 0.0
        for _ in range(config.ppo_epochs):
            permutation = torch.randperm(flat_batch.numel(), device=device)
            for start_idx in range(0, flat_batch.numel(), config.minibatch_size):
                indices = permutation[start_idx : start_idx + config.minibatch_size]
                minibatch = flat_batch[indices]

                t0 = now(device)
                losses = loss_module(minibatch)
                total_loss = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                loss_forward_sec += now(device) - t0

                if config.diversity_coeff:
                    t0 = now(device)
                    diversity_bonus, diversity_bonus_raw = compute_diversity_bonus(actor_module, minibatch, config, device)
                    total_loss = total_loss - config.diversity_coeff * diversity_bonus
                    diversity_sec += now(device) - t0
                    diversity_bonus_value = float(diversity_bonus.detach().cpu())
                    diversity_bonus_raw_value = float(diversity_bonus_raw.detach().cpu())
                else:
                    diversity_bonus_value = float("nan")
                    diversity_bonus_raw_value = float("nan")

                t0 = now(device)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm)
                backward_sec += now(device) - t0
                grad_norm_value = float(grad_norm.detach().cpu())

                t0 = now(device)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_sec += now(device) - t0

                last_losses = {
                    "loss_objective": scalar_from_loss(losses, "loss_objective"),
                    "loss_critic": scalar_from_loss(losses, "loss_critic"),
                    "loss_entropy": scalar_from_loss(losses, "loss_entropy"),
                    "entropy": scalar_from_loss(losses, "entropy"),
                    "kl_approx": scalar_from_loss(losses, "kl_approx"),
                    "clip_fraction": scalar_from_loss(losses, "clip_fraction"),
                }

        collector_window_return_mean = float(rewards.sum(dim=1).mean().detach().cpu())

        eval_sec = 0.0
        eval_values: dict[str, float | TensorDict]
        if iteration % config.eval_frequency == 0 or iteration == config.n_iters:
            t0 = now(device)
            eval_values = evaluate_policy(eval_env, actor, config.eval_rollout_length, config)
            eval_sec = now(device) - t0
            current_score = eval_score(eval_values)
            if best_score is None or current_score > best_score:
                best_score = current_score
                save_best_checkpoint(
                    output_dir,
                    iteration,
                    iteration * config.frames_per_batch,
                    actor,
                    critic,
                    config,
                    eval_values,
                )
                if config.render_frequency:
                    render_rollout_gif(eval_values["rollout"], output_dir / "best_policy.gif", config)
            if config.render_frequency and (iteration % config.render_frequency == 0 or iteration == config.n_iters):
                render_rollout_gif(
                    eval_values["rollout"],
                    output_dir / f"policy_iter_{iteration:04d}.gif",
                    config,
                )
        else:
            eval_values = {
                "eval_return_mean": float("nan"),
                "eval_success_rate": float("nan"),
                "eval_final_dirty_fraction": float("nan"),
                "eval_mode_return_mean": float("nan"),
                "eval_mode_success_rate": float("nan"),
                "eval_mode_final_dirty_fraction": float("nan"),
                "eval_mode_completion_step_mean": float("nan"),
                "eval_mode_completion_step_success_mean": float("nan"),
                "eval_sample_return_mean": float("nan"),
                "eval_sample_success_rate": float("nan"),
                "eval_sample_final_dirty_fraction": float("nan"),
                "eval_sample_completion_step_mean": float("nan"),
                "eval_sample_completion_step_success_mean": float("nan"),
            }

        row = {
            "iter": float(iteration),
            "frames": float(iteration * config.frames_per_batch),
            "collector_window_return_mean": collector_window_return_mean,
            "batch_return_mean": collector_window_return_mean,
            "episode_return_mean": episode_return_mean,
            "completed_episodes": float(completed_episodes),
            "eval_return_mean": float(eval_values["eval_return_mean"]),
            "eval_success_rate": float(eval_values["eval_success_rate"]),
            "eval_final_dirty_fraction": float(eval_values["eval_final_dirty_fraction"]),
            "eval_mode_return_mean": float(eval_values["eval_mode_return_mean"]),
            "eval_mode_success_rate": float(eval_values["eval_mode_success_rate"]),
            "eval_mode_final_dirty_fraction": float(eval_values["eval_mode_final_dirty_fraction"]),
            "eval_mode_completion_step_mean": float(eval_values["eval_mode_completion_step_mean"]),
            "eval_mode_completion_step_success_mean": float(eval_values["eval_mode_completion_step_success_mean"]),
            "eval_sample_return_mean": float(eval_values["eval_sample_return_mean"]),
            "eval_sample_success_rate": float(eval_values["eval_sample_success_rate"]),
            "eval_sample_final_dirty_fraction": float(eval_values["eval_sample_final_dirty_fraction"]),
            "eval_sample_completion_step_mean": float(eval_values["eval_sample_completion_step_mean"]),
            "eval_sample_completion_step_success_mean": float(eval_values["eval_sample_completion_step_success_mean"]),
            "diversity_cross_entropy": diversity_bonus_value,
            "diversity_cross_entropy_raw": diversity_bonus_raw_value,
            "elapsed_sec": time.perf_counter() - wall_start,
            "collection_sec": collection_sec,
            "gae_sec": gae_sec,
            "loss_forward_sec": loss_forward_sec,
            "diversity_sec": diversity_sec,
            "backward_sec": backward_sec,
            "optimizer_sec": optimizer_sec,
            "eval_sec": eval_sec,
            "grad_norm": grad_norm_value,
            **last_losses,
        }
        metrics.append(row)
        print(
            f"iter={iteration:03d} frames={int(row['frames'])} "
            f"episode_return={episode_return_mean:.3f} completed={completed_episodes} "
            f"window_return={collector_window_return_mean:.3f} eval_mode={row['eval_mode_return_mean']:.3f} "
            f"eval_sample={row['eval_sample_return_mean']:.3f} "
            f"mode_sr={row['eval_mode_success_rate']:.3f} sample_sr={row['eval_sample_success_rate']:.3f} "
            f"mode_step={row['eval_mode_completion_step_mean']:.1f} sample_step={row['eval_sample_completion_step_mean']:.1f} "
            f"entropy={row.get('entropy', 0.0):.4f} "
            f"ce={diversity_bonus_value:.4f} raw_ce={diversity_bonus_raw_value:.4f} time={row['elapsed_sec']:.1f}s"
        )
        write_metrics(output_dir, metrics)
        if (
            row["eval_mode_success_rate"] >= config.early_stop_mode_success
            and row["eval_sample_success_rate"] >= config.early_stop_sample_success
        ):
            print(
                "early stopping: "
                f"mode_sr={row['eval_mode_success_rate']:.3f} >= {config.early_stop_mode_success:.3f}, "
                f"sample_sr={row['eval_sample_success_rate']:.3f} >= {config.early_stop_sample_success:.3f}"
            )
            break

    print(f"metrics written to: {output_dir / 'metrics.csv'}")
    print(f"learning curve written to: {output_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
