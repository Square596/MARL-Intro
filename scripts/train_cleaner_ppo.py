#!/usr/bin/env python3
"""Train a TorchRL PPO policy on the custom Jumanji Cleaner seminar env."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp

# Initialize JAX before importing Torch so PyTorch's bundled cuDNN does not
# shadow the CUDA libraries used by the JAX CUDA plugin.
_JAX_DEVICES = jax.devices()
jnp.asarray(0, dtype=jnp.int8).block_until_ready()

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs import JumanjiWrapper
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MaskedCategorical, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from custom_cleaner import CustomCleaner, FixedObstacleGenerator


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    height: int = 8
    width: int = 8
    num_agents: int = 3
    num_envs: int = 16
    n_iters: int = 20
    frames_per_batch: int = 512
    ppo_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    lr: float = 3.0e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 1.0e-3
    critic_coeff: float = 1.0
    max_grad_norm: float = 1.0
    madpo_lite_coeff: float = 0.0
    output_dir: str = ".artifacts/cleaner_ppo"
    eval_rollout_length: int = 64
    check_specs: bool = True


class JointMaskedCategorical(MaskedCategorical):
    """Masked per-agent categorical policy viewed as one joint action."""

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value).sum(dim=-1, keepdim=True)

    def entropy(self) -> torch.Tensor:
        return super().entropy().sum(dim=-1, keepdim=True)


class CleanerActorNet(nn.Module):
    """Shared per-agent policy over four Cleaner moves."""

    def __init__(self, height: int, width: int, num_agents: int, hidden_size: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_agents = num_agents
        input_dim = height * width + num_agents * 2 + 1 + 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4),
        )

    def forward(
        self,
        grid: torch.Tensor,
        agents_locations: torch.Tensor,
        step_count: torch.Tensor,
    ) -> torch.Tensor:
        leading_shape = grid.shape[:-2]
        normalizer = torch.tensor([self.height, self.width], device=grid.device)
        grid_features = grid.float().flatten(-2, -1) / 2.0
        all_locations = (agents_locations.float() / normalizer).flatten(-2, -1)
        own_locations = agents_locations.float() / normalizer
        step = (step_count.float() / float(self.height * self.width)).unsqueeze(-1)
        base = torch.cat([grid_features, all_locations, step], dim=-1)
        base = base.unsqueeze(-2).expand(*leading_shape, self.num_agents, base.shape[-1])
        return self.net(torch.cat([base, own_locations], dim=-1))


class CleanerCriticNet(nn.Module):
    """Centralized value function for the shared Cleaner team reward."""

    def __init__(self, height: int, width: int, num_agents: int, hidden_size: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_agents = num_agents
        input_dim = height * width + num_agents * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        grid: torch.Tensor,
        agents_locations: torch.Tensor,
        step_count: torch.Tensor,
    ) -> torch.Tensor:
        normalizer = torch.tensor([self.height, self.width], device=grid.device)
        grid_features = grid.float().flatten(-2, -1) / 2.0
        all_locations = (agents_locations.float() / normalizer).flatten(-2, -1)
        step = (step_count.float() / float(self.height * self.width)).unsqueeze(-1)
        return self.net(torch.cat([grid_features, all_locations, step], dim=-1))


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
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--lmbda", type=float, default=TrainConfig.lmbda)
    parser.add_argument("--clip-epsilon", type=float, default=TrainConfig.clip_epsilon)
    parser.add_argument("--entropy-coeff", type=float, default=TrainConfig.entropy_coeff)
    parser.add_argument("--critic-coeff", type=float, default=TrainConfig.critic_coeff)
    parser.add_argument("--max-grad-norm", type=float, default=TrainConfig.max_grad_norm)
    parser.add_argument("--madpo-lite-coeff", type=float, default=TrainConfig.madpo_lite_coeff)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--eval-rollout-length", type=int, default=TrainConfig.eval_rollout_length)
    parser.add_argument("--skip-spec-check", action="store_true")
    args = parser.parse_args()
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
        lr=args.lr,
        gamma=args.gamma,
        lmbda=args.lmbda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
        max_grad_norm=args.max_grad_norm,
        madpo_lite_coeff=args.madpo_lite_coeff,
        output_dir=args.output_dir,
        eval_rollout_length=args.eval_rollout_length,
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


def build_env(config: TrainConfig, device: torch.device) -> JumanjiWrapper:
    generator = FixedObstacleGenerator(
        height=config.height,
        width=config.width,
        num_agents=config.num_agents,
    )
    cleaner = CustomCleaner(generator=generator, time_limit=config.height * config.width)
    env = JumanjiWrapper(
        cleaner,
        batch_size=[config.num_envs],
        categorical_action_encoding=True,
        jit=True,
        device=device,
    )
    env.set_seed(config.seed)
    return env


def build_modules(config: TrainConfig, device: torch.device) -> tuple[ProbabilisticActor, TensorDictModule]:
    actor_module = TensorDictModule(
        CleanerActorNet(config.height, config.width, config.num_agents, config.hidden_size).to(device),
        in_keys=["grid", "agents_locations", "step_count"],
        out_keys=["logits"],
    )
    actor = ProbabilisticActor(
        actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=JointMaskedCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    critic = TensorDictModule(
        CleanerCriticNet(config.height, config.width, config.num_agents, config.hidden_size).to(device),
        in_keys=["grid", "agents_locations", "step_count"],
        out_keys=["state_value"],
    )
    return actor, critic


def masked_adjacent_policy_divergence(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Symmetric KL over adjacent agent policies on common legal actions."""
    if logits.shape[-2] < 2:
        return logits.new_zeros(())
    left_logits = logits[..., :-1, :]
    right_logits = logits[..., 1:, :]
    common_mask = mask[..., :-1, :] & mask[..., 1:, :]
    fallback_mask = mask[..., :-1, :] | mask[..., 1:, :]
    has_common = common_mask.any(dim=-1, keepdim=True)
    common_mask = torch.where(has_common, common_mask, fallback_mask)
    left_logits = left_logits.masked_fill(~common_mask, -1.0e9)
    right_logits = right_logits.masked_fill(~common_mask, -1.0e9)
    left_log_probs = torch.log_softmax(left_logits, dim=-1)
    right_log_probs = torch.log_softmax(right_logits, dim=-1)
    left_probs = left_log_probs.exp().masked_fill(~common_mask, 0.0)
    right_probs = right_log_probs.exp().masked_fill(~common_mask, 0.0)
    kl_left_right = (left_probs * (left_log_probs - right_log_probs)).sum(dim=-1)
    kl_right_left = (right_probs * (right_log_probs - left_log_probs)).sum(dim=-1)
    return (0.5 * (kl_left_right + kl_right_left)).mean()


def compute_madpo_lite_penalty(
    actor_module: TensorDictModule,
    minibatch: TensorDict,
) -> torch.Tensor:
    td = minibatch.select("grid", "agents_locations", "step_count", "action_mask").clone()
    actor_module(td)
    return masked_adjacent_policy_divergence(td["logits"], td["action_mask"])


def flatten_batch(batch: TensorDict) -> TensorDict:
    return batch.reshape(-1)


def scalar_from_loss(losses: TensorDict, key: str) -> float:
    value = losses.get(key)
    if value is None:
        return 0.0
    return float(value.detach().mean().cpu())


def update_episode_returns(
    running_returns: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    completed: list[float] = []
    rewards_cpu = rewards.squeeze(-1).detach().cpu()
    dones_cpu = dones.squeeze(-1).detach().cpu()
    for timestep in range(rewards_cpu.shape[1]):
        running_returns += rewards_cpu[:, timestep]
        done_indices = torch.nonzero(dones_cpu[:, timestep], as_tuple=False).flatten()
        for env_idx in done_indices.tolist():
            completed.append(float(running_returns[env_idx]))
            running_returns[env_idx] = 0.0
    return running_returns, completed


def evaluate_policy(env: JumanjiWrapper, actor: ProbabilisticActor, rollout_length: int) -> float:
    with torch.no_grad():
        rollout = env.rollout(rollout_length, policy=actor)
    returns = rollout.get(("next", "reward")).sum(dim=1).squeeze(-1)
    return float(returns.mean().detach().cpu())


def write_metrics(output_dir: Path, metrics: list[dict[str, float]]) -> None:
    if not metrics:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    fig, ax = plt.subplots(figsize=(7, 4))
    iterations = [row["iter"] for row in metrics]
    ax.plot(iterations, [row["batch_return_mean"] for row in metrics], label="batch return")
    ax.plot(iterations, [row["eval_return_mean"] for row in metrics], label="eval return")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Return")
    ax.set_title("CustomCleaner PPO smoke training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = parse_args()
    if config.frames_per_batch % config.num_envs != 0:
        raise ValueError("frames_per_batch must be divisible by num_envs for this seminar script")
    if config.minibatch_size > config.frames_per_batch:
        raise ValueError("minibatch_size must be <= frames_per_batch")
    torch.manual_seed(config.seed)
    device = require_cuda()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    env = build_env(config, device)
    if config.check_specs:
        check_env_specs(env)
    actor, critic = build_modules(config, device)
    print("action_spec:", env.action_spec)
    print("reward_spec:", env.reward_spec)

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
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.lr,
    )

    metrics: list[dict[str, float]] = []
    running_returns = torch.zeros(config.num_envs)
    start = time.time()
    for iteration, batch in enumerate(collector, start=1):
        batch = batch.to(device)
        rewards = batch.get(("next", "reward"))
        dones = batch.get(("next", "done"))
        running_returns, completed_returns = update_episode_returns(running_returns, rewards, dones)
        with torch.no_grad():
            loss_module.value_estimator(batch)
        flat_batch = flatten_batch(batch.detach())
        last_losses: dict[str, float] = {}
        madpo_penalty_value = 0.0
        for _ in range(config.ppo_epochs):
            permutation = torch.randperm(flat_batch.numel(), device=device)
            for start_idx in range(0, flat_batch.numel(), config.minibatch_size):
                indices = permutation[start_idx : start_idx + config.minibatch_size]
                minibatch = flat_batch[indices]
                losses = loss_module(minibatch)
                total_loss = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                )
                if config.madpo_lite_coeff:
                    madpo_penalty = compute_madpo_lite_penalty(actor.module[0], minibatch)
                    total_loss = total_loss + config.madpo_lite_coeff * madpo_penalty
                    madpo_penalty_value = float(madpo_penalty.detach().cpu())
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                last_losses = {
                    "loss_objective": scalar_from_loss(losses, "loss_objective"),
                    "loss_critic": scalar_from_loss(losses, "loss_critic"),
                    "loss_entropy": scalar_from_loss(losses, "loss_entropy"),
                    "entropy": scalar_from_loss(losses, "entropy"),
                    "kl_approx": scalar_from_loss(losses, "kl_approx"),
                    "clip_fraction": scalar_from_loss(losses, "clip_fraction"),
                }
        batch_return_mean = float(rewards.sum(dim=1).mean().detach().cpu())
        episode_return_mean = (
            sum(completed_returns) / len(completed_returns) if completed_returns else float("nan")
        )
        eval_return_mean = evaluate_policy(env, actor, config.eval_rollout_length)
        row = {
            "iter": float(iteration),
            "frames": float(iteration * config.frames_per_batch),
            "batch_return_mean": batch_return_mean,
            "episode_return_mean": episode_return_mean,
            "eval_return_mean": eval_return_mean,
            "madpo_lite_penalty": madpo_penalty_value,
            "elapsed_sec": time.time() - start,
            **last_losses,
        }
        metrics.append(row)
        print(
            f"iter={iteration:03d} frames={int(row['frames'])} "
            f"batch_return={batch_return_mean:.3f} eval_return={eval_return_mean:.3f} "
            f"loss_pi={row.get('loss_objective', 0.0):.4f} "
            f"loss_v={row.get('loss_critic', 0.0):.4f} "
            f"entropy={row.get('entropy', 0.0):.4f} "
            f"madpo={madpo_penalty_value:.4f}"
        )
        write_metrics(output_dir, metrics)
        if iteration >= config.n_iters:
            break
    print(f"metrics written to: {output_dir / 'metrics.csv'}")
    print(f"learning curve written to: {output_dir / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
