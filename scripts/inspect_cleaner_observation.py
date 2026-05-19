#!/usr/bin/env python3
"""Print the tensors seen by the Cleaner PPO policy and critic."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

# Keep the same import order as training: initialize JAX before Torch/TorchRL.
jax.devices()
jnp.asarray(0, dtype=jnp.int8).block_until_ready()

import torch
from torchrl.envs import JumanjiWrapper

from custom_cleaner import CustomCleaner, FixedObstacleGenerator
from scripts.train_cleaner_ppo import CleanerObservationBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--obs-clean-channel", action="store_true")
    parser.add_argument("--obs-other-channel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = FixedObstacleGenerator(
        height=args.height,
        width=args.width,
        num_agents=args.num_agents,
    )
    cleaner = CustomCleaner(generator=generator, time_limit=args.height * args.width)
    env = JumanjiWrapper(
        cleaner,
        batch_size=[1],
        categorical_action_encoding=True,
        jit=True,
        device="cpu",
    )
    env.set_seed(args.seed)
    td = env.reset()
    builder = CleanerObservationBuilder(
        args.height,
        args.width,
        args.num_agents,
        obs_clean_channel=args.obs_clean_channel,
        obs_other_channel=args.obs_other_channel,
    )
    (
        agent_observation,
        spatial_observation,
        action_mask,
        global_observation,
        global_spatial_observation,
        step_feature,
    ) = builder(td["grid"], td["agents_locations"], td["step_count"], td["action_mask"])

    print("raw grid constants: 0=DIRTY, 1=CLEAN, 2=WALL")
    print(td["grid"][0].to(torch.int64))
    print("agents_locations:", td["agents_locations"][0].tolist())
    print("action_mask [up,right,down,left]:")
    print(action_mask[0].to(torch.int64))
    print("flat agent_observation shape:", tuple(agent_observation.shape))
    print("spatial_observation shape:", tuple(spatial_observation.shape))
    print("global_observation shape:", tuple(global_observation.shape))
    print("global_spatial_observation shape:", tuple(global_spatial_observation.shape))
    print("step_feature:", step_feature.tolist())
    actor_labels = ["dirty", "wall"]
    if args.obs_clean_channel:
        actor_labels.append("clean")
    actor_labels.extend(["own", "all_agents"])
    if args.obs_other_channel:
        actor_labels.append("other_agents")
    critic_labels = ["dirty", "wall", "all_agents"]
    if args.obs_clean_channel:
        critic_labels.append("clean")
    for agent_idx in range(args.num_agents):
        channels = spatial_observation[0, agent_idx]
        print(f"\nagent {agent_idx} channel sums {actor_labels}:")
        print(channels.sum(dim=(-1, -2)).tolist())
        print("own channel:")
        print(channels[actor_labels.index("own")].to(torch.int64))
        print("all_agents channel:")
        print(channels[actor_labels.index("all_agents")].to(torch.int64))
    print(f"\nglobal channel sums {critic_labels}:")
    print(global_spatial_observation[0].sum(dim=(-1, -2)).tolist())


if __name__ == "__main__":
    main()
