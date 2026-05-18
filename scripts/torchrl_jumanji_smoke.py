#!/usr/bin/env python3
"""Smoke-test CustomCleaner through TorchRL's JumanjiWrapper on CUDA."""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp

# Initialize JAX before importing Torch so PyTorch's bundled cuDNN does not
# shadow the CUDA libraries used by the JAX CUDA plugin.
_JAX_DEVICES = jax.devices()
jnp.asarray(0, dtype=jnp.int8).block_until_ready()

import torch
from torchrl.envs import JumanjiWrapper

from custom_cleaner import CustomCleaner, FixedObstacleGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=8)
    return parser.parse_args()


def build_cleaner_env(height: int, width: int, num_agents: int) -> CustomCleaner:
    generator = FixedObstacleGenerator(height=height, width=width, num_agents=num_agents)
    return CustomCleaner(generator=generator, time_limit=height * width)


def shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def require_cuda() -> torch.device:
    jax_devices = _JAX_DEVICES
    torch_cuda = torch.cuda.is_available()
    print(f"jax devices: {jax_devices}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda available: {torch_cuda}")
    print(f"torch cuda version: {torch.version.cuda}")
    if not any(device.platform == "gpu" for device in jax_devices):
        raise RuntimeError("JAX did not report a CUDA/GPU device")
    if not torch_cuda:
        raise RuntimeError("PyTorch is not CUDA-enabled")
    return torch.device("cuda")


def main() -> None:
    args = parse_args()
    device = require_cuda()
    env = build_cleaner_env(args.height, args.width, args.num_agents)
    wrapped = JumanjiWrapper(
        env,
        batch_size=[args.num_envs],
        categorical_action_encoding=True,
        jit=True,
        device=device,
    )
    wrapped.set_seed(args.seed)

    print(f"height x width: {args.height} x {args.width}")
    print(f"num_agents: {args.num_agents}")
    print(f"num_envs: {args.num_envs}")
    print(f"action_spec: {wrapped.action_spec}")

    reset_td = wrapped.reset()
    print(f"reset batch_size: {reset_td.batch_size}")
    print(f"reset keys: {list(reset_td.keys(include_nested=True, leaves_only=True))}")
    print(f"grid shape: {shape_of(reset_td.get('grid'))}")
    print(f"action_mask shape: {shape_of(reset_td.get('action_mask'))}")

    step_td = wrapped.rand_step(reset_td)
    reward = step_td.get(("next", "reward"))
    done = step_td.get(("next", "done"))
    print(f"step reward shape: {shape_of(reward)}")
    print(f"step done shape: {shape_of(done)}")

    rollout_td = wrapped.rollout(args.rollout_length)
    rollout_reward = rollout_td.get(("next", "reward"))
    rollout_done = rollout_td.get(("next", "done"))
    print(f"rollout batch_size: {rollout_td.batch_size}")
    print(f"rollout reward shape: {shape_of(rollout_reward)}")
    print(f"rollout done shape: {shape_of(rollout_done)}")
    print("TorchRL Jumanji smoke test passed")


if __name__ == "__main__":
    main()
