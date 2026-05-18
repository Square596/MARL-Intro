#!/usr/bin/env python3
"""Smoke-test Jumanji Cleaner, vectorization, and GIF rendering.

This is intentionally small and seminar-oriented. It answers three questions:

1. Can we import and step a custom Cleaner environment?
2. Can we parallelize environments with jax.vmap?
3. Can we render a short trajectory to a GIF?
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

# Use a non-interactive backend before Jumanji creates matplotlib viewers.
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
from jumanji.wrappers import VmapWrapper

from custom_cleaner import CustomCleaner, FixedObstacleGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=Path(".artifacts/cleaner_rollout.gif"),
        help="Where to save the rendered single-environment GIF.",
    )
    return parser.parse_args()


def build_cleaner_env(height: int, width: int, num_agents: int) -> CustomCleaner:
    generator = FixedObstacleGenerator(height=height, width=width, num_agents=num_agents)
    return CustomCleaner(generator=generator, time_limit=height * width)


def sample_legal_actions(key: jax.Array, action_mask: jax.Array) -> jax.Array:
    """Sample one valid Cleaner action per agent.

    Cleaner expects an action array of shape `(num_agents,)`. The observation
    provides `action_mask` with shape `(num_agents, 4)`, one legal-action mask
    per agent.
    """
    logits = jnp.where(action_mask, 0.0, -1.0e9)
    return jax.random.categorical(key, logits=logits, axis=-1).astype(jnp.int32)


def make_single_env_gif(env: Cleaner, states: Sequence[object], gif_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    env.animate(states, interval=200, save_path=str(gif_path))


def main() -> None:
    args = parse_args()
    env = build_cleaner_env(args.height, args.width, args.num_agents)

    key = jax.random.PRNGKey(args.seed)
    key, reset_key, action_key = jax.random.split(key, 3)

    state, timestep = jax.jit(env.reset)(reset_key)
    print("Custom Cleaner imported")
    print(f"jax devices: {jax.devices()}")
    print(f"height x width: {env.num_rows} x {env.num_cols}")
    print(f"num_agents: {env.num_agents}")
    print(f"single grid shape: {tuple(timestep.observation.grid.shape)}")
    print(f"single action_mask shape: {tuple(timestep.observation.action_mask.shape)}")
    print("fixed grid layout (0=dirty, 1=clean, 2=wall):")
    print(jax.device_get(timestep.observation.grid))

    action = sample_legal_actions(action_key, timestep.observation.action_mask)
    _, next_timestep = jax.jit(env.step)(state, action)
    print(f"single sampled action: {jax.device_get(action).tolist()}")
    print(f"single next reward: {float(jax.device_get(next_timestep.reward)):.3f}")
    print(f"single done after one step: {bool(jax.device_get(next_timestep.last()))}")

    # Direct JAX vectorization: batch reset keys and vmap reset/step.
    key, reset_batch_key, action_batch_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_batch_key, args.num_envs)
    batched_state, batched_timestep = jax.vmap(env.reset)(reset_keys)
    action_keys = jax.random.split(action_batch_key, args.num_envs)
    batched_actions = jax.vmap(sample_legal_actions)(
        action_keys,
        batched_timestep.observation.action_mask,
    )
    _, batched_next_timestep = jax.vmap(env.step)(
        batched_state,
        batched_actions,
    )
    print("\nPlain jax.vmap demo")
    print(f"batched grid shape: {tuple(batched_timestep.observation.grid.shape)}")
    print(f"batched action shape: {tuple(batched_actions.shape)}")
    print(f"batched reward shape: {tuple(batched_next_timestep.reward.shape)}")

    # Jumanji wrapper version: same first-dimension batching, nicer env-like API.
    vmap_env = VmapWrapper(build_cleaner_env(args.height, args.width, args.num_agents))
    wrapped_state, wrapped_timestep = vmap_env.reset(reset_keys)
    wrapped_actions = jax.vmap(sample_legal_actions)(
        action_keys,
        wrapped_timestep.observation.action_mask,
    )
    _, wrapped_next_timestep = vmap_env.step(wrapped_state, wrapped_actions)
    print("\nJumanji VmapWrapper demo")
    print(f"wrapped grid shape: {tuple(wrapped_timestep.observation.grid.shape)}")
    print(f"wrapped reward shape: {tuple(wrapped_next_timestep.reward.shape)}")

    # Render a single unbatched rollout. Rendering batched states is possible via
    # VmapWrapper.render, but it intentionally renders only the first env.
    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)
    states = [state]
    for _ in range(args.rollout_length):
        key, action_key = jax.random.split(key)
        action = sample_legal_actions(action_key, timestep.observation.action_mask)
        state, timestep = env.step(state, action)
        states.append(state)
        if bool(jax.device_get(timestep.last())):
            break

    make_single_env_gif(env, states, args.gif_path)
    print(f"\nGIF saved to: {args.gif_path}")
    print(f"frames: {len(states)}")


if __name__ == "__main__":
    main()
