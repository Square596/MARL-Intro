#!/usr/bin/env python3
"""Run a nearest-dirty heuristic on CustomCleaner to validate task solvability."""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from custom_cleaner import CustomCleaner, FixedObstacleGenerator

DIRTY = 0
WALL = 2
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--penalty-per-timestep", type=float, default=0.1)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--gif-path", type=Path, default=Path(".artifacts/heuristic_cleaner_rollout.gif"))
    return parser.parse_args()


def first_step_to_nearest_dirty(grid: np.ndarray, start: tuple[int, int], action_mask: np.ndarray) -> int:
    if grid[start] == DIRTY:
        valid = np.flatnonzero(action_mask)
        return int(valid[0]) if len(valid) else 0
    height, width = grid.shape
    queue: deque[tuple[tuple[int, int], int | None]] = deque([(start, None)])
    seen = {start}
    while queue:
        (row, col), first_action = queue.popleft()
        if grid[row, col] == DIRTY:
            if first_action is not None and action_mask[first_action]:
                return int(first_action)
            break
        for action, (dr, dc) in enumerate(MOVES):
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid[nr, nc] == WALL or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append(((nr, nc), action if first_action is None else first_action))
    valid = np.flatnonzero(action_mask)
    return int(valid[0]) if len(valid) else 0


def heuristic_action(timestep) -> jax.Array:
    observation = timestep.observation
    grid = np.asarray(jax.device_get(observation.grid))
    locations = np.asarray(jax.device_get(observation.agents_locations))
    action_mask = np.asarray(jax.device_get(observation.action_mask))
    actions = [
        first_step_to_nearest_dirty(grid, tuple(location.tolist()), action_mask[idx])
        for idx, location in enumerate(locations)
    ]
    return jnp.asarray(actions, dtype=jnp.int32)


def main() -> None:
    args = parse_args()
    generator = FixedObstacleGenerator(args.height, args.width, args.num_agents)
    env = CustomCleaner(
        generator=generator,
        time_limit=args.height * args.width,
        penalty_per_timestep=args.penalty_per_timestep,
    )
    key = jax.random.PRNGKey(args.seed)
    state, timestep = env.reset(key)
    states = [state]
    total_reward = 0.0
    success = False
    for _ in range(args.rollout_length):
        action = heuristic_action(timestep)
        state, timestep = env.step(state, action)
        states.append(state)
        total_reward += float(jax.device_get(timestep.reward))
        grid = np.asarray(jax.device_get(timestep.observation.grid))
        success = not bool((grid == DIRTY).any())
        if success or bool(jax.device_get(timestep.last())):
            break
    final_grid = np.asarray(jax.device_get(timestep.observation.grid))
    dirty = int((final_grid == DIRTY).sum())
    non_wall = int((final_grid != WALL).sum())
    args.gif_path.parent.mkdir(parents=True, exist_ok=True)
    env.animate(states, interval=180, save_path=str(args.gif_path))
    print(f"success: {success}")
    print(f"steps: {len(states) - 1}")
    print(f"total_reward: {total_reward:.3f}")
    print(f"final_dirty: {dirty}")
    print(f"final_dirty_fraction_non_wall: {dirty / max(non_wall, 1):.3f}")
    print(f"gif: {args.gif_path}")


if __name__ == "__main__":
    main()
