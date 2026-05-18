"""Deterministic, easy-to-read Cleaner generator for seminar demos."""

from __future__ import annotations

import chex
import jax.numpy as jnp

from jumanji.environments.routing.cleaner.constants import CLEAN, DIRTY, WALL
from jumanji.environments.routing.cleaner.generator import Generator
from jumanji.environments.routing.cleaner.types import State


class FixedObstacleGenerator(Generator):
    """Create a simple fixed Cleaner layout with proportional wall positions.

    The layout is deterministic but parameterized by grid size. This keeps class
    demos reproducible while avoiding the visual complexity of maze generation.
    """

    def __init__(self, height: int = 8, width: int = 8, num_agents: int = 3) -> None:
        if height < 6:
            raise ValueError("height must be at least 6 for the fixed wall layout")
        if width < 6:
            raise ValueError("width must be at least 6 for the fixed wall layout")
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        super().__init__(num_rows=height, num_cols=width, num_agents=num_agents)
        self.height = height
        self.width = width

    def __call__(self, key: chex.PRNGKey) -> State:
        grid = jnp.full((self.height, self.width), DIRTY, dtype=jnp.int8)
        grid = self._add_walls(grid)
        agents_locations = self._agent_start_locations()
        grid = grid.at[agents_locations[:, 0], agents_locations[:, 1]].set(CLEAN)
        return State(
            grid=grid,
            agents_locations=agents_locations,
            action_mask=None,
            step_count=jnp.array(0, dtype=jnp.int32),
            key=key,
        )

    def _add_walls(self, grid: chex.Array) -> chex.Array:
        """Add two horizontal walls and one short vertical wall.

        All indices are computed from margins/proportions, so the same generator
        works for different grid sizes. Slices are Python-static, which keeps the
        generator compatible with ``jax.jit(env.reset)``.
        """
        upper_row = max(1, self.height // 3)
        lower_row = min(self.height - 2, (2 * self.height) // 3)
        vertical_col = min(self.width - 2, max(2, (3 * self.width) // 4))

        upper_start = max(1, self.width // 5)
        upper_end = min(self.width - 1, max(upper_start + 2, (4 * self.width) // 5))

        lower_start = max(1, self.width // 4)
        lower_end = self.width - 1

        vertical_start = max(1, self.height // 5)
        vertical_end = min(self.height - 1, max(vertical_start + 2, self.height // 2))

        grid = grid.at[upper_row, upper_start:upper_end].set(WALL)
        grid = grid.at[lower_row, lower_start:lower_end].set(WALL)
        grid = grid.at[vertical_start:vertical_end, vertical_col].set(WALL)
        return grid

    def _agent_start_locations(self) -> chex.Array:
        """Spread agents across the top row so rollouts are easy to read."""
        if self.num_agents == 1:
            cols = jnp.array([0], dtype=jnp.int32)
        else:
            agent_ids = jnp.arange(self.num_agents, dtype=jnp.float32)
            cols = jnp.rint(agent_ids * (self.width - 1) / (self.num_agents - 1))
            cols = cols.astype(jnp.int32)
        rows = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        return jnp.stack([rows, cols], axis=-1)
