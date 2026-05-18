"""Cleaner environment variant used by the seminar smoke tests."""

from __future__ import annotations

from functools import cached_property

import chex
import jax
import jax.numpy as jnp

from jumanji import specs
from jumanji.environments.routing.cleaner.constants import MOVES, WALL
from jumanji.environments.routing.cleaner.env import Cleaner
from jumanji.environments.routing.cleaner.types import Observation, State
from jumanji.types import TimeStep, restart


class CustomCleaner(Cleaner):
    """Cleaner that respects custom starts and rectangular grids.

    Jumanji's upstream ``Cleaner.reset`` computes the initial action mask from
    hard-coded upper-left agent locations. Its action-mask bounds check also
    swaps rows and columns, which is harmless for square grids but wrong for
    rectangular ones. The seminar examples use explicit ``height`` and ``width``
    parameters, so this subclass keeps those two details consistent.
    """

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Observation spec with bounds shaped for TorchRL's converter."""
        grid = specs.BoundedArray(self.grid_shape, jnp.int8, 0, 2, "grid")
        min_locations = jnp.zeros((self.num_agents, 2), dtype=jnp.int32)
        max_locations = jnp.broadcast_to(
            jnp.asarray(self.grid_shape, dtype=jnp.int32),
            (self.num_agents, 2),
        )
        agents_locations = specs.BoundedArray(
            (self.num_agents, 2),
            jnp.int32,
            min_locations,
            max_locations,
            "agents_locations",
        )
        action_mask = specs.BoundedArray((self.num_agents, 4), bool, False, True, "action_mask")
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")
        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=grid,
            agents_locations=agents_locations,
            action_mask=action_mask,
            step_count=step_count,
        )

    def reset(self, key: chex.PRNGKey) -> tuple[State, TimeStep[Observation]]:
        state = self.generator(key)
        state.action_mask = self._compute_action_mask(state.grid, state.agents_locations)
        observation = self._observation_from_state(state)
        extras = self._compute_extras(state)
        return state, restart(observation, extras)

    def _compute_action_mask(self, grid: chex.Array, agents_locations: chex.Array) -> chex.Array:
        """Compute valid moves with row/column bounds in the expected order."""

        def is_move_valid(agent_location: chex.Array, move: chex.Array) -> chex.Array:
            y, x = agent_location + move
            in_bounds = (y >= 0) & (y < self.num_rows) & (x >= 0) & (x < self.num_cols)
            y_safe = jnp.clip(y, 0, self.num_rows - 1)
            x_safe = jnp.clip(x, 0, self.num_cols - 1)
            return in_bounds & (grid[y_safe, x_safe] != WALL)

        return jax.vmap(jax.vmap(is_move_valid, in_axes=(None, 0)), in_axes=(0, None))(
            agents_locations,
            MOVES,
        )
