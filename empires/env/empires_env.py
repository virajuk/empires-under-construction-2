"""Gymnasium-style environment wrapping :class:`~empires.core.world.World`.

Gymnasium is optional: if it is installed the env subclasses ``gym.Env`` and
exposes real spaces, otherwise it falls back to a plain object with the same
``reset``/``step`` signature. Either way the semantics are the standard
five-tuple, so ``pip install gymnasium`` later changes nothing about your code.

Action space
------------
``MultiDiscrete([max_units + 1, W, H])`` -- ``(unit_slot, x, y)``.

* ``unit_slot == 0`` is a no-op. Slots ``1..N`` index the agent's own units
  sorted by id, so slot identity is stable across a step.
* The order issued is the same "smart order" a right-click produces: harvest
  the tile if it holds a resource, otherwise walk to it.

This is intentionally the simplest action space that can express a real
economy, and it is the first thing you should replace. See the notes at the
bottom of the file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import SimConfig
from ..core.commands import Command, Gather, Move
from ..core.terrain import is_harvestable
from ..core.world import World
from . import observations as obs

try:  # pragma: no cover - exercised by whichever branch is installed
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _Base = object  # type: ignore[assignment,misc]
    spaces = None  # type: ignore[assignment]
    _HAS_GYM = False


class EmpiresEnv(_Base):  # type: ignore[misc,valid-type]
    """Single-agent resource-gathering task on the RTS simulation."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        sim_cfg: SimConfig | None = None,
        player: int = 0,
        max_units: int = 32,
        ticks_per_action: int = 4,
        reward_scale: float = 0.01,
        idle_penalty: float = 0.0,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.sim_cfg = sim_cfg or SimConfig()
        self.player = player
        self.max_units = max_units
        # Frame skip. The sim is cheap, the policy is not -- running 4 ticks per
        # decision roughly quadruples throughput at almost no cost in control.
        self.ticks_per_action = ticks_per_action
        self.reward_scale = reward_scale
        self.idle_penalty = idle_penalty
        self.render_mode = render_mode

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.world: World | None = None
        self._grid_buf = np.zeros(
            (obs.NUM_CHANNELS, self.sim_cfg.map_height, self.sim_cfg.map_width),
            dtype=np.float32,
        )
        self._renderer: Any = None

        if _HAS_GYM:
            w, h = self.sim_cfg.map_width, self.sim_cfg.map_height
            self.observation_space = spaces.Dict(
                {
                    "grid": spaces.Box(0.0, 1.0, (obs.NUM_CHANNELS, h, w), np.float32),
                    "scalars": spaces.Box(0.0, 1.0, (obs.NUM_SCALARS,), np.float32),
                }
            )
            self.action_space = spaces.MultiDiscrete([max_units + 1, w, h])

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        # A fresh map per episode unless a seed was pinned. Training on one
        # fixed map is the single easiest way to fool yourself about progress.
        world_seed = self._seed if self._seed is not None else int(self._rng.integers(2**31))
        self.world = World(self.sim_cfg, seed=world_seed)
        return self._observe(), {"world_seed": world_seed}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.world is not None, "call reset() before step()"
        world = self.world

        commands = self._decode_action(np.asarray(action).reshape(-1))

        gathered = 0
        for i in range(self.ticks_per_action):
            report = world.step(commands if i == 0 else None)
            gathered += report.gathered[self.player]

        reward = gathered * self.reward_scale
        if self.idle_penalty:
            own = world.units_of(self.player)
            if own:
                idle = sum(1 for u in own if int(u.order) == 0)
                reward -= self.idle_penalty * idle / len(own)

        truncated = world.tick >= self.sim_cfg.max_ticks
        terminated = not world.units_of(self.player)  # lost every villager

        info = {
            "tick": world.tick,
            "resources": list(world.players[self.player].resources),
            "gathered": gathered,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.world is None:
            return None
        return self._render_rgb()

    def close(self) -> None:
        if self._renderer is not None:
            import pygame

            pygame.quit()
            self._renderer = None

    # ------------------------------------------------------------ internals

    def _observe(self) -> dict[str, np.ndarray]:
        assert self.world is not None
        return {
            "grid": obs.encode_grid(self.world, self.player, out=self._grid_buf),
            "scalars": obs.encode_scalars(self.world, self.player),
        }

    def controllable_units(self) -> list[int]:
        """Unit ids addressable by action slots 1..N, in slot order."""
        assert self.world is not None
        ids = sorted(u.uid for u in self.world.units_of(self.player))
        return ids[: self.max_units]

    def _decode_action(self, action: np.ndarray) -> list[Command]:
        assert self.world is not None
        slot, x, y = int(action[0]), int(action[1]), int(action[2])
        if slot == 0:
            return []

        ids = self.controllable_units()
        if slot - 1 >= len(ids):
            return []  # slot points past the current army; treat as no-op
        uid = ids[slot - 1]

        tile = (x, y)
        if not self.world.in_bounds(tile):
            return []

        if is_harvestable(self.world.terrain[y, x]) and self.world.resources[y, x] > 0:
            return [Gather(self.player, (uid,), tile)]
        return [Move(self.player, (uid,), tile)]

    def _render_rgb(self) -> np.ndarray:
        """Offscreen render, for videos and for eyeballing a policy."""
        import pygame

        from ..config import RenderConfig
        from ..render.camera import Camera
        from ..render.renderer import Renderer

        assert self.world is not None
        if self._renderer is None:
            pygame.init()
            cfg = RenderConfig()
            surface = pygame.Surface(
                (self.world.width * cfg.tile_size, self.world.height * cfg.tile_size)
            )
            camera = Camera(
                self.world.width, self.world.height, cfg.tile_size,
                surface.get_width(), surface.get_height(),
            )
            self._renderer = (surface, Renderer(surface, camera, cfg))

        surface, renderer = self._renderer
        renderer.draw(self.world, set(), self.player)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


# --------------------------------------------------------------------------
# Where to go next with the action space
#
# One unit per step is fine for a first agent but it scales badly: with 50
# villagers the policy spends 50 decisions issuing one round of orders. The usual
# progressions, roughly in order of effort:
#
#   1. Auto-assign: add a single ACTION_TYPE "send all idle villagers to the
#      nearest <resource>". Collapses the economy to a handful of decisions and
#      is often enough to get a first learning curve.
#   2. Spatial action head: predict a (x, y) heatmap plus a unit-selection
#      distribution, as in the AlphaStar / PySC2 formulation.
#   3. Per-unit policy with shared weights: treat each unit as an agent and
#      batch them, which is what most modern RTS agents do.
#
# Whichever you pick, keep the rule that the env emits `Command` objects. As
# long as that holds, the human UI and the agent stay interchangeable.
# --------------------------------------------------------------------------
