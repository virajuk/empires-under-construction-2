"""World -> tensor.

Observations are egocentric: channel "own units" always means *the agent's*
units, whichever player that is. Train a policy as player 0 and it transfers to
player 1 unchanged, and self-play works without a second network.

The layout is deliberately a stack of spatial planes plus a small scalar
vector -- the shape a conv trunk with an MLP head expects.
"""

from __future__ import annotations

import numpy as np

from ..core.entities import Order
from ..core.terrain import NUM_RESOURCE, NUM_TERRAIN
from ..core.world import World

# Plane indices after the one-hot terrain block.
_RESOURCE_AMOUNT = NUM_TERRAIN + 0
_OWN_UNITS = NUM_TERRAIN + 1
_ENEMY_UNITS = NUM_TERRAIN + 2
_OWN_BUILDINGS = NUM_TERRAIN + 3
_ENEMY_BUILDINGS = NUM_TERRAIN + 4
_OWN_CARRYING = NUM_TERRAIN + 5
_OWN_IDLE = NUM_TERRAIN + 6

NUM_CHANNELS = NUM_TERRAIN + 7
NUM_SCALARS = NUM_RESOURCE + 2  # stockpile, episode progress, idle fraction

# Normalisers. Grid values land roughly in [0, 1] so the network sees a
# consistent scale regardless of map or economy tuning.
_MAX_TILE_RESOURCE = 500.0
_MAX_STOCKPILE = 2000.0
_MAX_UNITS_PER_TILE = 4.0


def encode_grid(world: World, player: int, out: np.ndarray | None = None) -> np.ndarray:
    """``(C, H, W)`` float32 planes describing the map from ``player``'s view."""
    h, w = world.height, world.width
    if out is None:
        out = np.zeros((NUM_CHANNELS, h, w), dtype=np.float32)
    else:
        out.fill(0.0)

    # Terrain one-hot.
    for t in range(NUM_TERRAIN):
        out[t] = world.terrain == t

    out[_RESOURCE_AMOUNT] = np.minimum(world.resources / _MAX_TILE_RESOURCE, 1.0)

    for uid in sorted(world.units):
        u = world.units[uid]
        tx, ty = u.tile
        if not (0 <= tx < w and 0 <= ty < h):
            continue
        if u.owner == player:
            out[_OWN_UNITS, ty, tx] += 1.0 / _MAX_UNITS_PER_TILE
            if u.carrying:
                out[_OWN_CARRYING, ty, tx] = u.carrying / max(
                    1, world.cfg.villager_carry_capacity
                )
            if u.order is Order.IDLE:
                out[_OWN_IDLE, ty, tx] += 1.0 / _MAX_UNITS_PER_TILE
        else:
            out[_ENEMY_UNITS, ty, tx] += 1.0 / _MAX_UNITS_PER_TILE

    for b in world.buildings.values():
        plane = _OWN_BUILDINGS if b.owner == player else _ENEMY_BUILDINGS
        out[plane, b.y : b.y + b.h, b.x : b.x + b.w] = 1.0

    np.clip(out, 0.0, 1.0, out=out)
    return out


def encode_scalars(world: World, player: int) -> np.ndarray:
    """Global state a convolution cannot see: stockpile, clock, idle villagers.

    The idle fraction comes from ``World.villager_activity`` -- the same call
    the HUD uses -- so what the policy is told and what a person sees on screen
    cannot drift apart. It counts Villagers only: a Soldier standing still is
    not idle economy.
    """
    act = world.villager_activity(player)
    return np.array(
        [
            *(min(r / _MAX_STOCKPILE, 1.0) for r in world.players[player].resources),
            world.tick / max(1, world.cfg.max_ticks),
            act.idle / max(1, act.total),
        ],
        dtype=np.float32,
    )
