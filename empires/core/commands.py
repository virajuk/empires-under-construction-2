"""Commands: the only way anything gets into the simulation.

Both the human UI and the RL policy produce these. That is the whole point --
``World.step`` cannot tell which one it is talking to, so an agent can play the
game through exactly the interface a person does.

Every command carries ``owner``; the world validates it so a player can never
issue orders to units they do not control.
"""

from __future__ import annotations

from dataclasses import dataclass

Tile = tuple[int, int]


@dataclass(frozen=True)
class Move:
    """Walk the given units to a tile."""
    owner: int
    unit_ids: tuple[int, ...]
    target: Tile


@dataclass(frozen=True)
class Gather:
    """Harvest the resource tile at ``target``, hauling loads to a drop-off."""
    owner: int
    unit_ids: tuple[int, ...]
    target: Tile


@dataclass(frozen=True)
class Stop:
    owner: int
    unit_ids: tuple[int, ...]


Command = Move | Gather | Stop
