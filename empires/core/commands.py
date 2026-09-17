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


@dataclass(frozen=True)
class Train:
    """Queue a unit at a building. Cost is charged when it joins the queue."""
    owner: int
    building_id: int
    unit_kind: int


@dataclass(frozen=True)
class CancelTrain:
    """Drop the last queued unit at a building and refund it."""
    owner: int
    building_id: int


@dataclass(frozen=True)
class SetRally:
    """Point a building's output at a tile.

    Every unit it finishes from now on walks there -- and harvests it, if
    there is anything on it to harvest. The tile is kept as a tile, not as a
    resolved order: a bush can be picked clean between setting the point and
    the next villager walking out of the door.
    """
    owner: int
    building_id: int
    target: Tile


Command = Move | Gather | Stop | Train | CancelTrain | SetRally
