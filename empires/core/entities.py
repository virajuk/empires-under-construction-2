"""Game entities.

Plain mutable dataclasses, no behaviour. All rules live in ``World.step`` so
there is exactly one place that advances time -- which is what makes the sim
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .terrain import Resource

Tile = tuple[int, int]


class UnitKind(IntEnum):
    WORKER = 0
    SOLDIER = 1


class Order(IntEnum):
    IDLE = 0
    MOVE = 1
    GATHER = 2   # walking to / harvesting a resource tile
    RETURN = 3   # carrying a full load back to a drop-off building


@dataclass
class Unit:
    uid: int
    owner: int
    kind: UnitKind
    # Position in *tile* units, as floats, so movement is smooth and the
    # renderer can draw sub-tile positions without the sim knowing about pixels.
    x: float
    y: float
    hp: int = 25
    order: Order = Order.IDLE
    path: list[Tile] = field(default_factory=list)
    # Tile the current order refers to (move destination or resource tile).
    target: Tile | None = None
    carrying: int = 0
    carry_kind: Resource | None = None
    gather_timer: int = 0

    @property
    def tile(self) -> Tile:
        return int(self.x), int(self.y)


@dataclass
class Building:
    bid: int
    owner: int
    # Top-left tile; buildings occupy a w x h footprint.
    x: int
    y: int
    w: int = 2
    h: int = 2
    hp: int = 600
    is_dropoff: bool = True

    @property
    def tiles(self) -> list[Tile]:
        return [(self.x + dx, self.y + dy) for dy in range(self.h) for dx in range(self.w)]

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass
class Player:
    pid: int
    # Indexed by ``Resource``.
    resources: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def add(self, kind: Resource, amount: int) -> None:
        self.resources[int(kind)] += amount

    @property
    def total_gathered(self) -> int:
        return sum(self.resources)
