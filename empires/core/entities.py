"""Game entities and the vocabulary the game is written in.

Plain mutable dataclasses, no behaviour. All rules live in ``World.step`` so
there is exactly one place that advances time -- which is what makes the sim
reproducible.

The names here are the game's domain language: a resource gatherer is a
**Villager**, the main building is a **Town Center**. Display strings live on
the specs below rather than being typed out at each call site, so the HUD, logs
and any future tooltip all read the same word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .terrain import Resource

Tile = tuple[int, int]


class UnitKind(IntEnum):
    VILLAGER = 0
    SOLDIER = 1


class BuildingKind(IntEnum):
    TOWN_CENTER = 0


class Order(IntEnum):
    IDLE = 0
    MOVE = 1
    GATHER = 2   # walking to / harvesting a resource tile
    RETURN = 3   # carrying a full load back to a drop-off building


def costs(food: int = 0, wood: int = 0, gold: int = 0, stone: int = 0) -> tuple[int, ...]:
    """Build a cost vector indexed by :class:`Resource`.

    Written out by index rather than as a bare ``(50, 0, 0, 0)`` literal so the
    numbers stay attached to their resource if the enum ever grows.
    """
    out = [0] * len(Resource)
    out[Resource.FOOD] = food
    out[Resource.WOOD] = wood
    out[Resource.GOLD] = gold
    out[Resource.STONE] = stone
    return tuple(out)


NO_COST: tuple[int, ...] = costs()


@dataclass(frozen=True)
class UnitSpec:
    label: str
    plural: str
    hp: int
    can_gather: bool
    # What it takes to produce one, and how long. A unit with no producer
    # (``train_ticks`` 0 and absent from every BuildingSpec.trains) simply
    # cannot be made yet.
    cost: tuple[int, ...] = NO_COST
    train_ticks: int = 0


@dataclass(frozen=True)
class BuildingSpec:
    label: str
    plural: str
    width: int
    height: int
    hp: int
    # Whether villagers can deliver a load here. Town Centers accept
    # everything; a future Mill or Mining Camp would accept a subset.
    is_dropoff: bool
    # Units this building can produce. One source of truth: the UI asks it what
    # to offer, and the simulation asks it whether an order is legal.
    trains: tuple[UnitKind, ...] = ()


UNIT_SPECS: dict[UnitKind, UnitSpec] = {
    UnitKind.VILLAGER: UnitSpec(
        "Villager", "Villagers", hp=25, can_gather=True,
        cost=costs(food=50), train_ticks=100,
    ),
    UnitKind.SOLDIER: UnitSpec("Soldier", "Soldiers", hp=45, can_gather=False),
}

BUILDING_SPECS: dict[BuildingKind, BuildingSpec] = {
    BuildingKind.TOWN_CENTER: BuildingSpec(
        "Town Center", "Town Centers", width=2, height=2, hp=600, is_dropoff=True,
        trains=(UnitKind.VILLAGER,),
    ),
}

# How many units may be waiting at one building. Bounded so a misclick -- or an
# agent spamming the train action -- cannot drain a stockpile into a queue that
# takes minutes to clear.
MAX_QUEUE = 10


def unit_label(kind: UnitKind, count: int = 1) -> str:
    spec = UNIT_SPECS[kind]
    return spec.label if count == 1 else spec.plural


def building_label(kind: BuildingKind, count: int = 1) -> str:
    spec = BUILDING_SPECS[kind]
    return spec.label if count == 1 else spec.plural


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
    # What this villager was told to collect. Outlives any one tile, so when a
    # bush is picked clean it knows to go looking for another bush rather than
    # standing there -- the target tile cannot say, because an exhausted tile
    # has already reverted to grass.
    gather_resource: Resource | None = None

    @property
    def tile(self) -> Tile:
        return int(self.x), int(self.y)

    @property
    def spec(self) -> UnitSpec:
        return UNIT_SPECS[self.kind]

    @property
    def label(self) -> str:
        return self.spec.label


@dataclass
class Building:
    bid: int
    owner: int
    kind: BuildingKind = BuildingKind.TOWN_CENTER
    # Top-left tile; buildings occupy a w x h footprint.
    x: int = 0
    y: int = 0
    w: int = 2
    h: int = 2
    hp: int = 600
    is_dropoff: bool = True
    # Production. ``queue`` holds what is waiting to come out, front first;
    # ``train_timer`` counts ticks spent on ``queue[0]``.
    queue: list[UnitKind] = field(default_factory=list)
    train_timer: int = 0

    @property
    def spec(self) -> BuildingSpec:
        return BUILDING_SPECS[self.kind]

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def tiles(self) -> list[Tile]:
        return [(self.x + dx, self.y + dy) for dy in range(self.h) for dx in range(self.w)]

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass
class Player:
    pid: int
    # Indexed by ``Resource``.
    resources: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def add(self, kind: Resource, amount: int) -> None:
        self.resources[int(kind)] += amount

    def can_afford(self, cost: tuple[int, ...]) -> bool:
        return all(have >= need for have, need in zip(self.resources, cost))

    def spend(self, cost: tuple[int, ...]) -> None:
        for i, need in enumerate(cost):
            self.resources[i] -= need

    def refund(self, cost: tuple[int, ...]) -> None:
        for i, amount in enumerate(cost):
            self.resources[i] += amount

    @property
    def total_gathered(self) -> int:
        return sum(self.resources)
