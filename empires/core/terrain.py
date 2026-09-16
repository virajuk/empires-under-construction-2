"""Terrain types and the resource they yield."""

from __future__ import annotations

from enum import IntEnum


class Terrain(IntEnum):
    GRASS = 0
    WATER = 1
    FOREST = 2
    GOLD = 3
    STONE = 4
    BERRY = 5


class Resource(IntEnum):
    # These values are indices into ``Player.resources`` and into the
    # observation's scalar vector, so they must stay distinct and contiguous
    # from 0. Duplicating a value makes IntEnum silently alias the two names
    # (``Resource.GOLD is Resource.FOOD``), which corrupts both the stockpile
    # accounting and the observation width.
    FOOD = 0
    WOOD = 1
    GOLD = 2
    STONE = 3


NUM_TERRAIN = len(Terrain)
NUM_RESOURCE = len(Resource)

# Resource tiles are *impassable* and harvested from an adjacent tile, as in
# AoE. Keeping them blocked means pathfinding naturally stops beside them.
PASSABLE: dict[Terrain, bool] = {
    Terrain.GRASS: True,
    Terrain.WATER: False,
    Terrain.FOREST: False,
    Terrain.GOLD: False,
    Terrain.STONE: False,
    Terrain.BERRY: False,
}

# Which terrain yields which resource when harvested. Absent => not harvestable.
YIELDS: dict[Terrain, Resource] = {
    Terrain.FOREST: Resource.WOOD,
    Terrain.GOLD: Resource.GOLD,
    Terrain.STONE: Resource.STONE,
    Terrain.BERRY: Resource.FOOD,
}

# Terrain that can be harvested, as a plain tuple -- handy for numpy masks
# (``np.isin(terrain, HARVESTABLE_TERRAIN)``) without rebuilding it each call.
HARVESTABLE_TERRAIN: tuple[int, ...] = tuple(int(t) for t in YIELDS)


def is_passable(t: int) -> bool:
    return PASSABLE[Terrain(t)]


def is_harvestable(t: int) -> bool:
    return Terrain(t) in YIELDS
