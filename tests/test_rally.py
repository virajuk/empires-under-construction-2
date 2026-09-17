"""Rally points: where a building sends what it produces.

The rule under test is that a rally point behaves exactly like having
right-clicked the tile yourself once the unit exists -- walk there, or harvest
it if there is anything on it to harvest. What matters is that the decision is
made when the unit walks out, not when the point was set, because the tile can
be picked clean in between.
"""

import numpy as np
import pytest

from empires.config import SimConfig
from empires.core.commands import SetRally, Train
from empires.core.entities import Order, UnitKind
from empires.core.terrain import Resource, Terrain
from empires.core.world import World

TRAIN_TICKS = 100


@pytest.fixture
def world():
    w = World(SimConfig(), seed=3)
    w.players[0].add(Resource.FOOD, 1000)
    return w


def tc_of(world: World, owner: int = 0):
    return next(b for b in world.buildings.values() if b.owner == owner)


def grass_tile(world: World, owner: int = 0) -> tuple[int, int]:
    """Open ground near the Town Center, with nothing on it to gather."""
    tc = tc_of(world, owner)
    ys, xs = np.where(world.terrain == Terrain.GRASS)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    return int(xs[i]), int(ys[i])


def resource_tile(world: World, terrain: Terrain, owner: int = 0) -> tuple[int, int]:
    tc = tc_of(world, owner)
    ys, xs = np.where(world.terrain == terrain)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    return int(xs[i]), int(ys[i])


def train_one(world: World, owner: int = 0):
    """Queue a Villager and run until it pops out. Returns the new unit."""
    tc = tc_of(world, owner)
    before = set(world.units)
    world.step([Train(owner, tc.bid, UnitKind.VILLAGER)])
    for _ in range(TRAIN_TICKS + 2):
        world.step()
        new = set(world.units) - before
        if new:
            return world.units[next(iter(new))]
    raise AssertionError("no villager was produced")


# ------------------------------------------------------------------ setting


def test_a_building_starts_with_no_rally_point(world):
    assert tc_of(world).rally is None


def test_setting_a_rally_point_stores_the_tile(world):
    tc = tc_of(world)
    tile = grass_tile(world)
    world.step([SetRally(0, tc.bid, tile)])
    assert tc.rally == tile


def test_a_rally_point_can_be_moved(world):
    tc = tc_of(world)
    first, second = grass_tile(world), resource_tile(world, Terrain.FOREST)
    world.step([SetRally(0, tc.bid, first)])
    world.step([SetRally(0, tc.bid, second)])
    assert tc.rally == second


def test_you_cannot_set_a_rally_point_on_someone_elses_building(world):
    enemy = tc_of(world, owner=1)
    world.step([SetRally(0, enemy.bid, grass_tile(world))])
    assert enemy.rally is None


def test_an_out_of_bounds_rally_point_is_ignored(world):
    tc = tc_of(world)
    world.step([SetRally(0, tc.bid, (world.width + 5, world.height + 5))])
    assert tc.rally is None


# ------------------------------------------------------------- what it does


def test_without_a_rally_point_a_new_villager_stands_still(world):
    u = train_one(world)
    assert u.order is Order.IDLE
    assert u.target is None


def test_a_new_villager_walks_to_a_plain_rally_tile(world):
    tc = tc_of(world)
    tile = grass_tile(world)
    world.step([SetRally(0, tc.bid, tile)])

    u = train_one(world)
    assert u.order is Order.MOVE
    assert u.target == tile


def test_a_new_villager_gathers_a_rally_tile_that_holds_a_resource(world):
    tc = tc_of(world)
    tile = resource_tile(world, Terrain.FOREST)
    world.step([SetRally(0, tc.bid, tile)])

    u = train_one(world)
    assert u.order is Order.GATHER
    assert u.target == tile
    assert u.gather_resource is Resource.WOOD


@pytest.mark.parametrize("terrain,resource", [
    (Terrain.FOREST, Resource.WOOD),
    (Terrain.GOLD, Resource.GOLD),
    (Terrain.STONE, Resource.STONE),
    (Terrain.BERRY, Resource.FOOD),
])
def test_a_rally_point_gathers_whatever_the_tile_yields(world, terrain, resource):
    tc = tc_of(world)
    tile = resource_tile(world, terrain)
    world.step([SetRally(0, tc.bid, tile)])

    u = train_one(world)
    assert u.order is Order.GATHER
    assert u.gather_resource is resource


def test_an_exhausted_rally_tile_falls_back_to_walking(world):
    """The tile is resolved when the villager walks out, not when the point
    was set -- a bush can be picked clean in between."""
    tc = tc_of(world)
    tile = resource_tile(world, Terrain.BERRY)
    world.step([SetRally(0, tc.bid, tile)])
    world.resources[tile[1], tile[0]] = 0

    u = train_one(world)
    assert u.order is Order.MOVE
    assert u.target == tile


def test_the_rally_point_applies_to_every_unit_after_it(world):
    tc = tc_of(world)
    tile = resource_tile(world, Terrain.FOREST)
    world.step([SetRally(0, tc.bid, tile)])

    for _ in range(3):
        u = train_one(world)
        assert u.order is Order.GATHER, "the point should outlast one villager"
    assert tc.rally == tile


def test_a_rallied_villager_actually_delivers(world):
    """End to end: the point is only useful if the wood reaches the stockpile."""
    tc = tc_of(world)
    tile = resource_tile(world, Terrain.FOREST)
    world.step([SetRally(0, tc.bid, tile)])
    train_one(world)

    before = world.players[0].resources[Resource.WOOD]
    for _ in range(1200):
        world.step()
        if world.players[0].resources[Resource.WOOD] > before:
            return
    raise AssertionError("a rallied villager never delivered any wood")


# -------------------------------------------------------------- determinism


def test_the_rally_point_is_part_of_the_state_hash(world):
    """It steers every unit the building makes from here on, so two worlds
    that differ by it are not in the same state."""
    other = World(SimConfig(), seed=3)
    other.players[0].add(Resource.FOOD, 1000)
    assert world.state_hash() == other.state_hash()

    world.step([SetRally(0, tc_of(world).bid, grass_tile(world))])
    other.step()
    assert world.state_hash() != other.state_hash()
