"""Villagers move on to the next patch when their tile runs dry.

A villager keeps collecting whatever it was told to collect. When the bush it
is working is picked clean it finds the nearest remaining bush and carries on,
rather than standing idle until the player notices.
"""

import numpy as np
import pytest

from empires.config import SimConfig
from empires.core.commands import Gather, Move, Stop
from empires.core.entities import Order
from empires.core.terrain import TERRAIN_FOR_RESOURCE, Resource, Terrain
from empires.core.world import World


@pytest.fixture
def world():
    return World(SimConfig(), seed=3)


def tc_of(world: World, owner: int = 0):
    return next(b for b in world.buildings.values() if b.owner == owner)


def nearest_of(world: World, terrain: Terrain, owner: int = 0) -> tuple[int, int]:
    ys, xs = np.where(world.terrain == terrain)
    tc = tc_of(world, owner)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    return int(xs[i]), int(ys[i])


def clear_resource(world: World, resource: Resource) -> None:
    world.resources[np.isin(world.terrain, TERRAIN_FOR_RESOURCE[resource])] = 0


def run(world: World, ticks: int, commands=None):
    world.step(commands)
    for _ in range(ticks - 1):
        world.step()


# ------------------------------------------------------------- assignment


def test_a_gather_order_records_what_to_collect(world):
    berry = nearest_of(world, Terrain.BERRY)
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])
    assert unit.gather_resource is Resource.FOOD


def test_a_move_order_clears_the_assignment(world):
    berry = nearest_of(world, Terrain.BERRY)
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])
    world.step([Move(0, (unit.uid,), world.nearest_free_tile(30, 20))])
    assert unit.gather_resource is None


def test_stop_clears_the_assignment(world):
    berry = nearest_of(world, Terrain.BERRY)
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])
    world.step([Stop(0, (unit.uid,))])
    assert unit.gather_resource is None


# ------------------------------------------------------------ retargeting


def test_villager_moves_to_the_next_bush_when_one_runs_dry(world):
    """The behaviour asked for: keep gathering food, not stand around."""
    berry = nearest_of(world, Terrain.BERRY)
    world.resources[berry[1], berry[0]] = 2  # nearly empty
    unit = world.units_of(0)[0]

    world.step([Gather(0, (unit.uid,), berry)])
    run(world, 800)

    assert world.terrain[berry[1], berry[0]] == Terrain.GRASS, "first bush not cleared"
    assert unit.order is not Order.IDLE, "villager stopped instead of moving on"
    assert unit.gather_resource is Resource.FOOD
    assert unit.target != berry
    assert world.players[0].resources[Resource.FOOD] > 2, "never reached a second bush"


def test_the_next_target_is_the_same_resource(world):
    """A food gatherer must not wander onto wood."""
    berry = nearest_of(world, Terrain.BERRY)
    world.resources[berry[1], berry[0]] = 1
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])

    for _ in range(900):
        world.step()
        if unit.target is not None and unit.target != berry:
            tx, ty = unit.target
            assert world.terrain[ty, tx] == Terrain.BERRY
    assert world.players[0].resources[Resource.WOOD] == 0


def test_it_picks_the_nearest_remaining_tile(world):
    """Two bushes left: it should take the closer one."""
    clear_resource(world, Resource.FOOD)
    unit = world.units_of(0)[0]
    origin = unit.tile

    near = world.nearest_free_tile(origin[0] + 4, origin[1])
    far = world.nearest_free_tile(origin[0] + 14, origin[1])
    for tile in (near, far):
        world.terrain[tile[1], tile[0]] = Terrain.BERRY
        world.resources[tile[1], tile[0]] = 100
    world._blocked = None

    found = world.nearest_resource_tile(origin, Resource.FOOD)
    assert found == near


def test_a_villager_already_walking_redirects_if_the_bush_is_taken(world):
    """Someone else finished it first -- head for another rather than idle."""
    berry = nearest_of(world, Terrain.BERRY)
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])
    world.step()
    assert unit.order is Order.GATHER

    # Empty it out from under them before they arrive.
    world.resources[berry[1], berry[0]] = 0
    world.terrain[berry[1], berry[0]] = Terrain.GRASS
    world._blocked = None
    world.step()

    assert unit.order is Order.GATHER
    assert unit.target != berry


def test_a_full_load_is_delivered_before_moving_on(world):
    """Retargeting must not throw away resources already carried."""
    berry = nearest_of(world, Terrain.BERRY)
    world.resources[berry[1], berry[0]] = 3
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), berry)])

    for _ in range(900):
        world.step()
        if unit.carrying == 0 and world.players[0].resources[Resource.FOOD] >= 3:
            break
    assert world.players[0].resources[Resource.FOOD] >= 3, "the partial load was lost"


def test_villager_stops_when_the_resource_is_gone(world):
    berry = nearest_of(world, Terrain.BERRY)
    clear_resource(world, Resource.FOOD)
    world.resources[berry[1], berry[0]] = 2
    unit = world.units_of(0)[0]

    world.step([Gather(0, (unit.uid,), berry)])
    run(world, 900)
    assert unit.order is Order.IDLE
    assert unit.gather_resource is None
    assert unit.carry_kind is None


def test_search_is_bounded_by_the_configured_radius():
    cfg = SimConfig(regather_radius=5)
    w = World(cfg, seed=3)
    clear_resource(w, Resource.FOOD)
    origin = w.units_of(0)[0].tile

    far = w.nearest_free_tile(origin[0] + 15, origin[1])
    w.terrain[far[1], far[0]] = Terrain.BERRY
    w.resources[far[1], far[0]] = 100
    w._blocked = None

    assert w.nearest_resource_tile(origin, Resource.FOOD) is None
    assert w.nearest_resource_tile(origin, Resource.FOOD, max_radius=30) == far


def test_unreachable_tiles_are_skipped(world):
    """The nearest bush as the crow flies may be across water."""
    clear_resource(world, Resource.FOOD)
    unit = world.units_of(0)[0]
    origin = unit.tile

    # A bush walled in by water, nearer than a reachable one.
    walled = (origin[0] + 5, origin[1])
    world.terrain[walled[1] - 1:walled[1] + 2, walled[0] - 1:walled[0] + 2] = Terrain.WATER
    world.terrain[walled[1], walled[0]] = Terrain.BERRY
    world.resources[walled[1], walled[0]] = 100

    reachable = world.nearest_free_tile(origin[0] + 10, origin[1])
    world.terrain[reachable[1], reachable[0]] = Terrain.BERRY
    world.resources[reachable[1], reachable[0]] = 100
    world._blocked = None

    assert world.nearest_resource_tile(origin, Resource.FOOD) == reachable


def test_no_resource_anywhere_returns_none(world):
    clear_resource(world, Resource.FOOD)
    assert world.nearest_resource_tile((10, 10), Resource.FOOD) is None


# ----------------------------------------------------------- consistency


def test_retargeting_stays_deterministic():
    a, b = World(SimConfig(), seed=5), World(SimConfig(), seed=5)
    for w in (a, b):
        ys, xs = np.where(w.terrain == Terrain.BERRY)
        tc = tc_of(w)
        i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
        target = (int(xs[i]), int(ys[i]))
        w.resources[target[1], target[0]] = 3
        w.step([Gather(0, tuple(sorted(u.uid for u in w.units_of(0))), target)])
    for _ in range(900):
        a.step()
        b.step()
    assert a.state_hash() == b.state_hash()


def test_a_crew_freed_together_stays_on_the_same_patch(world):
    """Each villager takes the tile nearest *itself*, so a crew spreads across
    neighbouring bushes rather than all queueing on one -- but it stays put,
    it does not scatter across the map."""
    berry = nearest_of(world, Terrain.BERRY)
    world.resources[berry[1], berry[0]] = 4
    ids = tuple(sorted(u.uid for u in world.units_of(0))[:2])
    world.step([Gather(0, ids, berry)])
    run(world, 900)

    targets = [world.units[uid].target for uid in ids]
    assert all(t is not None and t != berry for t in targets)
    for tx, ty in targets:
        assert world.terrain[ty, tx] == Terrain.BERRY
    spread = max(
        max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a in targets for b in targets
    )
    assert spread <= 4, f"crew scattered across {targets}"


def test_gathering_continues_far_past_a_single_tile(world):
    """End to end: one order keeps a crew fed for a long stretch."""
    berry = nearest_of(world, Terrain.BERRY)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, berry)])
    run(world, 3000)

    food = world.players[0].resources[Resource.FOOD]
    assert food > world.cfg.berry_amount, (
        f"only {food} food: the crew stopped at the first bush"
    )


def test_search_matches_a_brute_force_reference(world):
    """The scan is cropped to a box for speed. A Chebyshev radius is a square,
    so that must select exactly the same tiles as filtering the whole map."""
    from empires.core.pathfinding import path_to_adjacent
    from empires.core.world import RETARGET_CANDIDATES

    def reference(w: World, origin, resource, radius):
        cands = []
        for y in range(w.height):
            for x in range(w.width):
                if w.resources[y, x] <= 0:
                    continue
                if int(w.terrain[y, x]) not in TERRAIN_FOR_RESOURCE[resource]:
                    continue
                d = max(abs(x - origin[0]), abs(y - origin[1]))
                if radius and d > radius:
                    continue
                cands.append((d, y, x))
        cands.sort()
        for _, y, x in cands[:RETARGET_CANDIDATES]:
            if path_to_adjacent(w.blocked, origin, (x, y)) is not None:
                return (x, y)
        return None

    rng = np.random.default_rng(0)
    for _ in range(25):
        origin = (int(rng.integers(0, world.width)), int(rng.integers(0, world.height)))
        for resource in (Resource.FOOD, Resource.WOOD, Resource.GOLD):
            for radius in (5, 24, 0):
                assert world.nearest_resource_tile(origin, resource, radius) == \
                    reference(world, origin, resource, radius), (
                        f"mismatch at {origin} for {resource.name} r={radius}"
                    )
