import numpy as np
import pytest

from empires.config import SimConfig
from empires.core.commands import Gather, Move, Stop
from empires.core.entities import Order
from empires.core.terrain import Resource, Terrain
from empires.core.world import World


@pytest.fixture
def world():
    return World(SimConfig(), seed=3)


def nearest_forest(world: World, owner: int) -> tuple[int, int]:
    ys, xs = np.where(world.terrain == Terrain.FOREST)
    tc = next(b for b in world.buildings.values() if b.owner == owner)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    return int(xs[i]), int(ys[i])


def run(world: World, ticks: int, commands=None):
    world.step(commands)
    for _ in range(ticks - 1):
        world.step()


# --------------------------------------------------------------- determinism


def test_same_seed_same_initial_state():
    a, b = World(seed=11), World(seed=11)
    assert a.state_hash() == b.state_hash()


def test_different_seeds_differ():
    assert World(seed=1).state_hash() != World(seed=2).state_hash()


def test_identical_command_streams_stay_in_sync():
    """The property the whole RL setup rests on.

    Two worlds fed the same seed and the same commands must be bit-identical
    hundreds of ticks later. If this ever fails, something non-deterministic
    crept into the sim and replays, debugging and off-policy training all break.
    """
    a, b = World(seed=5), World(seed=5)
    target = nearest_forest(a, 0)
    cmd = [Gather(0, tuple(sorted(u.uid for u in a.units_of(0))), target)]

    a.step(cmd)
    b.step([Gather(0, tuple(sorted(u.uid for u in b.units_of(0))), target)])
    for _ in range(400):
        a.step()
        b.step()
    assert a.state_hash() == b.state_hash()


# ---------------------------------------------------------------- gathering


def test_full_gather_cycle_deposits_resources(world):
    target = nearest_forest(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    assert world.players[0].resources[Resource.WOOD] > 0


def test_exhausted_tile_becomes_walkable(world):
    target = nearest_forest(world, 0)
    tx, ty = target
    world.resources[ty, tx] = 3
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    assert world.terrain[ty, tx] == Terrain.GRASS
    assert not world.blocked[ty, tx]
    assert world.players[0].resources[Resource.WOOD] == 3


def test_gather_on_empty_tile_is_ignored(world):
    target = nearest_forest(world, 0)
    world.resources[target[1], target[0]] = 0
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), target)])
    assert unit.order is Order.IDLE


def test_workers_do_not_exceed_carry_capacity():
    cfg = SimConfig(worker_carry_capacity=5, gather_ticks_per_unit=1)
    w = World(cfg, seed=3)
    target = nearest_forest(w, 0)
    ids = tuple(sorted(u.uid for u in w.units_of(0)))
    w.step([Gather(0, ids, target)])
    for _ in range(500):
        w.step()
        assert all(u.carrying <= cfg.worker_carry_capacity for u in w.units.values())


# ----------------------------------------------------------------- commands


def test_cannot_command_another_players_units(world):
    enemy = world.units_of(1)[0]
    before = (enemy.x, enemy.y, enemy.order)
    world.step([Move(0, (enemy.uid,), (20, 20))])
    assert (enemy.x, enemy.y, enemy.order) == before


def test_move_reaches_its_destination(world):
    from empires.core.pathfinding import find_path

    unit = world.units_of(0)[0]
    # Pick the nearest free tile a few steps away that is provably reachable,
    # so the test measures movement rather than map luck.
    ys, xs = np.where(~world.blocked)
    order = np.argsort(np.abs(xs - unit.x - 6) + np.abs(ys - unit.y - 4))
    dest = next(
        (int(xs[i]), int(ys[i]))
        for i in order
        if find_path(world.blocked, unit.tile, (int(xs[i]), int(ys[i]))) not in (None, [])
    )
    run(world, 800, [Move(0, (unit.uid,), dest)])
    assert unit.tile == dest
    assert unit.order is Order.IDLE


def test_stop_clears_orders(world):
    target = nearest_forest(world, 0)
    unit = world.units_of(0)[0]
    world.step([Gather(0, (unit.uid,), target)])
    assert unit.order is not Order.IDLE
    world.step([Stop(0, (unit.uid,))])
    assert unit.order is Order.IDLE
    assert unit.path == []


def test_out_of_bounds_target_is_ignored(world):
    unit = world.units_of(0)[0]
    world.step([Move(0, (unit.uid,), (999, 999))])
    assert unit.order is Order.IDLE


def test_unknown_unit_id_is_ignored(world):
    world.step([Move(0, (99999,), (5, 5))])  # must not raise


# -------------------------------------------------------------------- setup


def test_starting_state():
    cfg = SimConfig(num_players=2, start_workers=4)
    w = World(cfg, seed=0)
    assert len(w.units) == 8
    assert len(w.buildings) == 2
    for u in w.units.values():
        assert not w.blocked[u.tile[1], u.tile[0]], "unit spawned inside a wall"


def test_tick_advances_by_one():
    w = World(seed=0)
    assert w.tick == 0
    w.step()
    assert w.tick == 1


# --------------------------------------------------------------------- food


def nearest_berry(world: World, owner: int) -> tuple[int, int]:
    ys, xs = np.where(world.terrain == Terrain.BERRY)
    tc = next(b for b in world.buildings.values() if b.owner == owner)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    return int(xs[i]), int(ys[i])


def test_berries_yield_food(world):
    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    res = world.players[0].resources
    assert res[Resource.FOOD] > 0
    assert res[Resource.WOOD] == 0, "berries must not credit the wrong stockpile"


def test_food_and_gold_are_separate_stockpiles():
    """Regression: Resource.FOOD once shared a value with Resource.GOLD, which
    made IntEnum alias them and silently merge the two stockpiles."""
    assert Resource.FOOD is not Resource.GOLD
    assert len({int(r) for r in Resource}) == 4

    w = World(SimConfig(), seed=3)
    w.players[0].add(Resource.FOOD, 5)
    assert w.players[0].resources[Resource.FOOD] == 5
    assert w.players[0].resources[Resource.GOLD] == 0


def test_depleted_bush_becomes_walkable(world):
    target = nearest_berry(world, 0)
    tx, ty = target
    world.resources[ty, tx] = 2
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    assert world.terrain[ty, tx] == Terrain.GRASS
    assert world.players[0].resources[Resource.FOOD] == 2
