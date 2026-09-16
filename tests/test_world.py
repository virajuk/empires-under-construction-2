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


def strip_all_but(world: World, resource: Resource, keep: tuple[int, int],
                  amount: int) -> None:
    """Leave exactly one tile of ``resource`` on the map, holding ``amount``.

    Villagers move on to the next patch when one runs dry, so a test about a
    single tile has to remove the alternatives or it measures the whole map.
    """
    world.resources[np.isin(world.terrain, TERRAIN_FOR_RESOURCE[resource])] = 0
    world.resources[keep[1], keep[0]] = amount


def test_exhausted_tile_becomes_walkable(world):
    target = nearest_forest(world, 0)
    tx, ty = target
    strip_all_but(world, Resource.WOOD, target, 3)
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


def test_villagers_do_not_exceed_carry_capacity():
    cfg = SimConfig(villager_carry_capacity=5, gather_ticks_per_unit=1)
    w = World(cfg, seed=3)
    target = nearest_forest(w, 0)
    ids = tuple(sorted(u.uid for u in w.units_of(0)))
    w.step([Gather(0, ids, target)])
    for _ in range(500):
        w.step()
        assert all(u.carrying <= cfg.villager_carry_capacity for u in w.units.values())


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
    cfg = SimConfig(num_players=2, start_villagers=4)
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
    strip_all_but(world, Resource.FOOD, target, 2)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    assert world.terrain[ty, tx] == Terrain.GRASS
    assert world.players[0].resources[Resource.FOOD] == 2


def test_villagers_go_idle_when_nothing_of_that_resource_remains(world):
    """The other half of retargeting: stop looking once the map is picked clean."""
    target = nearest_berry(world, 0)
    strip_all_but(world, Resource.FOOD, target, 2)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    run(world, 600, [Gather(0, ids, target)])
    assert all(u.order is Order.IDLE for u in world.units_of(0))
    assert all(u.gather_resource is None for u in world.units_of(0))


# ------------------------------------------------------- villager activity


def test_activity_categories_are_exhaustive(world):
    """harvesting + walking + idle must always equal total.

    Regression: the HUD once counted "gathering" and "idle" and treated
    everything else as nothing, so a Villager under a move order vanished from
    the readout entirely.
    """
    from empires.core.entities import Order

    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids[:2], target)])
    world.step([Move(0, ids[2:], world.nearest_free_tile(30, 20))])

    seen_orders = set()
    seen_states = set()
    for _ in range(400):
        world.step()
        act = world.villager_activity(0)
        assert act.harvesting + act.walking + act.idle == act.total
        seen_orders.update(u.order for u in world.units_of(0))
        seen_states.update(
            k for k, v in (("harvesting", act.harvesting),
                           ("walking", act.walking),
                           ("idle", act.idle)) if v
        )

    # The run has to actually exercise the interesting states, or the
    # invariant above is trivially satisfied.
    assert {Order.GATHER, Order.MOVE, Order.IDLE} <= seen_orders
    assert seen_states == {"harvesting", "walking", "idle"}


def test_walking_counts_villagers_on_their_way_to_a_resource(world):
    """Regression: a Villager crossing the map to reach a bush was classified
    by its *order* (GATHER), so the HUD reported nobody moving while four of
    them were plainly walking."""
    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, target)])
    world.step()

    act = world.villager_activity(0)
    assert act.gathering == len(ids), "all four are assigned to the resource"
    assert act.walking > 0, "and they are walking there, not standing still"

    # And they really do cross the map: over a short window at least as many
    # villagers change position as were counted walking.
    before = {u.uid: (u.x, u.y) for u in world.units_of(0)}
    for _ in range(10):
        world.step()
    moved = sum(1 for u in world.units_of(0) if (u.x, u.y) != before[u.uid])
    assert moved >= act.walking


def test_walking_is_exactly_the_villagers_with_a_path(world):
    """Pins the definition: walking == "has somewhere left to walk".

    Deliberately not asserted against frame-by-frame motion -- a villager
    arriving in range stops with waypoints still queued, and one that has just
    filled its load is routed home only on the following tick. Both are
    one-tick transitions; see VillagerActivity.
    """
    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, target)])

    for _ in range(600):
        world.step()
        act = world.villager_activity(0)
        assert act.walking == sum(1 for u in world.units_of(0) if u.path)


def test_harvesting_counts_only_villagers_that_have_arrived(world):
    from empires.core.entities import Order

    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, target)])
    for _ in range(600):
        world.step()
        act = world.villager_activity(0)
        arrived = sum(
            1 for u in world.units_of(0)
            if u.order in (Order.GATHER, Order.RETURN) and not u.path
        )
        assert act.harvesting == arrived


def test_activity_counts_only_villagers(world):
    """A Soldier standing still is not an idle Villager."""
    from empires.core.entities import UnitKind

    before = world.villager_activity(0)
    world.add_unit(0, UnitKind.SOLDIER, *world.nearest_free_tile(25, 20))
    after = world.villager_activity(0)
    assert after == before


def test_activity_is_per_player(world):
    a = world.villager_activity(0)
    b = world.villager_activity(1)
    assert a.total == b.total == world.cfg.start_villagers


def test_activity_tracks_a_move_order(world):
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    assert world.villager_activity(0).idle == world.cfg.start_villagers

    world.step([Move(0, ids, world.nearest_free_tile(30, 20))])
    act = world.villager_activity(0)
    assert act.walking == len(ids)
    assert act.idle == 0
    assert act.harvesting == 0
    assert act.gathering == 0, "a plain move is not resource work"


def test_activity_counts_hauling_as_gathering(world):
    """A Villager walking a load home is working, not idle."""
    from empires.core.entities import Order

    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, target)])
    for _ in range(600):
        world.step()
        if any(u.order is Order.RETURN for u in world.units_of(0)):
            break
    else:
        pytest.fail("no villager ever started hauling a load back")

    act = world.villager_activity(0)
    returning = sum(1 for u in world.units_of(0) if u.order is Order.RETURN)
    assert returning > 0
    assert act.gathering >= returning
    assert act.harvesting + act.walking + act.idle == act.total


def test_activity_carrying_matches_units(world):
    target = nearest_berry(world, 0)
    ids = tuple(sorted(u.uid for u in world.units_of(0)))
    world.step([Gather(0, ids, target)])
    for _ in range(300):
        world.step()
        expected = sum(u.carrying for u in world.units_of(0) if u.spec.can_gather)
        assert world.villager_activity(0).carrying == expected
