import numpy as np
import pytest

from empires.config import SimConfig
from empires.core.mapgen import PLAZA_RADIUS
from empires.core.pathfinding import adjacent_tiles, find_path
from empires.core.terrain import HARVESTABLE_TERRAIN, Terrain
from empires.core.world import World


@pytest.mark.parametrize("seed", range(20))
def test_no_unit_ever_spawns_inside_terrain(seed):
    """Regression: resource patches used to be stamped over the start plaza,
    burying villagers in rock and leaving them permanently unable to path."""
    w = World(SimConfig(), seed=seed)
    for u in w.units.values():
        tx, ty = u.tile
        assert not w.blocked[ty, tx], f"seed {seed}: unit {u.uid} spawned in a wall"


@pytest.mark.parametrize("seed", range(20))
def test_every_villager_can_reach_its_own_town_center(seed):
    w = World(SimConfig(), seed=seed)
    for u in w.units.values():
        tc = next(b for b in w.buildings.values() if b.owner == u.owner)
        target = w.nearest_free_tile(int(tc.center[0]), int(tc.center[1]))
        assert target is not None
        assert find_path(w.blocked, u.tile, target) is not None, (
            f"seed {seed}: unit {u.uid} is walled off from its own base"
        )


@pytest.mark.parametrize("seed", range(10))
def test_each_player_has_harvestable_resources_nearby(seed):
    w = World(SimConfig(), seed=seed)
    for b in w.buildings.values():
        ys, xs = np.where(w.resources > 0)
        assert len(xs), f"seed {seed}: map has no resources at all"
        dist = np.maximum(np.abs(xs - b.x), np.abs(ys - b.y))
        assert dist.min() < 20, f"seed {seed}: player {b.owner} has nothing in reach"


def test_generation_is_deterministic():
    a = World(SimConfig(), seed=42)
    b = World(SimConfig(), seed=42)
    assert np.array_equal(a.terrain, b.terrain)
    assert np.array_equal(a.resources, b.resources)


def test_map_respects_configured_size():
    cfg = SimConfig(map_width=31, map_height=19)
    w = World(cfg, seed=0)
    assert w.terrain.shape == (19, 31)
    assert w.resources.shape == (19, 31)


def test_resource_tiles_and_amounts_agree():
    """A tile has stock if and only if it is a harvestable terrain type."""
    w = World(SimConfig(), seed=5)
    harvestable = np.isin(w.terrain, HARVESTABLE_TERRAIN)
    assert np.array_equal(harvestable, w.resources > 0)


@pytest.mark.parametrize("seed", range(20))
def test_every_player_starts_with_berries_in_reach(seed):
    """Food is the first thing you need, so no start may be without it."""
    w = World(SimConfig(), seed=seed)
    ys, xs = np.where(w.terrain == Terrain.BERRY)
    assert len(xs), f"seed {seed}: map generated no berries at all"
    for b in w.buildings.values():
        dist = np.maximum(np.abs(xs - b.x), np.abs(ys - b.y))
        assert dist.min() <= PLAZA_RADIUS + 6, (
            f"seed {seed}: player {b.owner} has no berries near their base"
        )


@pytest.mark.parametrize("seed", range(20))
def test_every_berry_bush_can_be_stood_next_to(seed):
    """A bush ringed by other bushes is unharvestable -- and looks like a bug
    to the player rather than a design choice."""
    w = World(SimConfig(), seed=seed)
    ys, xs = np.where(w.terrain == Terrain.BERRY)
    for x, y in zip(xs.tolist(), ys.tolist()):
        assert any(
            not w.blocked[ny, nx]
            for nx, ny in adjacent_tiles((x, y), w.width, w.height)
        ), f"seed {seed}: bush at {(x, y)} is walled in"


def test_berry_clusters_are_clusters_not_single_tiles():
    cfg = SimConfig()
    w = World(cfg, seed=1)
    ys, xs = np.where(w.terrain == Terrain.BERRY)
    # At least as many bushes as one guaranteed cluster per player.
    assert len(xs) >= cfg.num_players * cfg.berry_bushes_per_cluster // 2
    # And they clump: most bushes touch at least one other bush.
    touching = sum(
        1
        for x, y in zip(xs.tolist(), ys.tolist())
        if any(
            w.terrain[ny, nx] == Terrain.BERRY
            for nx, ny in adjacent_tiles((x, y), w.width, w.height)
        )
    )
    # Anchored clumps, not tiles smeared around a ring: essentially every bush
    # should have a neighbour, so a patch fits several villagers at once.
    assert touching >= len(xs) * 0.9
