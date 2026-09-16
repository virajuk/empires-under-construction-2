"""The game's vocabulary: a resource gatherer is a Villager, the main building
is a Town Center. These tests pin the names and the specs that carry them, so a
rename cannot silently leave half the codebase speaking the old language."""

import pytest

from empires.config import SimConfig
from empires.core.entities import (
    BUILDING_SPECS,
    UNIT_SPECS,
    BuildingKind,
    UnitKind,
    building_label,
    unit_label,
)
from empires.core.world import World


def test_villager_is_the_gatherer():
    assert UnitKind.VILLAGER.name == "VILLAGER"
    assert UNIT_SPECS[UnitKind.VILLAGER].label == "Villager"
    assert UNIT_SPECS[UnitKind.VILLAGER].can_gather


def test_soldiers_cannot_gather():
    assert not UNIT_SPECS[UnitKind.SOLDIER].can_gather


def test_town_center_is_the_main_building():
    spec = BUILDING_SPECS[BuildingKind.TOWN_CENTER]
    assert spec.label == "Town Center"
    assert spec.is_dropoff


def test_every_kind_has_a_spec():
    """A new kind without a spec would blow up at render time, not import time."""
    assert set(UNIT_SPECS) == set(UnitKind)
    assert set(BUILDING_SPECS) == set(BuildingKind)


@pytest.mark.parametrize(
    "count, expected",
    [(1, "Villager"), (2, "Villagers"), (0, "Villagers")],
)
def test_unit_label_pluralises(count, expected):
    assert unit_label(UnitKind.VILLAGER, count) == expected


@pytest.mark.parametrize(
    "count, expected",
    [(1, "Town Center"), (3, "Town Centers")],
)
def test_building_label_pluralises(count, expected):
    assert building_label(BuildingKind.TOWN_CENTER, count) == expected


# ------------------------------------------------------------------- in-world


def test_starting_units_are_villagers():
    w = World(SimConfig(), seed=0)
    assert all(u.kind is UnitKind.VILLAGER for u in w.units.values())
    assert all(u.label == "Villager" for u in w.units.values())


def test_starting_buildings_are_town_centers():
    w = World(SimConfig(), seed=0)
    assert all(b.kind is BuildingKind.TOWN_CENTER for b in w.buildings.values())
    assert all(b.label == "Town Center" for b in w.buildings.values())


def test_building_footprint_comes_from_its_spec():
    """Callers say what to place, not how big it is."""
    w = World(SimConfig(), seed=0)
    spec = BUILDING_SPECS[BuildingKind.TOWN_CENTER]
    b = next(iter(w.buildings.values()))
    assert (b.w, b.h, b.hp) == (spec.width, spec.height, spec.hp)
    assert len(b.tiles) == spec.width * spec.height


def test_unit_hp_comes_from_its_spec():
    w = World(SimConfig(), seed=0)
    soldier = w.add_unit(0, UnitKind.SOLDIER, *w.nearest_free_tile(20, 20))
    assert soldier.hp == UNIT_SPECS[UnitKind.SOLDIER].hp
    villager = next(u for u in w.units.values() if u.kind is UnitKind.VILLAGER)
    assert villager.hp == UNIT_SPECS[UnitKind.VILLAGER].hp


def test_only_gatherers_accept_a_gather_order():
    """A Soldier ordered onto a berry bush must not start harvesting."""
    import numpy as np

    from empires.core.commands import Gather
    from empires.core.entities import Order
    from empires.core.terrain import Terrain

    w = World(SimConfig(), seed=3)
    ys, xs = np.where(w.terrain == Terrain.BERRY)
    target = (int(xs[0]), int(ys[0]))
    soldier = w.add_unit(0, UnitKind.SOLDIER, *w.nearest_free_tile(20, 20))
    w.step([Gather(0, (soldier.uid,), target)])
    assert soldier.order is Order.IDLE
