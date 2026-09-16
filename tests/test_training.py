"""Producing Villagers at a Town Center.

Covers the simulation rules and the UI path that reaches them, since a button
that looks right but queues nothing is the failure mode that matters.
"""

import pygame
import pytest

from empires.app import App
from empires.config import RenderConfig, SimConfig
from empires.core.commands import CancelTrain, Train
from empires.core.entities import (
    BUILDING_SPECS,
    MAX_QUEUE,
    UNIT_SPECS,
    BuildingKind,
    UnitKind,
)
from empires.core.terrain import Resource, Terrain
from empires.core.world import World

TRAIN_TICKS = UNIT_SPECS[UnitKind.VILLAGER].train_ticks
FOOD_COST = UNIT_SPECS[UnitKind.VILLAGER].cost[Resource.FOOD]


@pytest.fixture
def world():
    w = World(SimConfig(), seed=3)
    w.players[0].add(Resource.FOOD, 1000)
    return w


def tc_of(world: World, owner: int):
    return next(b for b in world.buildings.values() if b.owner == owner)


def food(world: World, owner: int) -> int:
    return world.players[owner].resources[Resource.FOOD]


# ------------------------------------------------------------------ cost


def test_villager_costs_fifty_food():
    assert FOOD_COST == 50
    assert UNIT_SPECS[UnitKind.VILLAGER].cost[Resource.WOOD] == 0


def test_queueing_charges_the_cost_immediately(world):
    tc = tc_of(world, 0)
    before = food(world, 0)
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    assert food(world, 0) == before - FOOD_COST
    assert len(tc.queue) == 1


def test_cannot_queue_without_the_food(world):
    tc = tc_of(world, 0)
    world.players[0].resources[Resource.FOOD] = FOOD_COST - 1
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    assert tc.queue == []
    assert food(world, 0) == FOOD_COST - 1, "a rejected order must not charge"


def test_exactly_enough_food_is_enough(world):
    tc = tc_of(world, 0)
    world.players[0].resources[Resource.FOOD] = FOOD_COST
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    assert len(tc.queue) == 1
    assert food(world, 0) == 0


# -------------------------------------------------------------- production


def test_villager_appears_after_the_training_time(world):
    """Exactly ``train_ticks`` ticks from the order landing.

    The tick the order is applied in is the first tick of training -- commands
    are processed before production within a step -- so the elapsed time from
    click to Villager is train_ticks, not train_ticks + 1.
    """
    tc = tc_of(world, 0)
    before = len(world.units_of(0))
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])  # tick 1 of training

    for _ in range(TRAIN_TICKS - 2):
        world.step()
    assert len(world.units_of(0)) == before, "arrived early"
    assert tc.train_timer == TRAIN_TICKS - 1

    world.step()
    assert len(world.units_of(0)) == before + 1
    assert tc.queue == []
    assert tc.train_timer == 0


def test_the_new_unit_is_a_villager_owned_by_the_trainer(world):
    tc = tc_of(world, 0)
    known = {u.uid for u in world.units_of(0)}
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    for _ in range(TRAIN_TICKS):
        world.step()

    new = next(u for u in world.units_of(0) if u.uid not in known)
    assert new.kind is UnitKind.VILLAGER
    assert new.owner == 0
    assert new.hp == UNIT_SPECS[UnitKind.VILLAGER].hp


def test_the_new_unit_spawns_on_open_ground(world):
    tc = tc_of(world, 0)
    known = {u.uid for u in world.units_of(0)}
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    for _ in range(TRAIN_TICKS):
        world.step()

    new = next(u for u in world.units_of(0) if u.uid not in known)
    tx, ty = new.tile
    assert not world.blocked[ty, tx], "spawned inside terrain or its own building"


def test_a_queue_produces_one_at_a_time(world):
    tc = tc_of(world, 0)
    before = len(world.units_of(0))
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)] * 3)
    assert len(tc.queue) == 3

    for _ in range(TRAIN_TICKS):
        world.step()
    assert len(world.units_of(0)) == before + 1
    assert len(tc.queue) == 2

    for _ in range(TRAIN_TICKS * 2):
        world.step()
    assert len(world.units_of(0)) == before + 3
    assert tc.queue == []


def test_production_is_reported(world):
    tc = tc_of(world, 0)
    report = world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    assert report.trained[0] == 0
    for _ in range(TRAIN_TICKS - 2):
        assert world.step().trained[0] == 0
    assert world.step().trained[0] == 1


def test_a_finished_villager_waits_when_there_is_nowhere_to_stand(world):
    """Rather than being dropped, so it pops out when a tile frees up."""
    tc = tc_of(world, 0)
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    before = len(world.units_of(0))

    world.terrain[:, :] = Terrain.WATER  # nowhere on the map is walkable
    world._blocked = None
    for _ in range(TRAIN_TICKS * 2):
        world.step()
    assert len(world.units_of(0)) == before
    assert len(tc.queue) == 1, "the queued villager should still be waiting"

    world.terrain[tc.y + tc.h, tc.x] = Terrain.GRASS  # open one tile
    world._blocked = None
    world.step()
    assert len(world.units_of(0)) == before + 1


# ------------------------------------------------------------- validation


def test_cannot_train_at_another_players_building(world):
    enemy_tc = tc_of(world, 1)
    before = food(world, 0)
    world.step([Train(0, enemy_tc.bid, UnitKind.VILLAGER)])
    assert enemy_tc.queue == []
    assert food(world, 0) == before


def test_cannot_train_a_unit_the_building_does_not_make(world):
    tc = tc_of(world, 0)
    assert UnitKind.SOLDIER not in BUILDING_SPECS[BuildingKind.TOWN_CENTER].trains
    world.step([Train(0, tc.bid, UnitKind.SOLDIER)])
    assert tc.queue == []


def test_unknown_building_id_is_ignored(world):
    before = food(world, 0)
    world.step([Train(0, 99999, UnitKind.VILLAGER)])
    assert food(world, 0) == before


def test_queue_is_capped(world):
    tc = tc_of(world, 0)
    world.players[0].resources[Resource.FOOD] = FOOD_COST * (MAX_QUEUE + 5)
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)] * (MAX_QUEUE + 5))
    assert len(tc.queue) == MAX_QUEUE
    assert food(world, 0) == FOOD_COST * 5, "orders past the cap must not charge"


def test_can_train_agrees_with_what_the_world_accepts(world):
    """The HUD greys the button out with can_train, so the two must not drift."""
    tc = tc_of(world, 0)
    for food_amount in (0, FOOD_COST - 1, FOOD_COST, FOOD_COST * 3):
        world.players[0].resources[Resource.FOOD] = food_amount
        tc.queue.clear()
        predicted = world.can_train(0, tc.bid, UnitKind.VILLAGER)
        world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
        assert bool(tc.queue) == predicted

    assert not world.can_train(0, tc_of(world, 1).bid, UnitKind.VILLAGER)
    assert not world.can_train(0, tc.bid, UnitKind.SOLDIER)


# ----------------------------------------------------------------- cancel


def test_cancel_refunds_in_full(world):
    tc = tc_of(world, 0)
    before = food(world, 0)
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    world.step([CancelTrain(0, tc.bid)])
    assert tc.queue == []
    assert food(world, 0) == before


def test_cancel_takes_from_the_back_of_the_queue(world):
    tc = tc_of(world, 0)
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)] * 3)
    for _ in range(10):
        world.step()
    progress = tc.train_timer
    world.step([CancelTrain(0, tc.bid)])
    assert len(tc.queue) == 2
    assert tc.train_timer >= progress, "the one in progress must keep its work"


def test_cancel_on_an_empty_queue_does_nothing(world):
    tc = tc_of(world, 0)
    before = food(world, 0)
    world.step([CancelTrain(0, tc.bid)])
    assert food(world, 0) == before


def test_cannot_cancel_another_players_production(world):
    enemy_tc = tc_of(world, 1)
    world.players[1].add(Resource.FOOD, 100)
    world.step([Train(1, enemy_tc.bid, UnitKind.VILLAGER)])
    world.step([CancelTrain(0, enemy_tc.bid)])
    assert len(enemy_tc.queue) == 1


# ------------------------------------------------------------ determinism


def test_training_keeps_the_world_deterministic():
    a, b = World(SimConfig(), seed=5), World(SimConfig(), seed=5)
    for w in (a, b):
        w.players[0].add(Resource.FOOD, 500)
    cmd = [Train(0, tc_of(a, 0).bid, UnitKind.VILLAGER)] * 3
    a.step(cmd)
    b.step([Train(0, tc_of(b, 0).bid, UnitKind.VILLAGER)] * 3)
    for _ in range(TRAIN_TICKS * 2):
        a.step()
        b.step()
    assert a.state_hash() == b.state_hash()


def test_state_hash_notices_production(world):
    tc = tc_of(world, 0)
    before = world.state_hash()
    world.step([Train(0, tc.bid, UnitKind.VILLAGER)])
    world.tick -= 1  # isolate the change to the queue, not the clock
    assert world.state_hash() != before


# ------------------------------------------------------------------- UI


@pytest.fixture
def app():
    a = App(SimConfig(), RenderConfig(), seed=3)
    a.world.players[0].add(Resource.FOOD, 1000)
    yield a
    pygame.quit()


def centre_on(app: App, b) -> tuple[int, int]:
    app.camera.center_on_tile(*b.center)
    return app.camera.world_to_screen(b.x + 0.5, b.y + 0.5)


def test_clicking_a_town_center_selects_it(app):
    tc = tc_of(app.world, 0)
    app._select_click(centre_on(app, tc), additive=False, now_ms=0)
    assert app.selected_building == tc.bid


def test_selecting_a_building_clears_the_unit_selection(app):
    tc = tc_of(app.world, 0)
    app.selected = {u.uid for u in app.world.units_of(0)}
    app._select_click(centre_on(app, tc), additive=False, now_ms=0)
    assert app.selected == set()
    assert app.selected_building == tc.bid


def test_selecting_a_unit_clears_the_building_selection(app):
    tc = tc_of(app.world, 0)
    app._select_click(centre_on(app, tc), additive=False, now_ms=0)
    assert app.selected_building is not None

    unit = app.world.units_of(0)[0]
    unit.x, unit.y = 30.5, 30.5
    app.camera.center_on_tile(unit.x, unit.y)
    app._select_click(app.camera.world_to_screen(unit.x, unit.y),
                      additive=False, now_ms=5000)
    assert app.selected == {unit.uid}
    assert app.selected_building is None


def test_cannot_select_an_enemy_town_center(app):
    enemy_tc = tc_of(app.world, 1)
    app._select_click(centre_on(app, enemy_tc), additive=False, now_ms=0)
    assert app.selected_building is None


def test_box_select_clears_the_building_selection(app):
    tc = tc_of(app.world, 0)
    app._select_click(centre_on(app, tc), additive=False, now_ms=0)

    unit = app.world.units_of(0)[0]
    unit.x, unit.y = 30.5, 30.5
    app.camera.center_on_tile(unit.x, unit.y)
    sx, sy = app.camera.world_to_screen(unit.x, unit.y)
    app._select_box(pygame.Rect(sx - 40, sy - 40, 80, 80), additive=False)
    assert app.selected_building is None


def test_hotkey_queues_a_villager(app):
    tc = tc_of(app.world, 0)
    app.selected_building = tc.bid
    app._on_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    assert app.pending == [Train(0, tc.bid, UnitKind.VILLAGER)]


def test_hotkey_does_nothing_without_a_building_selected(app):
    app._on_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    assert app.pending == []


def test_the_panel_button_queues_a_villager(app):
    tc = tc_of(app.world, 0)
    app.selected_building = tc.bid
    app.hud.draw(app.world, app.camera, app.selected, 0, 60.0, app.selected_building)
    assert app.hud.train_button is not None

    app._on_panel_click(app.hud.train_button.center)
    assert app.pending == [Train(0, tc.bid, UnitKind.VILLAGER)]


def test_clicking_empty_panel_space_does_nothing(app):
    tc = tc_of(app.world, 0)
    app.selected_building = tc.bid
    app.hud.draw(app.world, app.camera, app.selected, 0, 60.0, app.selected_building)
    app._on_panel_click((app.hud.rect.x + 5, app.hud.rect.y + 5))
    assert app.pending == []


def test_no_train_button_without_a_building_selected(app):
    app.hud.draw(app.world, app.camera, app.selected, 0, 60.0, None)
    assert app.hud.train_button is None, "a stale button rect would stay clickable"


def test_the_full_click_to_villager_loop(app):
    """Select the Town Center, press the button, get a Villager."""
    tc = tc_of(app.world, 0)
    before = len(app.world.units_of(0))

    app._select_click(centre_on(app, tc), additive=False, now_ms=0)
    app.hud.draw(app.world, app.camera, app.selected, 0, 60.0, app.selected_building)
    app._on_panel_click(app.hud.train_button.center)

    app.world.step(app.pending)
    app.pending = []
    for _ in range(TRAIN_TICKS):
        app.world.step()

    assert len(app.world.units_of(0)) == before + 1
