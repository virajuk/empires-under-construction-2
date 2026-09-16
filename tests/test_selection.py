"""Selection is UI state, not simulation state, so it is tested through App.

The interesting cases are the ones that used to fail: clicking a unit that is
part-way between tiles, and picking apart a stack of units standing on the same
spot.
"""

import pygame
import pytest

from empires.app import DOUBLE_CLICK_MS, App
from empires.config import RenderConfig, SimConfig
from empires.core.entities import UnitKind
from empires.render.renderer import unit_radius


@pytest.fixture
def app():
    a = App(SimConfig(), RenderConfig(), seed=3)
    yield a
    pygame.quit()


def screen_pos(app: App, unit) -> tuple[int, int]:
    return app.camera.world_to_screen(unit.x, unit.y)


def put(app: App, unit, tx: float, ty: float) -> None:
    """Move a unit to an exact float position and centre the camera on it."""
    unit.x, unit.y = tx, ty


# ------------------------------------------------------------ click picking


def test_click_selects_the_unit_under_the_cursor(app):
    u = app.world.units_of(0)[0]
    app.camera.center_on_tile(u.x, u.y)
    app._select_click(screen_pos(app, u), additive=False, now_ms=0)
    assert app.selected == {u.uid}


def test_click_hits_a_unit_between_tiles(app):
    """Regression: hit-testing used to floor the click to a tile and look up
    who was standing there, so a unit drawn overlapping its neighbouring tile
    could not be clicked on the half of it that overhung."""
    u = app.world.units_of(0)[0]
    # Sit the unit hard against a tile boundary; its circle now spills over.
    put(app, u, 20.98, 20.5)
    app.camera.center_on_tile(u.x, u.y)

    sx, sy = screen_pos(app, u)
    overhang = (sx + unit_radius(app.camera.tile_size) - 1, sy)
    assert app.camera.screen_to_tile(*overhang) != u.tile, "not testing the overhang"

    app._select_click(overhang, additive=False, now_ms=0)
    assert app.selected == {u.uid}


def test_click_on_empty_ground_clears_the_selection(app):
    u = app.world.units_of(0)[0]
    app.camera.center_on_tile(u.x, u.y)
    app._select_click(screen_pos(app, u), additive=False, now_ms=0)
    assert app.selected

    sx, sy = screen_pos(app, u)
    app._select_click((sx + 200, sy + 150), additive=False, now_ms=5000)
    assert app.selected == set()


def test_click_ignores_enemy_units(app):
    enemy = app.world.units_of(1)[0]
    app.camera.center_on_tile(enemy.x, enemy.y)
    app._select_click(screen_pos(app, enemy), additive=False, now_ms=0)
    assert app.selected == set()


def test_click_picks_the_nearest_of_two_overlapping_units(app):
    a, b = app.world.units_of(0)[:2]
    put(app, a, 20.5, 20.5)
    put(app, b, 20.8, 20.5)
    app.camera.center_on_tile(20.5, 20.5)

    app._select_click(screen_pos(app, a), additive=False, now_ms=0)
    assert app.selected == {a.uid}


# ---------------------------------------------------------------- cycling


def test_repeated_clicks_cycle_through_a_stack(app):
    """Units do not collide, so several routinely share a spot."""
    stack = app.world.units_of(0)[:3]
    for u in stack:
        put(app, u, 20.5, 20.5)
    app.camera.center_on_tile(20.5, 20.5)
    pos = screen_pos(app, stack[0])

    picked = []
    t = 0
    for _ in range(len(stack) + 1):
        t += DOUBLE_CLICK_MS + 50  # slow clicks, so this is cycling not double-click
        app._select_click(pos, additive=False, now_ms=t)
        assert len(app.selected) == 1
        picked.append(next(iter(app.selected)))

    assert len(set(picked[:3])) == 3, f"expected 3 distinct units, got {picked}"
    assert picked[3] == picked[0], "cycle should wrap around"


def test_cycling_resets_when_you_click_elsewhere(app):
    a, b = app.world.units_of(0)[:2]
    put(app, a, 20.5, 20.5)
    put(app, b, 20.5, 20.5)
    put(app, app.world.units_of(0)[2], 30.5, 30.5)
    app.camera.center_on_tile(20.5, 20.5)

    stack_pos = screen_pos(app, a)
    app._select_click(stack_pos, additive=False, now_ms=1000)
    first = next(iter(app.selected))

    far = screen_pos(app, app.world.units_of(0)[2])
    app._select_click(far, additive=False, now_ms=3000)
    app._select_click(stack_pos, additive=False, now_ms=5000)
    assert app.selected == {first}, "returning to a stack should start from the nearest again"


# ----------------------------------------------------------------- modifiers


def test_shift_click_adds_then_toggles_off(app):
    a, b = app.world.units_of(0)[:2]
    put(app, a, 20.5, 20.5)
    put(app, b, 24.5, 20.5)
    app.camera.center_on_tile(22.5, 20.5)

    app._select_click(screen_pos(app, a), additive=False, now_ms=0)
    app._select_click(screen_pos(app, b), additive=True, now_ms=1000)
    assert app.selected == {a.uid, b.uid}

    app._select_click(screen_pos(app, b), additive=True, now_ms=2000)
    assert app.selected == {a.uid}, "shift-clicking a selected unit should drop it"


# -------------------------------------------------------------- double-click


def test_double_click_selects_all_of_that_kind_on_screen(app):
    villagers = app.world.units_of(0)
    for i, u in enumerate(villagers):
        put(app, u, 20.5 + i, 20.5)
    app.camera.center_on_tile(20.5, 20.5)

    soldier = app.world.add_unit(0, UnitKind.SOLDIER, 21, 22)
    pos = screen_pos(app, villagers[0])

    app._select_click(pos, additive=False, now_ms=1000)
    app._select_click(pos, additive=False, now_ms=1000 + DOUBLE_CLICK_MS - 50)

    assert app.selected == {u.uid for u in villagers}
    assert soldier.uid not in app.selected, "double-click must not cross unit kinds"


def test_double_click_ignores_units_off_screen(app):
    villagers = app.world.units_of(0)
    near = villagers[0]
    put(app, near, 20.5, 20.5)
    for u in villagers[1:]:
        put(app, u, 62.5, 46.5)  # far corner of a 64x48 map
    app.camera.center_on_tile(near.x, near.y)

    pos = screen_pos(app, near)
    app._select_click(pos, additive=False, now_ms=1000)
    app._select_click(pos, additive=False, now_ms=1200)
    assert app.selected == {near.uid}


def test_slow_second_click_is_not_a_double_click(app):
    villagers = app.world.units_of(0)
    for i, u in enumerate(villagers):
        put(app, u, 20.5 + i, 20.5)
    app.camera.center_on_tile(20.5, 20.5)
    pos = screen_pos(app, villagers[0])

    app._select_click(pos, additive=False, now_ms=0)
    app._select_click(pos, additive=False, now_ms=DOUBLE_CLICK_MS + 100)
    assert len(app.selected) == 1


# -------------------------------------------------------------- box select


def test_box_selects_every_own_unit_inside_it(app):
    villagers = app.world.units_of(0)
    for i, u in enumerate(villagers):
        put(app, u, 20.5 + i * 0.5, 20.5)
    app.camera.center_on_tile(21.0, 20.5)

    xs = [screen_pos(app, u) for u in villagers]
    left = min(p[0] for p in xs) - 10
    right = max(p[0] for p in xs) + 10
    top = min(p[1] for p in xs) - 10
    bottom = max(p[1] for p in xs) + 10
    app._select_box(pygame.Rect(left, top, right - left, bottom - top), additive=False)

    assert app.selected == {u.uid for u in villagers}


def test_box_excludes_enemies(app):
    enemy = app.world.units_of(1)[0]
    mine = app.world.units_of(0)[0]
    put(app, mine, 20.5, 20.5)
    put(app, enemy, 20.9, 20.5)
    app.camera.center_on_tile(20.5, 20.5)

    sx, sy = screen_pos(app, mine)
    app._select_box(pygame.Rect(sx - 40, sy - 40, 80, 80), additive=False)
    assert app.selected == {mine.uid}
