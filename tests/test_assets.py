"""Sprite loading and per-tile variety.

The property that matters most is stability: the renderer redraws every frame,
so a tree chosen at random per call would make the whole forest flicker.
"""

import pygame
import pytest

from empires.config import RenderConfig, SimConfig
from empires.core.terrain import Terrain
from empires.core.world import World
from empires.render import assets as assets_mod
from empires.render.assets import Assets, variant_index
from empires.render.camera import Camera
from empires.render.renderer import Renderer

TS = 22


@pytest.fixture(scope="module", autouse=True)
def _display():
    pygame.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.quit()


@pytest.fixture
def art():
    return Assets(TS)


# --------------------------------------------------------------- variety


def test_variant_index_is_stable():
    """Same tile, same seed, same sprite -- every time, or the forest flickers."""
    first = variant_index(12, 7, seed=3, count=8)
    for _ in range(100):
        assert variant_index(12, 7, seed=3, count=8) == first


def test_variant_index_stays_in_range():
    for x in range(40):
        for y in range(40):
            assert 0 <= variant_index(x, y, 0, 8) < 8


def test_variant_index_handles_empty_set():
    assert variant_index(3, 4, 0, 0) == 0


def test_variant_index_varies_across_tiles():
    seen = {variant_index(x, y, 1, 8) for x in range(20) for y in range(20)}
    assert len(seen) == 8, "every variant should get used across a patch of map"


def test_variant_index_varies_across_seeds():
    a = [variant_index(x, 0, 1, 8) for x in range(60)]
    b = [variant_index(x, 0, 2, 8) for x in range(60)]
    assert a != b, "a different map seed should grow a different forest"


# ------------------------------------------------------------ loading


def test_project_art_is_present(art):
    """Guards against a rename or a move quietly emptying the sprite set."""
    assert art.has_trees, "no tree sprites found in empires/graphics/tree"
    assert art.has_bushes, "no bush sprites found in empires/graphics/berry_bushes"
    assert art.town_center(44, 44) is not None, "home.png missing"


def test_sprites_are_scaled_to_the_tile_size():
    cfg = RenderConfig()
    art = Assets(TS, cfg.tree_scale, cfg.bush_scale)
    # The longest side meets the box; the other is shorter because the aspect
    # ratio of the trimmed artwork is preserved.
    for sprite in art.trees:
        assert max(sprite.get_size()) == int(TS * cfg.tree_scale)
    for sprite in art.bushes:
        assert max(sprite.get_size()) == int(TS * cfg.bush_scale)


def test_sprites_are_trimmed_of_transparent_margin():
    """Regression: the source art is padded inconsistently -- trees fill their
    64x64 canvas, bushes occupy barely half of it. Scaling the raw canvas made
    trees render 2.2x the size of bushes while the config said 1.4x."""
    art = Assets(TS)
    for sprite in art.trees + art.bushes:
        rects = pygame.mask.from_surface(sprite).get_bounding_rects()
        assert rects, "sprite is fully transparent"
        bbox = rects[0].unionall(rects[1:])
        # Content should reach the edges of its own surface, give or take a
        # pixel of antialiasing lost in the rescale.
        assert bbox.width >= sprite.get_width() - 2
        assert bbox.height >= sprite.get_height() - 2


def test_visible_size_ratio_matches_the_configured_ratio():
    """The whole point of trimming: the scale knobs mean what they say."""
    art = Assets(TS, tree_scale=1.6, bush_scale=1.4)
    tree = sum(max(s.get_size()) for s in art.trees) / len(art.trees)
    bush = sum(max(s.get_size()) for s in art.bushes) / len(art.bushes)
    assert tree / bush == pytest.approx(1.6 / 1.4, rel=0.05)


def test_trees_stay_larger_than_bushes():
    art = Assets(TS, RenderConfig().tree_scale, RenderConfig().bush_scale)
    smallest_tree = min(max(s.get_size()) for s in art.trees)
    largest_bush = max(max(s.get_size()) for s in art.bushes)
    assert smallest_tree > largest_bush * 0.9


def test_tree_lookup_returns_the_same_surface_object(art):
    a = art.tree(9, 4, seed=7)
    b = art.tree(9, 4, seed=7)
    assert a is b, "lookups should hit the cached, pre-scaled surface"


def test_building_sprite_is_cached_per_size(art):
    assert art.town_center(44, 44) is art.town_center(44, 44)
    assert art.town_center(44, 44) is not art.town_center(66, 66)


def test_town_center_fits_its_box_without_distortion(art):
    """Fitted, not stretched -- the trimmed source is wider than it is tall,
    so forcing both axes would squash the roof."""
    src = art._town_center_src
    sprite = art.town_center(57, 61)
    w, h = sprite.get_size()
    assert max(w, h) == min(57, 61)
    assert w / h == pytest.approx(src.get_width() / src.get_height(), rel=0.05)


# ------------------------------------------------------- graceful fallback


def test_disabled_assets_return_nothing():
    art = Assets(TS, enabled=False)
    assert not art.has_trees and not art.has_bushes
    assert art.tree(0, 0, 0) is None
    assert art.bush(0, 0, 0) is None
    assert art.town_center(44, 44) is None


def test_missing_graphics_dir_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(assets_mod, "GRAPHICS_DIR", tmp_path / "nope")
    art = Assets(TS)
    assert not art.has_trees and not art.has_bushes
    assert art.town_center(44, 44) is None


def test_renderer_draws_without_art(monkeypatch, tmp_path):
    """A fresh clone with no graphics directory must still render."""
    monkeypatch.setattr(assets_mod, "GRAPHICS_DIR", tmp_path / "nope")
    world = World(SimConfig(), seed=3)
    surface = pygame.Surface((320, 240))
    camera = Camera(world.width, world.height, TS, 320, 240)
    Renderer(surface, camera).draw(world, set(), 0)  # must not raise


def test_renderer_draws_with_sprites_disabled():
    world = World(SimConfig(), seed=3)
    surface = pygame.Surface((320, 240))
    camera = Camera(world.width, world.height, TS, 320, 240)
    r = Renderer(surface, camera, RenderConfig(use_sprites=False))
    assert not r.assets.has_trees
    r.draw(world, set(), 0)


# ------------------------------------------------------------- rendering


def test_repeated_draws_are_identical():
    """The frame-to-frame flicker regression, caught at the pixel level."""
    world = World(SimConfig(), seed=3)
    surface = pygame.Surface((320, 240))
    camera = Camera(world.width, world.height, TS, 320, 240)
    camera.center_on_tile(20, 20)
    renderer = Renderer(surface, camera)

    renderer.draw(world, set(), 0)
    first = pygame.image.tostring(surface, "RGB")
    for _ in range(5):
        renderer.draw(world, set(), 0)
        assert pygame.image.tostring(surface, "RGB") == first


def test_forest_tiles_are_drawn_over_plain_ground():
    """Decor sits on grass, so a forest tile must not still be a dark square."""
    world = World(SimConfig(), seed=3)
    surface = pygame.Surface((320, 240))
    camera = Camera(world.width, world.height, TS, 320, 240)
    renderer = Renderer(surface, camera)
    from empires.render.renderer import TERRAIN_COLOURS

    ground = renderer._tiles[int(Terrain.FOREST)].get_at((TS // 2, TS // 4))[:3]
    assert ground == TERRAIN_COLOURS[Terrain.GRASS]
