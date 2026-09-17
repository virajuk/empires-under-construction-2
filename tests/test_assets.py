"""Sprite loading and per-tile variety.

The property that matters most is stability: the renderer redraws every frame,
so a tree chosen at random per call would make the whole forest flicker.

Per-resource checks are parametrised over ``DECOR_DIRS``, so art added for a new
terrain is covered the moment it is registered.
"""

import pygame
import pytest

from empires.config import RenderConfig, SimConfig
from empires.core.terrain import Terrain
from empires.core.world import World
from empires.render import assets as assets_mod
from empires.render.assets import DECOR_DIRS, Assets, variant_index
from empires.render.camera import Camera
from empires.render.renderer import DECOR_TERRAIN, TERRAIN_COLOURS, Renderer

TS = 22

# Which RenderConfig field sizes which terrain. Spelled out here rather than
# imported so the test fails if the renderer's wiring is changed by accident.
SCALE_FIELDS = {
    Terrain.FOREST: "tree_scale",
    Terrain.BERRY: "bush_scale",
    Terrain.GOLD: "gold_scale",
    Terrain.STONE: "stone_scale",
}

DECORATED = [Terrain(t) for t in DECOR_DIRS]


@pytest.fixture(scope="module", autouse=True)
def _display():
    pygame.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.quit()


def build(cfg: RenderConfig | None = None) -> Renderer:
    """A renderer on a scratch surface -- exercises the real config wiring.

    Sprites are forced on: this module tests the art pipeline, so it must not
    go quiet just because the app now defaults to flat colour tiles.
    """
    cfg = cfg or RenderConfig(use_sprites=True)
    world = World(SimConfig(), seed=3)
    camera = Camera(world.width, world.height, TS, 320, 240)
    return Renderer(pygame.Surface((320, 240)), camera, cfg)


@pytest.fixture
def art():
    return build().assets


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


@pytest.mark.parametrize("terrain", DECORATED)
def test_sprite_lookup_is_stable(art, terrain):
    first = art.decor_sprite(terrain, 9, 4, seed=7)
    for _ in range(30):
        assert art.decor_sprite(terrain, 9, 4, seed=7) is first


# ------------------------------------------------------------ loading


@pytest.mark.parametrize("terrain", DECORATED)
def test_art_is_present_for_every_decorated_terrain(art, terrain):
    """Guards against a rename or a move quietly emptying a sprite set."""
    assert art.has_decor(terrain), (
        f"no sprites found in empires/graphics/{DECOR_DIRS[int(terrain)]}"
    )
    assert art.decor_sprite(terrain, 0, 0, 0) is not None


def test_town_center_art_is_present(art):
    assert art.town_center(44, 44) is not None, "home.png missing"


@pytest.mark.parametrize("terrain,field", SCALE_FIELDS.items())
def test_each_terrain_is_scaled_by_its_own_knob(terrain, field):
    cfg = RenderConfig(use_sprites=True)
    art = build(cfg).assets
    box = int(TS * getattr(cfg, field))
    for sprite in art.decor[int(terrain)]:
        # The longest side meets the box; the other is shorter because the
        # aspect ratio of the trimmed artwork is preserved.
        assert max(sprite.get_size()) == box


@pytest.mark.parametrize("terrain", DECORATED)
def test_sprites_are_trimmed_of_transparent_margin(art, terrain):
    """Regression: the source art is padded inconsistently -- trees fill their
    64x64 canvas, bushes occupy barely half of it. Scaling the raw canvas made
    trees render 2.2x the size of bushes while the config said 1.4x."""
    for sprite in art.decor[int(terrain)]:
        rects = pygame.mask.from_surface(sprite).get_bounding_rects()
        assert rects, "sprite is fully transparent"
        bbox = rects[0].unionall(rects[1:])
        # Content should reach the edges of its own surface, give or take a
        # pixel of antialiasing lost in the rescale.
        assert bbox.width >= sprite.get_width() - 2
        assert bbox.height >= sprite.get_height() - 2


def test_visible_size_ratio_matches_the_configured_ratio():
    """The whole point of trimming: the scale knobs mean what they say."""
    art = Assets(TS, scales={int(Terrain.FOREST): 1.6, int(Terrain.BERRY): 1.4})
    trees = art.decor[int(Terrain.FOREST)]
    bushes = art.decor[int(Terrain.BERRY)]
    tree = sum(max(s.get_size()) for s in trees) / len(trees)
    bush = sum(max(s.get_size()) for s in bushes) / len(bushes)
    assert tree / bush == pytest.approx(1.6 / 1.4, rel=0.05)


def test_trees_stay_larger_than_bushes(art):
    smallest_tree = min(max(s.get_size()) for s in art.decor[int(Terrain.FOREST)])
    largest_bush = max(max(s.get_size()) for s in art.decor[int(Terrain.BERRY)])
    assert smallest_tree > largest_bush * 0.9


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
    assert art.decor == {}
    for terrain in DECORATED:
        assert not art.has_decor(terrain)
        assert art.decor_sprite(terrain, 0, 0, 0) is None
    assert art.town_center(44, 44) is None


def test_missing_graphics_dir_is_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(assets_mod, "GRAPHICS_DIR", tmp_path / "nope")
    art = Assets(TS)
    assert art.decor == {}
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
    renderer = build(RenderConfig(use_sprites=False))
    assert renderer.assets.decor == {}
    renderer.draw(world, set(), 0)


def test_unknown_terrain_has_no_decor(art):
    assert not art.has_decor(Terrain.GRASS)
    assert art.decor_sprite(Terrain.GRASS, 0, 0, 0) is None


# ------------------------------------------------------------- rendering


def test_repeated_draws_are_identical():
    """The frame-to-frame flicker regression, caught at the pixel level."""
    world = World(SimConfig(), seed=3)
    renderer = build()
    renderer.camera.center_on_tile(20, 20)

    renderer.draw(world, set(), 0)
    first = pygame.image.tostring(renderer.surface, "RGB")
    for _ in range(5):
        renderer.draw(world, set(), 0)
        assert pygame.image.tostring(renderer.surface, "RGB") == first


@pytest.mark.parametrize("terrain", DECORATED)
def test_decorated_tiles_are_drawn_over_plain_ground(terrain):
    """Decor sits on grass, so the tile beneath must not still be a flat
    coloured square showing round the edges of the artwork."""
    renderer = build()
    ground = renderer._tiles[int(terrain)].get_at((TS // 2, TS // 4))[:3]
    assert ground == TERRAIN_COLOURS[Terrain.GRASS]


def test_decor_terrain_matches_the_art_table():
    """DECOR_TERRAIN is derived from DECOR_DIRS; a terrain in one but not the
    other would render as bare grass with nothing on it."""
    assert set(DECOR_TERRAIN) == set(DECORATED)


def test_every_decorated_terrain_has_a_scale_knob():
    cfg = RenderConfig()
    for terrain in DECORATED:
        assert terrain in SCALE_FIELDS, f"{terrain.name} has no configured scale"
        assert hasattr(cfg, SCALE_FIELDS[terrain])


# ----------------------------------------------------- surface formats


def test_narrow_surfaces_are_widened_for_scaling():
    """Regression: the gold and stone icons are 8-bit palettised PNGs, and
    smoothscale only accepts 24- or 32-bit surfaces. With no display
    convert_alpha cannot run, so an offscreen render (the RL env's rgb_array
    mode) crashed on them while the 32-bit tree art happened to work."""
    narrow = pygame.Surface((8, 8), depth=8)
    widened = assets_mod._widen(narrow)
    assert widened.get_bitsize() >= 24
    pygame.transform.smoothscale(widened, (4, 4))  # must not raise


def test_widen_leaves_wide_surfaces_alone():
    wide = pygame.Surface((8, 8), pygame.SRCALPHA, 32)
    assert assets_mod._widen(wide) is wide


def test_every_loaded_sprite_can_be_scaled(art):
    for variants in art.decor.values():
        for sprite in variants:
            assert sprite.get_bitsize() >= 24
    assert art._town_center_src.get_bitsize() >= 24
