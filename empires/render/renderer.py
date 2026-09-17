"""Drawing. Reads the world, writes pixels, mutates nothing.

Sprites come from ``empires/graphics`` via :mod:`empires.render.assets`. Every
sprite lookup can return ``None`` and each one has a flat-colour fallback, so
the game still runs with the art directory missing or incomplete.
"""

from __future__ import annotations

import pygame

from ..config import RenderConfig
from ..core.entities import UNIT_SPECS, BuildingKind, Order, Unit
from ..core.terrain import Terrain
from ..core.world import World
from .assets import DECOR_DIRS, Assets, variant_index
from .camera import Camera

TERRAIN_COLOURS: dict[int, tuple[int, int, int]] = {
    Terrain.GRASS: (74, 112, 58),
    Terrain.WATER: (46, 84, 130),
    Terrain.FOREST: (34, 68, 38),
    Terrain.GOLD: (176, 142, 48),
    Terrain.STONE: (120, 120, 126),
    Terrain.BERRY: (150, 54, 76),
}

PLAYER_COLOURS: tuple[tuple[int, int, int], ...] = (
    (72, 142, 232),   # player 0 -- blue
    (214, 78, 62),    # player 1 -- red
    (96, 186, 96),
    (206, 188, 72),
)

GRID_COLOUR = (0, 0, 0, 28)
SELECT_COLOUR = (250, 250, 210)
PATH_COLOUR = (240, 240, 200)

# Terrain drawn as bare ground with a sprite on top, not as a flat colour.
# Derived from the art table, so the two cannot disagree.
DECOR_TERRAIN: tuple[Terrain, ...] = tuple(Terrain(t) for t in DECOR_DIRS)

# Decor is taller than its tile and stands on the tile's bottom edge, so
# sprites belonging to tiles just outside the viewport still reach into it.
DECOR_MARGIN_TILES = 3


def unit_radius(tile_size: int) -> int:
    """Radius a unit is drawn at, in pixels.

    Selection hit-testing imports this too, so what you can click is exactly
    what you can see. When the two were computed separately, clicking a moving
    unit's visible circle could miss it entirely.
    """
    return max(3, tile_size // 3)


class Renderer:
    def __init__(self, surface: pygame.Surface, camera: Camera,
                 cfg: RenderConfig | None = None) -> None:
        self.surface = surface
        self.camera = camera
        self.cfg = cfg or RenderConfig()
        ts = camera.tile_size
        # The only place a config field is tied to a terrain. Adding a
        # decorated resource is a line here, a line in DECOR_DIRS, and a scale
        # on RenderConfig.
        self.assets = Assets(
            ts,
            scales={
                int(Terrain.FOREST): self.cfg.tree_scale,
                int(Terrain.BERRY): self.cfg.bush_scale,
                int(Terrain.GOLD): self.cfg.gold_scale,
                int(Terrain.STONE): self.cfg.stone_scale,
            },
            enabled=self.cfg.use_sprites,
        )

        # One pre-filled surface per terrain type, blitted rather than
        # re-filled. Cheap, and it gives a place to hang textures later.
        self._tiles: dict[int, pygame.Surface] = {
            int(terrain): self._make_tile(terrain, colour, ts)
            for terrain, colour in TERRAIN_COLOURS.items()
        }

    # ---------------------------------------------------------------- setup

    def _has_decor_for(self, terrain: Terrain) -> bool:
        return self.assets.has_decor(terrain)

    def _make_tile(self, terrain: Terrain, colour: tuple[int, int, int],
                   ts: int) -> pygame.Surface:
        surf = pygame.Surface((ts, ts))
        if terrain in DECOR_TERRAIN and self._has_decor_for(terrain):
            # A sprite is drawn over this tile, so the ground beneath it is
            # plain grass -- otherwise the tile colour rings the artwork.
            colour = TERRAIN_COLOURS[Terrain.GRASS]
            surf.fill(colour)
        elif terrain is Terrain.BERRY:
            # No art: a solid red square reads as something alarming, so draw
            # bushes sitting on grass.
            surf.fill(TERRAIN_COLOURS[Terrain.GRASS])
            bush = _shade(colour, 0.55)
            for bx, by in ((0.32, 0.36), (0.66, 0.34), (0.5, 0.68)):
                pygame.draw.circle(surf, bush, (int(bx * ts), int(by * ts)),
                                   max(2, ts // 5))
            for bx, by in ((0.32, 0.36), (0.66, 0.34), (0.5, 0.68)):
                pygame.draw.circle(surf, colour, (int(bx * ts), int(by * ts)),
                                   max(1, ts // 9))
            colour = TERRAIN_COLOURS[Terrain.GRASS]
        else:
            surf.fill(colour)
        pygame.draw.line(surf, _shade(colour, 0.9), (0, ts - 1), (ts - 1, ts - 1))
        pygame.draw.line(surf, _shade(colour, 0.9), (ts - 1, 0), (ts - 1, ts - 1))
        return surf

    # ----------------------------------------------------------------- draw

    def draw(self, world: World, selected: set[int], player: int,
             drag_rect: pygame.Rect | None = None,
             selected_building: int | None = None) -> None:
        self.surface.fill((20, 24, 20))
        self._draw_terrain(world)
        # Decor is a second pass, not part of the terrain loop: a tree is
        # taller than its tile, so drawing it inline would let the next row of
        # ground paint over its trunk.
        self._draw_decor(world)
        self._draw_buildings(world, selected_building)
        self._draw_paths(world, selected)
        self._draw_units(world, selected)
        if drag_rect is not None:
            pygame.draw.rect(self.surface, SELECT_COLOUR, drag_rect, width=1)

    # -------------------------------------------------------------- layers

    def _draw_terrain(self, world: World) -> None:
        cam = self.camera
        ts = cam.tile_size
        x0, y0, x1, y1 = cam.visible_tiles
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(world.width, x1), min(world.height, y1)
        blit = self.surface.blit
        for ty in range(y0, y1):
            sy = int(ty * ts - cam.y)
            row = world.terrain[ty]
            for tx in range(x0, x1):
                blit(self._tiles[int(row[tx])], (int(tx * ts - cam.x), sy))

    def _draw_decor(self, world: World) -> None:
        if not self.assets.decor:
            return
        cam = self.camera
        ts = cam.tile_size
        x0, y0, x1, y1 = cam.visible_tiles
        m = DECOR_MARGIN_TILES
        x0, y0 = max(0, x0 - m), max(0, y0 - m)
        x1, y1 = min(world.width, x1 + m), min(world.height, y1 + m)

        seed = world.seed
        blit = self.surface.blit
        decor = self.assets.decor
        # Top row first, so a nearer tree overlaps the one behind it.
        for ty in range(y0, y1):
            row = world.terrain[ty]
            base_y = int(ty * ts - cam.y) + ts
            for tx in range(x0, x1):
                variants = decor.get(int(row[tx]))
                if not variants:
                    continue
                # Indexed inline rather than through Assets: this runs for
                # every visible tile, so it is the one hot loop in the renderer.
                sprite = variants[variant_index(tx, ty, seed, len(variants))]
                # Centred on the tile, standing on its bottom edge.
                sx = int(tx * ts - cam.x) + (ts - sprite.get_width()) // 2
                blit(sprite, (sx, base_y - sprite.get_height()))

    def _draw_buildings(self, world: World,
                        selected_building: int | None = None) -> None:
        cam = self.camera
        ts = cam.tile_size
        view = self.surface.get_rect()
        for b in world.buildings.values():
            sx, sy = cam.world_to_screen(b.x, b.y)
            footprint = pygame.Rect(sx, sy, b.w * ts, b.h * ts)
            if not footprint.colliderect(view):
                continue
            colour = PLAYER_COLOURS[b.owner % len(PLAYER_COLOURS)]

            sprite = None
            if b.kind is BuildingKind.TOWN_CENTER:
                # Drawn larger than the plot it occupies and standing on its
                # bottom edge, so it reads as a building on the ground rather
                # than a flat tile the size of its footprint.
                sprite = self.assets.town_center(
                    int(footprint.width * self.cfg.building_scale),
                    int(footprint.height * self.cfg.building_scale),
                )

            if sprite is None:
                pygame.draw.rect(self.surface, _shade(colour, 0.55), footprint)
                pygame.draw.rect(self.surface, colour, footprint, width=3)
                # Overlays belong on both paths -- without this, selection and
                # training progress disappear whenever the art is missing.
                self._draw_building_overlays(b, footprint, b.bid == selected_building)
                continue

            self.surface.blit(sprite, (
                footprint.centerx - sprite.get_width() // 2,
                footprint.bottom - sprite.get_height(),
            ))
            # Ownership still has to be legible. An outline round the plot gets
            # drawn straight across the building's face, so use a colour bar at
            # its base instead -- visible, and it leaves the artwork alone.
            bar_h = max(3, ts // 6)
            bar = pygame.Rect(footprint.x, footprint.bottom - bar_h,
                              footprint.width, bar_h)
            pygame.draw.rect(self.surface, colour, bar, border_radius=2)
            pygame.draw.rect(self.surface, _shade(colour, 0.5), bar, width=1,
                             border_radius=2)
            self._draw_building_overlays(b, footprint, b.bid == selected_building)

    def _draw_building_overlays(self, b, footprint: pygame.Rect,
                                is_selected: bool) -> None:
        """Selection ring and training progress.

        Both hug the *footprint*, not the artwork: the footprint is what you
        click and what the unit walks out of, so highlighting anything else
        would misreport where the building actually is.
        """
        if is_selected:
            ring = footprint.inflate(6, 6)
            pygame.draw.rect(self.surface, SELECT_COLOUR, ring, width=2,
                             border_radius=3)

        if not b.queue:
            return
        total = max(1, UNIT_SPECS[b.queue[0]].train_ticks)
        frac = max(0.0, min(1.0, b.train_timer / total))
        bar = pygame.Rect(footprint.x, footprint.top - 8, footprint.width, 5)
        pygame.draw.rect(self.surface, (24, 26, 28), bar, border_radius=2)
        pygame.draw.rect(self.surface, (120, 190, 120),
                         (bar.x, bar.y, max(2, int(bar.width * frac)), bar.height),
                         border_radius=2)

    def _draw_paths(self, world: World, selected: set[int]) -> None:
        cam = self.camera
        ts = cam.tile_size
        for uid in selected:
            u = world.units.get(uid)
            if u is None or not u.path:
                continue
            points = [cam.world_to_screen(u.x, u.y)]
            points += [cam.world_to_screen(px + 0.5, py + 0.5) for px, py in u.path]
            if len(points) > 1:
                pygame.draw.lines(self.surface, PATH_COLOUR, False, points, 1)
            if u.target is not None:
                tx, ty = cam.world_to_screen(u.target[0], u.target[1])
                pygame.draw.rect(self.surface, PATH_COLOUR, (tx, ty, ts, ts), width=1)

    def _draw_units(self, world: World, selected: set[int]) -> None:
        cam = self.camera
        ts = cam.tile_size
        radius = unit_radius(ts)
        view = self.surface.get_rect()
        for uid in sorted(world.units):
            u = world.units[uid]
            sx, sy = cam.world_to_screen(u.x, u.y)
            if not view.collidepoint(sx, sy):
                continue
            colour = PLAYER_COLOURS[u.owner % len(PLAYER_COLOURS)]
            if uid in selected:
                pygame.draw.circle(self.surface, SELECT_COLOUR, (sx, sy), radius + 3, width=1)
            pygame.draw.circle(self.surface, colour, (sx, sy), radius)
            pygame.draw.circle(self.surface, _shade(colour, 0.5), (sx, sy), radius, width=1)
            if u.carrying:
                # A small pip so you can see who is hauling a load.
                pygame.draw.circle(self.surface, (240, 230, 160), (sx, sy - radius - 3), 2)
            if u.order is Order.GATHER and u.gather_timer:
                pygame.draw.line(
                    self.surface, (250, 240, 180),
                    (sx - radius, sy + radius + 2),
                    (sx - radius + int(2 * radius * _gather_frac(u, world)), sy + radius + 2),
                    2,
                )


def _gather_frac(u: Unit, world: World) -> float:
    return min(1.0, u.gather_timer / max(1, world.cfg.gather_ticks_per_unit))


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in colour)  # type: ignore[return-value]
