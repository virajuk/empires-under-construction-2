"""Drawing. Reads the world, writes pixels, mutates nothing.

Sprites come from ``empires/graphics`` via :mod:`empires.render.assets`. Every
sprite lookup can return ``None`` and each one has a flat-colour fallback, so
the game still runs with the art directory missing or incomplete.

The view is isometric: ground tiles are diamonds (see :mod:`.camera`), and
decor, buildings and units all stand on that ground rather than being drawn in
fixed layers. A villager walking behind the Town Center has to actually
disappear behind it, so those three are depth-sorted together in one pass
(``_draw_world_objects``) instead of decor-then-buildings-then-units in a
fixed order -- the fixed order was fine when nothing could occlude anything
else diagonally, which stopped being true the moment the camera tilted.
"""

from __future__ import annotations

import pygame

from ..config import RenderConfig
from ..core.entities import BUILDING_SPECS, UNIT_SPECS, Building, BuildingKind, Order, Unit
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
RALLY_COLOUR = (250, 216, 96)

# Terrain drawn as bare ground with a sprite on top, not as a flat colour.
# Derived from the art table, so the two cannot disagree.
DECOR_TERRAIN: tuple[Terrain, ...] = tuple(Terrain(t) for t in DECOR_DIRS)

# Decor is taller than its tile and stands at the tile's front corner, so
# sprites belonging to tiles just outside the viewport still reach into it.
DECOR_MARGIN_TILES = 3


def unit_radius(tile_size: int) -> int:
    """Radius a unit is drawn at, in pixels.

    Selection hit-testing imports this too, so what you can click is exactly
    what you can see. When the two were computed separately, clicking a moving
    unit's visible circle could miss it entirely.
    """
    return max(3, tile_size // 3)


def _tile_diamond(cam: Camera, tx: int, ty: int) -> list[tuple[int, int]]:
    """The four screen corners of tile ``(tx, ty)``, in perimeter order."""
    return [
        cam.world_to_screen(tx, ty),
        cam.world_to_screen(tx + 1, ty),
        cam.world_to_screen(tx + 1, ty + 1),
        cam.world_to_screen(tx, ty + 1),
    ]


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

        # One pre-filled diamond surface per terrain type, blitted rather than
        # re-filled. Cheap, and it gives a place to hang textures later.
        self._tiles: dict[int, pygame.Surface] = {
            int(terrain): self._make_tile(terrain, colour, camera.tile_w, camera.tile_h)
            for terrain, colour in TERRAIN_COLOURS.items()
        }

        # Only used on the flat-colour building fallback, so it can stay tiny.
        self._building_font = pygame.font.SysFont("consolas,menlo,monospace", 14, bold=True)

    # ---------------------------------------------------------------- setup

    def _has_decor_for(self, terrain: Terrain) -> bool:
        return self.assets.has_decor(terrain)

    def _make_tile(self, terrain: Terrain, colour: tuple[int, int, int],
                   tw: int, th: int) -> pygame.Surface:
        """A ``tw`` x ``th`` diamond, transparent outside it.

        The image is the full bounding box of the diamond -- pygame has no
        cheaper way to hand back an arbitrary polygon -- so it needs alpha to
        keep the corners from painting a rectangle around the ground.
        """
        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        points = [(tw / 2, 0), (tw, th / 2), (tw / 2, th), (0, th / 2)]
        if terrain in DECOR_TERRAIN and self._has_decor_for(terrain):
            # A sprite is drawn over this tile, so the ground beneath it is
            # plain grass -- otherwise the tile colour rings the artwork.
            colour = TERRAIN_COLOURS[Terrain.GRASS]
            pygame.draw.polygon(surf, colour, points)
        elif terrain is Terrain.BERRY:
            # No art: a solid red diamond reads as something alarming, so
            # draw bushes sitting on grass.
            grass = TERRAIN_COLOURS[Terrain.GRASS]
            pygame.draw.polygon(surf, grass, points)
            bush = _shade(colour, 0.55)
            for bx, by in ((0.32, 0.42), (0.66, 0.4), (0.5, 0.62)):
                pygame.draw.circle(surf, bush, (int(bx * tw), int(by * th)),
                                   max(2, th // 5))
            for bx, by in ((0.32, 0.42), (0.66, 0.4), (0.5, 0.62)):
                pygame.draw.circle(surf, colour, (int(bx * tw), int(by * th)),
                                   max(1, th // 9))
            colour = grass
        else:
            pygame.draw.polygon(surf, colour, points)
        pygame.draw.polygon(surf, _shade(colour, 0.9), points, width=1)
        return surf

    # ----------------------------------------------------------------- draw

    def draw(self, world: World, selected: set[int], player: int,
             drag_rect: pygame.Rect | None = None,
             selected_building: int | None = None) -> None:
        self.surface.fill((20, 24, 20))
        self._draw_terrain(world)
        self._draw_world_objects(world, selected)
        self._draw_rally_point(world, selected_building)
        self._draw_paths(world, selected)
        if drag_rect is not None:
            pygame.draw.rect(self.surface, SELECT_COLOUR, drag_rect, width=1)

    # -------------------------------------------------------------- layers

    def _draw_terrain(self, world: World) -> None:
        cam = self.camera
        tw, th = cam.tile_w, cam.tile_h
        x0, y0, x1, y1 = cam.visible_tiles
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(world.width, x1), min(world.height, y1)
        blit = self.surface.blit
        tiles = self._tiles
        for ty in range(y0, y1):
            row = world.terrain[ty]
            for tx in range(x0, x1):
                # world_to_screen(tx, ty) is the diamond's top corner; the
                # image's top corner sits at (tw/2, 0) within its own canvas.
                sx, sy = cam.world_to_screen(tx, ty)
                blit(tiles[int(row[tx])], (sx - tw // 2, sy))

    def _draw_world_objects(self, world: World, selected: set[int]) -> None:
        """Decor, buildings and units, back-to-front in one depth order.

        Depth is just ``x + y`` at each thing's ground contact point: higher
        means closer to the camera under this projection, so sorting ascending
        and painting in that order gives correct occlusion for free, the same
        trick every isometric renderer uses instead of a real z-buffer.
        """
        cam = self.camera
        view = self.surface.get_rect()
        items: list[tuple[float, object]] = []

        if self.assets.decor:
            x0, y0, x1, y1 = cam.visible_tiles
            m = DECOR_MARGIN_TILES
            dx0, dy0 = max(0, x0 - m), max(0, y0 - m)
            dx1, dy1 = min(world.width, x1 + m), min(world.height, y1 + m)
            seed = world.seed
            decor = self.assets.decor
            for ty in range(dy0, dy1):
                row = world.terrain[ty]
                for tx in range(dx0, dx1):
                    variants = decor.get(int(row[tx]))
                    if not variants:
                        continue
                    sprite = variants[variant_index(tx, ty, seed, len(variants))]
                    items.append((tx + ty + 1, ("decor", sprite, tx, ty)))

        for b in world.buildings.values():
            # The footprint's *centre*, not its far corner: a unit's depth is
            # its own continuous x+y, so comparing against the building's
            # near edge would keep it "behind" the whole way across the
            # footprint, and its far edge would keep it "in front" the whole
            # way in. The centre is the one point where crossing the building
            # visually and crossing it in depth happen at the same place.
            items.append((b.x + b.w / 2 + b.y + b.h / 2, ("building", b)))

        for uid in sorted(world.units):
            u = world.units[uid]
            items.append((u.x + u.y, ("unit", u, uid)))

        items.sort(key=lambda item: item[0])
        for _, payload in items:
            kind = payload[0]
            if kind == "decor":
                self._draw_decor_sprite(payload[1], payload[2], payload[3])
            elif kind == "building":
                self._draw_building(payload[1], view)
            else:
                self._draw_unit(world, payload[1], payload[2] in selected, view)

    def _draw_decor_sprite(self, sprite: pygame.Surface, tx: int, ty: int) -> None:
        cam = self.camera
        # Anchored at the tile's front (south) corner, standing on the ground
        # the way the Town Center art does.
        ax, ay = cam.world_to_screen(tx + 1, ty + 1)
        self.surface.blit(sprite, (ax - sprite.get_width() // 2, ay - sprite.get_height()))

    def _draw_building(self, b: Building, view: pygame.Rect) -> None:
        cam = self.camera
        # The footprint's four ground corners, in perimeter order -- a
        # rhombus for anything wider than 1x1, not a screen-aligned rectangle.
        points = [
            cam.world_to_screen(b.x, b.y),
            cam.world_to_screen(b.x + b.w, b.y),
            cam.world_to_screen(b.x + b.w, b.y + b.h),
            cam.world_to_screen(b.x, b.y + b.h),
        ]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = pygame.Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if not bbox.colliderect(view):
            return
        colour = PLAYER_COLOURS[b.owner % len(PLAYER_COLOURS)]

        sprite = None
        if b.kind is BuildingKind.TOWN_CENTER:
            # Drawn larger than the plot it occupies and standing on its
            # front corner, so it reads as a building on the ground rather
            # than a flat diamond the size of its footprint.
            sprite = self.assets.town_center(
                int(bbox.width * self.cfg.building_scale),
                int(bbox.height * self.cfg.building_scale),
            )

        if sprite is None:
            pygame.draw.polygon(self.surface, _shade(colour, 0.55), points)
            pygame.draw.polygon(self.surface, colour, points, width=3)
            # No art to tell buildings apart by silhouette, so stamp the
            # kind's tag on the fallback diamond.
            tag = self._building_font.render(BUILDING_SPECS[b.kind].abbr, True, colour)
            self.surface.blit(tag, tag.get_rect(center=bbox.center))
            # Overlays belong on both paths -- without this, selection and
            # training progress disappear whenever the art is missing.
            self._draw_building_overlays(b, bbox)
            return

        self.surface.blit(sprite, (
            bbox.centerx - sprite.get_width() // 2,
            bbox.bottom - sprite.get_height(),
        ))
        # Ownership still has to be legible. An outline round the plot gets
        # drawn straight across the building's face, so use a colour bar at
        # its base instead -- visible, and it leaves the artwork alone. Half
        # the footprint's width, not all of it, so the bar sits inside the
        # diamond's front edges rather than overhanging them.
        bar_h = max(3, cam.tile_h // 6)
        bar = pygame.Rect(0, 0, max(8, bbox.width // 2), bar_h)
        bar.midtop = (bbox.centerx, bbox.bottom - bar_h)
        pygame.draw.rect(self.surface, colour, bar, border_radius=2)
        pygame.draw.rect(self.surface, _shade(colour, 0.5), bar, width=1,
                         border_radius=2)
        self._draw_building_overlays(b, bbox)

    def _draw_building_overlays(self, b: Building, footprint: pygame.Rect) -> None:
        """Training progress.

        Hugs the footprint's screen *bounding box*, not the artwork: the
        footprint is what you click (see ``App._building_under`` via
        ``Camera.screen_to_tile``) and what a unit walks out of, so anchoring
        to anything else would misreport where the building actually is.
        """
        if not b.queue:
            return
        total = max(1, UNIT_SPECS[b.queue[0]].train_ticks)
        frac = max(0.0, min(1.0, b.train_timer / total))
        bar = pygame.Rect(footprint.x, footprint.top - 8, footprint.width, 5)
        pygame.draw.rect(self.surface, (24, 26, 28), bar, border_radius=2)
        pygame.draw.rect(self.surface, (120, 190, 120),
                         (bar.x, bar.y, max(2, int(bar.width * frac)), bar.height),
                         border_radius=2)

    def _draw_rally_point(self, world: World, selected_building: int | None) -> None:
        """The selected building's rally point: a line to a marked tile.

        Only while it is selected. Drawing every building's point at once
        would criss-cross the map with lines nobody asked to see, and since
        you can only select your own, it never shows where an opponent is
        sending its production either.
        """
        if selected_building is None:
            return
        b = world.buildings.get(selected_building)
        if b is None or b.rally is None:
            return
        cam = self.camera
        tx, ty = b.rally
        start = cam.world_to_screen(b.x + b.w, b.y + b.h)
        end = cam.world_to_screen(tx + 0.5, ty + 0.5)
        pygame.draw.line(self.surface, RALLY_COLOUR, start, end, 1)
        pygame.draw.polygon(self.surface, RALLY_COLOUR,
                            _tile_diamond(cam, tx, ty), width=2)

    def _draw_paths(self, world: World, selected: set[int]) -> None:
        cam = self.camera
        for uid in selected:
            u = world.units.get(uid)
            if u is None or not u.path:
                continue
            points = [cam.world_to_screen(u.x, u.y)]
            points += [cam.world_to_screen(px + 0.5, py + 0.5) for px, py in u.path]
            if len(points) > 1:
                pygame.draw.lines(self.surface, PATH_COLOUR, False, points, 1)
            if u.target is not None:
                tx, ty = u.target
                pygame.draw.polygon(self.surface, PATH_COLOUR,
                                    _tile_diamond(cam, tx, ty), width=1)

    def _draw_unit(self, world: World, u: Unit, is_selected: bool,
                   view: pygame.Rect) -> None:
        cam = self.camera
        radius = unit_radius(cam.tile_size)
        sx, sy = cam.world_to_screen(u.x, u.y)
        if not view.collidepoint(sx, sy):
            return
        colour = PLAYER_COLOURS[u.owner % len(PLAYER_COLOURS)]
        if is_selected:
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
