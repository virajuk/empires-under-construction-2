"""Drawing. Reads the world, writes pixels, mutates nothing.

Everything is flat colour rects -- no art assets, so the project runs the
moment it is cloned. When you swap in sprites, only this file changes.
"""

from __future__ import annotations

import pygame

from ..core.entities import Order, Unit
from ..core.terrain import Terrain
from ..core.world import World
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


class Renderer:
    def __init__(self, surface: pygame.Surface, camera: Camera) -> None:
        self.surface = surface
        self.camera = camera
        ts = camera.tile_size
        # One pre-filled surface per terrain type, blitted rather than
        # re-filled. Cheap, and it gives a place to hang textures later.
        self._tiles: dict[int, pygame.Surface] = {}
        for terrain, colour in TERRAIN_COLOURS.items():
            surf = pygame.Surface((ts, ts))
            if terrain is Terrain.BERRY:
                # A solid red square reads as something alarming. Draw bushes
                # sitting on grass instead, so the tile still says "food".
                surf.fill(TERRAIN_COLOURS[Terrain.GRASS])
                bush = _shade(colour, 0.55)
                for bx, by in ((0.32, 0.36), (0.66, 0.34), (0.5, 0.68)):
                    pygame.draw.circle(surf, bush, (int(bx * ts), int(by * ts)),
                                       max(2, ts // 5))
                for bx, by in ((0.32, 0.36), (0.66, 0.34), (0.5, 0.68)):
                    pygame.draw.circle(surf, colour, (int(bx * ts), int(by * ts)),
                                       max(1, ts // 9))
            else:
                surf.fill(colour)
            pygame.draw.line(surf, _shade(colour, 0.9), (0, ts - 1), (ts - 1, ts - 1))
            pygame.draw.line(surf, _shade(colour, 0.9), (ts - 1, 0), (ts - 1, ts - 1))
            self._tiles[int(terrain)] = surf

    def draw(self, world: World, selected: set[int], player: int,
             drag_rect: pygame.Rect | None = None) -> None:
        self.surface.fill((20, 24, 20))
        self._draw_terrain(world)
        self._draw_buildings(world)
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

    def _draw_buildings(self, world: World) -> None:
        cam = self.camera
        ts = cam.tile_size
        for b in world.buildings.values():
            sx, sy = cam.world_to_screen(b.x, b.y)
            rect = pygame.Rect(sx, sy, b.w * ts, b.h * ts)
            if not rect.colliderect(self.surface.get_rect()):
                continue
            colour = PLAYER_COLOURS[b.owner % len(PLAYER_COLOURS)]
            pygame.draw.rect(self.surface, _shade(colour, 0.55), rect)
            pygame.draw.rect(self.surface, colour, rect, width=3)

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
        radius = max(3, ts // 3)
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
