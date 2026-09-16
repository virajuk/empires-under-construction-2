"""Bottom panel: resources, selection summary, and a minimap."""

from __future__ import annotations

import pygame

from ..core.entities import Order
from ..core.terrain import Resource
from ..core.world import World
from .camera import Camera
from .renderer import PLAYER_COLOURS, TERRAIN_COLOURS

PANEL_BG = (28, 30, 34)
PANEL_LINE = (58, 62, 68)
TEXT = (226, 226, 220)
DIM = (150, 152, 148)

RESOURCE_LABELS = {
    Resource.FOOD: "Food",
    Resource.WOOD: "Wood",
    Resource.GOLD: "Gold",
    Resource.STONE: "Stone",
}


class Hud:
    def __init__(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.surface = surface
        self.rect = rect
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 12)
        self._minimap: pygame.Surface | None = None
        self._minimap_tick = -1

    def draw(self, world: World, camera: Camera, selected: set[int],
             player: int, fps: float) -> None:
        pygame.draw.rect(self.surface, PANEL_BG, self.rect)
        pygame.draw.line(self.surface, PANEL_LINE,
                         self.rect.topleft, self.rect.topright, 2)

        x = self.rect.x + 14
        y = self.rect.y + 12

        res = world.players[player].resources
        for kind, label in RESOURCE_LABELS.items():
            text = f"{label}: {res[int(kind)]}"
            self.surface.blit(self.font.render(text, True, TEXT), (x, y))
            y += 19

        x += 150
        y = self.rect.y + 12
        sel = [world.units[u] for u in sorted(selected) if u in world.units]
        lines = [
            f"Selected: {len(sel)}",
            f"Gathering: {sum(1 for u in sel if u.order in (Order.GATHER, Order.RETURN))}",
            f"Idle: {sum(1 for u in sel if u.order is Order.IDLE)}",
            f"Carrying: {sum(u.carrying for u in sel)}",
        ]
        for line in lines:
            self.surface.blit(self.font.render(line, True, TEXT), (x, y))
            y += 19

        x += 170
        y = self.rect.y + 12
        info = [
            f"Tick: {world.tick}",
            f"FPS: {fps:5.1f}",
            f"Seed: {world.seed}",
            "LMB select  RMB order  WASD pan  E idle  X stop  F fast  P pause",
        ]
        for i, line in enumerate(info):
            self.surface.blit(self.small.render(line, True, DIM if i else TEXT), (x, y))
            y += 17

        self._draw_minimap(world, camera, player)

    # ------------------------------------------------------------- minimap

    def _draw_minimap(self, world: World, camera: Camera, player: int) -> None:
        # Fit the map inside the panel height while preserving its aspect, so a
        # wide map does not come out stretched.
        box = self.rect.height - 16
        scale = min(box / world.width, box / world.height)
        mm_w, mm_h = max(1, int(world.width * scale)), max(1, int(world.height * scale))
        mm_rect = pygame.Rect(self.rect.right - mm_w - 14, self.rect.y + 8, mm_w, mm_h)

        # The terrain layer only changes when a resource tile is exhausted, so
        # redraw it a few times a second rather than every frame.
        if self._minimap is None or world.tick - self._minimap_tick > 20:
            self._minimap = self._render_terrain_minimap(world, mm_w, mm_h)
            self._minimap_tick = world.tick

        self.surface.blit(self._minimap, mm_rect.topleft)

        sx = mm_w / world.width
        sy = mm_h / world.height
        for uid in sorted(world.units):
            u = world.units[uid]
            colour = PLAYER_COLOURS[u.owner % len(PLAYER_COLOURS)]
            self.surface.fill(colour, (mm_rect.x + int(u.x * sx), mm_rect.y + int(u.y * sy), 2, 2))
        for b in world.buildings.values():
            colour = PLAYER_COLOURS[b.owner % len(PLAYER_COLOURS)]
            self.surface.fill(colour, (mm_rect.x + int(b.x * sx), mm_rect.y + int(b.y * sy), 4, 4))

        # Viewport box.
        ts = camera.tile_size
        view = pygame.Rect(
            mm_rect.x + int(camera.x / ts * sx),
            mm_rect.y + int(camera.y / ts * sy),
            max(2, int(camera.view_w / ts * sx)),
            max(2, int(camera.view_h / ts * sy)),
        )
        pygame.draw.rect(self.surface, TEXT, view, width=1)
        pygame.draw.rect(self.surface, PANEL_LINE, mm_rect, width=1)

    def _render_terrain_minimap(self, world: World, w: int, h: int) -> pygame.Surface:
        small = pygame.Surface((world.width, world.height))
        for terrain, colour in TERRAIN_COLOURS.items():
            ys, xs = (world.terrain == int(terrain)).nonzero()
            for x, y in zip(xs.tolist(), ys.tolist()):
                small.set_at((x, y), colour)
        return pygame.transform.scale(small, (w, h))
