"""Bottom panel: resources, selection summary, and a minimap."""

from __future__ import annotations

import pygame

from collections import Counter

from ..core.entities import UNIT_SPECS, UnitKind, unit_label
from ..core.terrain import Resource
from ..core.world import World
from .camera import Camera
from .renderer import PLAYER_COLOURS, TERRAIN_COLOURS

PANEL_BG = (28, 30, 34)
PANEL_LINE = (58, 62, 68)
TEXT = (226, 226, 220)
DIM = (150, 152, 148)
BUTTON_BG = (52, 58, 66)
BUTTON_BG_HOT = (70, 80, 92)
BUTTON_EDGE = (108, 116, 126)
BUTTON_OFF = (44, 46, 50)
TEXT_OFF = (112, 114, 118)

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
        # Where the Train button landed this frame, for App to hit-test.
        # None when no trainer is selected, so a stale rect cannot stay live.
        self.train_button: pygame.Rect | None = None

    def draw(self, world: World, camera: Camera, selected: set[int],
             player: int, fps: float, selected_building: int | None = None) -> None:
        pygame.draw.rect(self.surface, PANEL_BG, self.rect)
        pygame.draw.line(self.surface, PANEL_LINE,
                         self.rect.topleft, self.rect.topright, 2)

        # Fixed column origins rather than cumulative offsets: several blocks
        # are variable-width ("4 Villagers, 1 Soldier"), so advancing by a
        # guessed amount lets a long one run into the next column. Widths are
        # sized for the longest string each block can produce.
        col_resources = self.rect.x + 14
        col_selection = self.rect.x + 150     # .. 370: "12 Villagers, 34 Soldiers"
        col_status = self.rect.x + 530        # .. 800: the hint line
        # Actions are pinned to the right, clear of the minimap, so the button
        # does not move when the blocks to its left change width.
        col_actions = self.rect.right - 300

        x = col_resources
        y = self.rect.y + 12

        # Gatherer counts sit right next to the resource they're assigned to,
        # not off in a separate economy block, so "who's on wood" reads at a
        # glance against "how much wood". Idle and the running total close
        # out the column -- every Villager the player owns, selected or not,
        # so these must not depend on what happens to be highlighted.
        # "All villagers" rather than "Villagers" so the bottom row cannot be
        # misread as a count of the selection in the next column.
        act = world.villager_activity(player)
        res = world.players[player].resources
        for kind, label in RESOURCE_LABELS.items():
            text = f"{label}: {res[int(kind)]}"
            gatherers = act.gathering_by_resource[int(kind)]
            if gatherers:
                text += f" ({gatherers})"
            self.surface.blit(self.font.render(text, True, TEXT), (x, y))
            y += 19
        for line in (f"Idle: {act.idle}", f"All villagers: {act.total}"):
            self.surface.blit(self.font.render(line, True, TEXT), (x, y))
            y += 19

        # Selection block -- scoped to what is highlighted right now.
        self.train_button = None
        building = world.buildings.get(selected_building) if selected_building else None
        if building is not None:
            self._draw_building_panel(world, building, col_selection, col_actions)
        else:
            x = col_selection
            y = self.rect.y + 12
            sel = [world.units[u] for u in sorted(selected) if u in world.units]
            # Blank rather than "Nothing selected" -- there is nothing to say
            # about a selection that does not exist.
            if sel:
                for line in (_selection_summary(sel),
                             f"Carrying: {sum(u.carrying for u in sel)}"):
                    self.surface.blit(self.font.render(line, True, TEXT), (x, y))
                    y += 19

        x = col_status
        y = self.rect.y + 12
        info = [
            f"Tick: {world.tick}",
            f"FPS: {fps:5.1f}",
            f"Seed: {world.seed}",
            "LMB select  RMB order  V train  E idle  P pause",
        ]
        for i, line in enumerate(info):
            self.surface.blit(self.small.render(line, True, DIM if i else TEXT), (x, y))
            y += 17

        self._draw_minimap(world, camera, player)

    # ------------------------------------------------------- building panel

    def _draw_building_panel(self, world: World, building, x: int,
                             action_x: int) -> None:
        y = self.rect.y + 12
        self.surface.blit(self.font.render(building.label, True, TEXT), (x, y))
        y += 19

        if building.queue:
            kind = building.queue[0]
            total = max(1, UNIT_SPECS[kind].train_ticks)
            pct = min(100, int(100 * building.train_timer / total))
            self.surface.blit(
                self.small.render(f"Training {unit_label(kind)}  {pct}%", True, TEXT),
                (x, y),
            )
            y += 15
            self._progress_bar(x, y, 150, 6, building.train_timer / total)
            y += 12
            if len(building.queue) > 1:
                self.surface.blit(
                    self.small.render(f"Queued: {len(building.queue) - 1}", True, DIM),
                    (x, y),
                )
        else:
            self.surface.blit(self.small.render("Idle", True, DIM), (x, y))

        # Only offer what this building can actually make.
        for kind in building.spec.trains:
            self.train_button = self._draw_train_button(world, building, kind, action_x)
            break  # one button for now; a row of them when there is more to make

    def _draw_train_button(self, world: World, building, kind: UnitKind,
                           x: int) -> pygame.Rect:
        spec = UNIT_SPECS[kind]
        enabled = world.can_train(building.owner, building.bid, kind)
        rect = pygame.Rect(x, self.rect.y + 14, 190, self.rect.height - 30)

        pygame.draw.rect(self.surface, BUTTON_BG if enabled else BUTTON_OFF,
                         rect, border_radius=4)
        pygame.draw.rect(self.surface, BUTTON_EDGE if enabled else BUTTON_OFF,
                         rect, width=1, border_radius=4)

        fg = TEXT if enabled else TEXT_OFF
        # Abbreviated rather than the full building name: the button is narrow
        # and sits right next to the minimap, so "Town Center: Train Villager"
        # would run straight into it.
        label = self.font.render(f"{building.spec.abbr}: Train {spec.label}", True, fg)
        self.surface.blit(label, (rect.x + 12, rect.y + 8))
        cost = self.small.render(_cost_text(spec.cost) + "   [V]", True,
                                 DIM if enabled else TEXT_OFF)
        self.surface.blit(cost, (rect.x + 12, rect.y + 30))
        if not enabled:
            self.surface.blit(self.small.render("not enough food", True, TEXT_OFF),
                              (rect.x + 12, rect.y + 46))
        return rect

    def _progress_bar(self, x: int, y: int, w: int, h: int, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(self.surface, BUTTON_OFF, (x, y, w, h), border_radius=3)
        if frac > 0:
            pygame.draw.rect(self.surface, (120, 190, 120),
                             (x, y, max(2, int(w * frac)), h), border_radius=3)

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

        # Viewport box. The minimap stays top-down regardless of how the main
        # view is projected, so this maps each screen corner through
        # ``screen_to_world`` rather than assuming ``camera.x`` is a plain
        # multiple of a tile -- true for an orthogonal camera, not for an
        # isometric one, where the viewport back-projects to a rotated
        # quadrilateral rather than an axis-aligned box.
        corners = ((0, 0), (camera.view_w, 0), (camera.view_w, camera.view_h), (0, camera.view_h))
        points = []
        for cx, cy in corners:
            wx, wy = camera.screen_to_world(cx, cy)
            points.append((mm_rect.x + wx * sx, mm_rect.y + wy * sy))
        pygame.draw.polygon(self.surface, TEXT, points, width=1)
        pygame.draw.rect(self.surface, PANEL_LINE, mm_rect, width=1)

    def _render_terrain_minimap(self, world: World, w: int, h: int) -> pygame.Surface:
        small = pygame.Surface((world.width, world.height))
        for terrain, colour in TERRAIN_COLOURS.items():
            ys, xs = (world.terrain == int(terrain)).nonzero()
            for x, y in zip(xs.tolist(), ys.tolist()):
                small.set_at((x, y), colour)
        return pygame.transform.scale(small, (w, h))


def _selection_summary(units: list) -> str:
    """"4 Villagers", or "3 Villagers, 1 Soldier" for a mixed selection.

    Only called with a non-empty selection -- the HUD skips this block
    entirely rather than asking it to describe an empty one."""
    counts = Counter(u.kind for u in units)
    return ", ".join(
        f"{n} {unit_label(kind, n)}" for kind, n in sorted(counts.items())
    )


def _cost_text(cost: tuple[int, ...]) -> str:
    """"50 Food", or several resources joined, skipping the zeros."""
    parts = [
        f"{amount} {RESOURCE_LABELS[Resource(i)]}"
        for i, amount in enumerate(cost)
        if amount
    ]
    return "  ".join(parts) if parts else "Free"
