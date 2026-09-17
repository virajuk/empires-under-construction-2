"""Bottom panel: resources, selection summary, and a minimap."""

from __future__ import annotations

import pygame

from collections import Counter
from dataclasses import dataclass

from ..core.entities import UNIT_SPECS, unit_label
from ..core.terrain import Resource
from ..core.world import World
from .camera import Camera
from .renderer import PLAYER_COLOURS, TERRAIN_COLOURS

PANEL_BG = (28, 30, 34)
PANEL_LINE = (58, 62, 68)
CARD_BG = (34, 37, 42)
SLOT_EMPTY = (24, 26, 30)
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

# Keyboard shortcuts. App owns the key events themselves; these are the
# labels, so a button, the hint line and the key that actually fires cannot
# drift apart. ``CANCEL_KEY`` does double duty exactly as App's handler does:
# cancel production at a building, stop whatever units are selected.
TRAIN_KEY = "V"
CANCEL_KEY = "X"

# Command card: a grid of slots, most of them empty, in the manner of an RTS
# command card. Slots carry no icon art, so the shortcut key is the face of
# the button and a slot only has to be big enough to read one letter.
COMMAND_SLOT = 40
COMMAND_GAP = 5
COMMAND_COLS = 6
COMMAND_ROWS = 2
CARD_PAD = 10
# Left half of the card: what is selected. The widest line it has to hold is
# a mixed selection, "12 Villagers, 3 Soldiers".
CARD_INFO_W = 170
CARD_W = CARD_PAD * 3 + CARD_INFO_W + COMMAND_COLS * COMMAND_SLOT + (COMMAND_COLS - 1) * COMMAND_GAP


@dataclass(frozen=True)
class CommandSlot:
    """One command button: where it is, and what App should do about it.

    ``enabled`` is presentation only. Clicking a greyed slot still sends the
    command -- the simulation is the single authority on what is legal, and
    the HUD greying things out is a hint, not the rule.
    """

    rect: pygame.Rect
    action: str
    enabled: bool


class Hud:
    def __init__(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.surface = surface
        self.rect = rect
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 12)
        # For the shortcut letter on a command button, which stands in for
        # the icon art this HUD does not have.
        self.key_font = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)
        self._minimap: pygame.Surface | None = None
        self._minimap_tick = -1
        # Where this frame's command buttons landed, for App to hit-test.
        # Rebuilt every frame, so a slot from a selection you no longer have
        # cannot stay live.
        self.commands: list[CommandSlot] = []
        # The train slot's rect on its own, which is the one App and its
        # tests reach for by name. None when nothing that trains is selected.
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
        # The command card follows the resource column and carries both what
        # is selected and what it can do, so those two never drift apart.
        card = pygame.Rect(self.rect.x + 150, self.rect.y + 8,
                           CARD_W, self.rect.height - 16)
        col_status = card.right + 16

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
        building = world.buildings.get(selected_building) if selected_building else None
        units = [world.units[u] for u in sorted(selected) if u in world.units]
        self._draw_command_card(world, card, building, units)

        x = col_status
        y = self.rect.y + 12
        info = [
            f"Tick: {world.tick}",
            f"FPS: {fps:5.1f}",
            f"Seed: {world.seed}",
            f"LMB select  RMB order  {TRAIN_KEY} train  E idle  P pause",
        ]
        for i, line in enumerate(info):
            self.surface.blit(self.small.render(line, True, DIM if i else TEXT), (x, y))
            y += 17

        self._draw_minimap(world, camera, player)

    # ---------------------------------------------------------- command card

    def _draw_command_card(self, world: World, card: pygame.Rect, building,
                           units: list) -> None:
        """What is selected, and what it can be told to do.

        The frame is drawn whether or not anything is selected, so the blocks
        either side of it do not shuffle about as the selection changes.
        """
        self.commands = []
        self.train_button = None

        pygame.draw.rect(self.surface, CARD_BG, card, border_radius=4)
        pygame.draw.rect(self.surface, PANEL_LINE, card, width=1, border_radius=4)

        info_x = card.x + CARD_PAD
        if building is not None:
            self._draw_building_info(info_x, card.y + 8, building)
        elif units:
            self._draw_unit_info(info_x, card.y + 8, units)

        self._draw_command_grid(world, card, building, units)

    def _draw_building_info(self, x: int, y: int, building) -> None:
        self.surface.blit(self.font.render(building.label, True, TEXT), (x, y))
        y += 19
        self.surface.blit(
            self.small.render(f"HP: {building.hp}/{building.spec.hp}", True, DIM), (x, y))
        y += 16

        if building.queue:
            kind = building.queue[0]
            total = max(1, UNIT_SPECS[kind].train_ticks)
            pct = min(100, int(100 * building.train_timer / total))
            self.surface.blit(
                self.small.render(f"Training {unit_label(kind)}  {pct}%", True, TEXT),
                (x, y),
            )
            y += 15
            self._progress_bar(x, y, CARD_INFO_W, 6, building.train_timer / total)
            y += 12
            if len(building.queue) > 1:
                self.surface.blit(
                    self.small.render(f"Queued: {len(building.queue) - 1}", True, DIM),
                    (x, y),
                )
        else:
            self.surface.blit(self.small.render("Idle", True, DIM), (x, y))

    def _draw_unit_info(self, x: int, y: int, units: list) -> None:
        self.surface.blit(self.font.render(_selection_summary(units), True, TEXT), (x, y))
        y += 19
        # Hit points only for a lone unit: for a group the number would have
        # to stand for several different ones at once.
        if len(units) == 1:
            u = units[0]
            self.surface.blit(
                self.small.render(f"HP: {u.hp}/{u.spec.hp}", True, DIM), (x, y))
            y += 16
        self.surface.blit(
            self.small.render(f"Carrying: {sum(u.carrying for u in units)}", True, TEXT),
            (x, y))

    def _draw_command_grid(self, world: World, card: pygame.Rect, building,
                           units: list) -> None:
        grid_x = card.x + CARD_PAD * 2 + CARD_INFO_W
        grid_y = card.y + CARD_PAD

        actions = _actions_for(world, building, units)
        for row in range(COMMAND_ROWS):
            for col in range(COMMAND_COLS):
                rect = pygame.Rect(
                    grid_x + col * (COMMAND_SLOT + COMMAND_GAP),
                    grid_y + row * (COMMAND_SLOT + COMMAND_GAP),
                    COMMAND_SLOT, COMMAND_SLOT,
                )
                index = row * COMMAND_COLS + col
                if index >= len(actions):
                    # An empty slot, drawn rather than left blank so the card
                    # reads as a command grid with room left in it.
                    pygame.draw.rect(self.surface, SLOT_EMPTY, rect, border_radius=3)
                    pygame.draw.rect(self.surface, PANEL_LINE, rect, width=1, border_radius=3)
                    continue
                action, key, enabled, _ = actions[index]
                self._draw_slot(rect, key, enabled)
                self.commands.append(CommandSlot(rect, action, enabled))
                if action == "train":
                    self.train_button = rect

        # One legend line under the grid says what the keys on it do. Per-slot
        # captions would have to be cut down to fit a 40px square.
        if actions:
            legend = "   ".join(caption for *_, caption in actions)
            self.surface.blit(
                self.small.render(legend, True, DIM),
                (grid_x, grid_y + COMMAND_ROWS * (COMMAND_SLOT + COMMAND_GAP) + 2),
            )

    def _draw_slot(self, rect: pygame.Rect, key: str, enabled: bool) -> None:
        pygame.draw.rect(self.surface, BUTTON_BG if enabled else BUTTON_OFF,
                         rect, border_radius=3)
        pygame.draw.rect(self.surface, BUTTON_EDGE if enabled else BUTTON_OFF,
                         rect, width=1, border_radius=3)
        # No icon art, so the shortcut key is the face of the button.
        face = self.key_font.render(key, True, TEXT if enabled else TEXT_OFF)
        self.surface.blit(face, face.get_rect(center=rect.center))

    def _progress_bar(self, x: int, y: int, w: int, h: int, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        pygame.draw.rect(self.surface, BUTTON_OFF, (x, y, w, h), border_radius=3)
        if frac > 0:
            pygame.draw.rect(self.surface, (120, 190, 120),
                             (x, y, max(2, int(w * frac)), h), border_radius=3)

    # ------------------------------------------------------------- minimap

    def _minimap_rect(self, world: World) -> pygame.Rect:
        """Where the minimap sits, pinned to the panel's right corner.

        Its own method because the command column has to be placed against
        its left edge, and a second copy of this arithmetic is exactly how
        the two ended up overlapping.
        """
        # Fit the map inside the panel height while preserving its aspect, so a
        # wide map does not come out stretched.
        box = self.rect.height - 16
        scale = min(box / world.width, box / world.height)
        mm_w, mm_h = max(1, int(world.width * scale)), max(1, int(world.height * scale))
        return pygame.Rect(self.rect.right - mm_w - 14, self.rect.y + 8, mm_w, mm_h)

    def _draw_minimap(self, world: World, camera: Camera, player: int) -> None:
        mm_rect = self._minimap_rect(world)
        mm_w, mm_h = mm_rect.width, mm_rect.height

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


def _actions_for(world: World, building, units: list) -> list[tuple[str, str, bool, str]]:
    """``(action, key, enabled, caption)`` for what this selection can do.

    A building's own spec says what it can make, so a second kind of
    building needs nothing added here. ``enabled`` drives the greying only;
    App still sends the command, and the simulation still decides.
    """
    if building is not None:
        out = [
            ("train", TRAIN_KEY, world.can_train(building.owner, building.bid, kind),
             f"{TRAIN_KEY} {UNIT_SPECS[kind].label} {_cost_text(UNIT_SPECS[kind].cost)}")
            for kind in building.spec.trains
        ]
        out.append(("cancel", CANCEL_KEY, bool(building.queue), f"{CANCEL_KEY} Cancel"))
        return out
    if units:
        return [("stop", CANCEL_KEY, True, f"{CANCEL_KEY} Stop")]
    return []


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
