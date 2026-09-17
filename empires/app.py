"""The human-facing game: a pygame window driving the simulation.

The important detail is the loop at the bottom. Rendering runs as fast as the
display allows; the simulation runs at a **fixed** ``ticks_per_second``,
decoupled via an accumulator. So the game plays identically on a 60 Hz laptop
and a 144 Hz monitor, and -- more to the point -- the tick rate you train an
agent at is the tick rate you play at.

Input never touches the world directly. It produces the same ``Command``
objects an RL policy emits, queued until the next tick.
"""

from __future__ import annotations

import pygame

from .config import RenderConfig, SimConfig
from .core.commands import CancelTrain, Command, Gather, Move, SetRally, Stop, Train
from .core.entities import Order, UnitKind
from .core.terrain import is_harvestable
from .core.world import World
from .render.camera import Camera
from .render.hud import Hud
from .render.renderer import Renderer, unit_radius

DRAG_THRESHOLD = 6      # pixels of travel below which a drag counts as a click
PICK_MARGIN = 3         # forgiveness added to a unit's drawn radius when clicking
SAME_SPOT_RADIUS = 8    # two clicks this close count as aimed at the same place
DOUBLE_CLICK_MS = 400


class App:
    def __init__(self, sim_cfg: SimConfig | None = None,
                 render_cfg: RenderConfig | None = None,
                 seed: int = 0, player: int = 0) -> None:
        self.sim_cfg = sim_cfg or SimConfig()
        self.cfg = render_cfg or RenderConfig()
        self.world = World(self.sim_cfg, seed=seed)
        self.player = player

        pygame.init()
        pygame.display.set_caption("EMPIRES UNDER CONSTRUCTION")
        self.screen = pygame.display.set_mode(
            (self.cfg.window_width, self.cfg.window_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        self.viewport = self.screen.subsurface(self._viewport_rect())
        self.camera = Camera(
            self.world.width, self.world.height, self.cfg.tile_size,
            self.viewport.get_width(), self.viewport.get_height(),
        )
        self.renderer = Renderer(self.viewport, self.camera, self.cfg)
        self.hud = Hud(self.screen, self._panel_rect())

        self.selected: set[int] = set()
        # Buildings select separately from units: an RTS never has both at
        # once, and the panel shows completely different things for each.
        self.selected_building: int | None = None
        self.pending: list[Command] = []
        self.paused = False
        self.speed_multiplier = 1
        self.running = True
        self._drag_start: tuple[int, int] | None = None
        self._accumulator = 0.0
        # Click bookkeeping. Cycling through a stack and detecting a
        # double-click both need to know where and when you last clicked.
        self._last_click_ms = 0
        self._last_click_pos: tuple[int, int] | None = None
        self._last_pick_pos: tuple[int, int] | None = None
        self._last_pick_uid: int | None = None

        self._center_on_home()

    # ------------------------------------------------------------- layout

    def _viewport_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(0, 0, w, max(1, h - self.cfg.panel_height))

    def _panel_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(0, h - self.cfg.panel_height, w, self.cfg.panel_height)

    def _on_resize(self, size: tuple[int, int]) -> None:
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.viewport = self.screen.subsurface(self._viewport_rect())
        self.camera.resize(self.viewport.get_width(), self.viewport.get_height())
        self.renderer.surface = self.viewport
        self.hud.surface = self.screen
        self.hud.rect = self._panel_rect()

    def _center_on_home(self) -> None:
        for b in self.world.buildings.values():
            if b.owner == self.player:
                self.camera.center_on_tile(*b.center)
                return

    # -------------------------------------------------------------- input

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self._on_resize((event.w, event.h))
            elif event.type == pygame.KEYDOWN:
                self._on_key(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._on_mouse_up(event)

    def _on_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            self._center_on_home()
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_f:
            # Fast-forward. Handy for watching a long gathering run, and the
            # same mechanism an eval harness uses to run faster than real time.
            self.speed_multiplier = 1 if self.speed_multiplier > 1 else 8
        elif event.key == pygame.K_e:
            self.selected = {
                u.uid for u in self.world.units_of(self.player)
                if u.order is Order.IDLE and u.spec.can_gather
            }
        elif event.key == pygame.K_v:
            self._queue_villager()
        elif event.key == pygame.K_x:
            # Same split the command card draws: cancel production at a
            # building, stop whatever units are selected.
            if self.selected_building is not None:
                self._cancel_training()
            else:
                self._stop_selected()

    def _on_mouse_down(self, event: pygame.event.Event) -> None:
        if self.hud.rect.collidepoint(event.pos):
            if event.button == 1:
                self._on_panel_click(event.pos)
            return
        if not self.viewport.get_rect().collidepoint(event.pos):
            return
        if event.button == 1:
            self._drag_start = event.pos
        elif event.button == 3:
            self._issue_order(event.pos)

    def _on_mouse_up(self, event: pygame.event.Event) -> None:
        if event.button != 1 or self._drag_start is None:
            return
        start, end = self._drag_start, event.pos
        self._drag_start = None
        additive = pygame.key.get_mods() & pygame.KMOD_SHIFT

        if max(abs(end[0] - start[0]), abs(end[1] - start[1])) < DRAG_THRESHOLD:
            self._select_click(end, additive)
        else:
            self._select_box(pygame.Rect(
                min(start[0], end[0]), min(start[1], end[1]),
                abs(end[0] - start[0]), abs(end[1] - start[1]),
            ), additive)

    def _units_under(self, pos: tuple[int, int]) -> list[int]:
        """Your unit ids whose drawn circle covers ``pos``, nearest first.

        Hit-testing against the *drawn* circle rather than the unit's tile is
        what makes clicking a moving unit reliable: a unit part-way between
        tiles is drawn overlapping its neighbour, and a tile lookup would miss
        it. Ties break on uid so the order is stable between clicks, which is
        what lets cycling below work.
        """
        r = unit_radius(self.camera.tile_size) + PICK_MARGIN
        hits: list[tuple[int, int]] = []
        for u in self.world.units_of(self.player):
            sx, sy = self.camera.world_to_screen(u.x, u.y)
            d2 = (sx - pos[0]) ** 2 + (sy - pos[1]) ** 2
            if d2 <= r * r:
                hits.append((d2, u.uid))
        hits.sort()
        return [uid for _, uid in hits]

    def _building_under(self, pos: tuple[int, int]) -> int | None:
        """Your building at ``pos``, if any.

        Tile lookup rather than the pixel test used for units: a building fills
        whole tiles, so its footprint *is* its clickable area -- and the art is
        drawn taller than the plot, so hit-testing the sprite would let you
        select it by clicking the ground several tiles above.
        """
        tile = self.camera.screen_to_tile(*pos)
        b = self.world.building_at(tile)
        return b.bid if b is not None and b.owner == self.player else None

    def _select_click(self, pos: tuple[int, int], additive: bool,
                      now_ms: int | None = None) -> None:
        now = pygame.time.get_ticks() if now_ms is None else now_ms
        is_double = (
            now - self._last_click_ms <= DOUBLE_CLICK_MS
            and _near(pos, self._last_click_pos, SAME_SPOT_RADIUS)
        )
        self._last_click_ms = now
        self._last_click_pos = pos

        candidates = self._units_under(pos)
        if not candidates:
            # Units win ties: a villager standing at the door of its Town
            # Center should be clickable without having to walk it away first.
            bid = self._building_under(pos)
            if bid is not None:
                self.selected.clear()
                self.selected_building = bid
                self._last_pick_uid = None
                return
            if not additive:
                self.selected.clear()
            self._last_pick_uid = None
            self.selected_building = None
            return

        self.selected_building = None

        if is_double:
            # Double-click grabs the whole type on screen, as an RTS player
            # expects. Note this takes priority over cycling, so walking a
            # stack means clicking at a normal pace, not hammering.
            self._select_kind_on_screen(self.world.units[candidates[0]].kind, additive)
            self._last_pick_uid = None
            self.selected_building = None
            return

        # Units do not collide, so several routinely sit on the same spot.
        # Clicking the same place again advances to the next one instead of
        # re-picking the one already selected.
        uid = candidates[0]
        if (self._last_pick_uid in candidates
                and _near(pos, self._last_pick_pos, SAME_SPOT_RADIUS)):
            nxt = candidates.index(self._last_pick_uid) + 1
            uid = candidates[nxt % len(candidates)]
        self._last_pick_pos = pos
        self._last_pick_uid = uid

        if additive:
            # Shift-click toggles, so you can drop one unit from a group.
            self.selected ^= {uid}
        else:
            self.selected = {uid}

    def _select_kind_on_screen(self, kind: UnitKind, additive: bool) -> None:
        view = self.viewport.get_rect()
        ids = set()
        for u in self.world.units_of(self.player):
            if u.kind is not kind:
                continue
            sx, sy = self.camera.world_to_screen(u.x, u.y)
            if view.collidepoint(sx, sy):
                ids.add(u.uid)
        self.selected = (self.selected | ids) if additive else ids

    def _select_box(self, rect: pygame.Rect, additive: bool) -> None:
        if not additive:
            self.selected.clear()
        for u in self.world.units_of(self.player):
            sx, sy = self.camera.world_to_screen(u.x, u.y)
            if rect.collidepoint(sx, sy):
                self.selected.add(u.uid)
        # A box replaces whatever the click cycle was walking through.
        self._last_pick_uid = None
        if self.selected:
            self.selected_building = None

    def _on_panel_click(self, pos: tuple[int, int]) -> None:
        """Route a click on the command card to the same action its key fires.

        A greyed slot is still clickable, for the same reason the hotkey
        still fires: the click only queues a Command, and the simulation
        decides on the next tick whether it was legal.
        """
        for slot in self.hud.commands:
            if slot.rect.collidepoint(pos):
                self._run_command(slot.action)
                return

    def _run_command(self, action: str) -> None:
        if action == "train":
            self._queue_villager()
        elif action == "cancel":
            self._cancel_training()
        elif action == "stop":
            self._stop_selected()

    def _stop_selected(self) -> None:
        if self.selected:
            self.pending.append(Stop(self.player, tuple(sorted(self.selected))))

    def _queue_villager(self) -> None:
        """Queue a Villager at the selected building.

        Like every other input this only appends a Command; the simulation
        decides on the next tick whether it is affordable. The HUD greys the
        button out, but that is a hint, not the rule -- the sim stays the single
        authority on what is legal.
        """
        if self.selected_building is not None:
            self.pending.append(
                Train(self.player, self.selected_building, UnitKind.VILLAGER)
            )

    def _cancel_training(self) -> None:
        if self.selected_building is not None:
            self.pending.append(CancelTrain(self.player, self.selected_building))

    def _issue_order(self, pos: tuple[int, int]) -> None:
        """Right-click: gather if the tile holds a resource, otherwise move.

        This "smart order" is deliberately the same rule the RL action space
        uses, so a policy and a person express intent identically. With a
        building selected the same click sets its rally point instead, which
        is the same rule again, applied to units that do not exist yet.
        """
        tile = self.camera.screen_to_tile(*pos)
        if not self.world.in_bounds(tile):
            return
        if self.selected_building is not None:
            self.pending.append(SetRally(self.player, self.selected_building, tile))
            return
        if not self.selected:
            return
        ids = tuple(sorted(self.selected))
        tx, ty = tile
        if is_harvestable(self.world.terrain[ty, tx]) and self.world.resources[ty, tx] > 0:
            self.pending.append(Gather(self.player, ids, tile))
        else:
            self.pending.append(Move(self.player, ids, tile))

    def _pan(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        dx = dy = 0.0
        speed = self.cfg.camera_speed * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += speed
        if self.cfg.edge_scroll_margin and pygame.mouse.get_focused():
            mx, my = pygame.mouse.get_pos()
            m = self.cfg.edge_scroll_margin
            if mx < m:
                dx -= speed
            elif mx > self.viewport.get_width() - m:
                dx += speed
            if my < m:
                dy -= speed
            elif my > self.viewport.get_height() - m:
                dy += speed
        if dx or dy:
            self.camera.move(dx, dy)

    # --------------------------------------------------------------- loop

    def run(self) -> None:
        sim_dt = 1.0 / self.sim_cfg.ticks_per_second
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.fps) / 1000.0
                # Clamp dt so that dragging the window or hitting a breakpoint
                # does not make the sim try to catch up hundreds of ticks at
                # once (the classic "spiral of death").
                dt = min(dt, 0.25)

                self._handle_events()
                self._pan(dt)

                if not self.paused:
                    self._accumulator += dt * self.speed_multiplier
                    while self._accumulator >= sim_dt:
                        self.world.step(self.pending)
                        self.pending = []
                        self._accumulator -= sim_dt
                        # Drop selections for things that no longer exist.
                        self.selected &= self.world.units.keys()
                        if self.selected_building not in self.world.buildings:
                            self.selected_building = None

                drag_rect = None
                if self._drag_start is not None:
                    end = pygame.mouse.get_pos()
                    drag_rect = pygame.Rect(
                        min(self._drag_start[0], end[0]), min(self._drag_start[1], end[1]),
                        abs(end[0] - self._drag_start[0]), abs(end[1] - self._drag_start[1]),
                    )

                self.renderer.draw(self.world, self.selected, self.player,
                                   drag_rect, self.selected_building)
                self.hud.draw(self.world, self.camera, self.selected,
                              self.player, self.clock.get_fps(),
                              self.selected_building)
                pygame.display.flip()
        finally:
            pygame.quit()


def _near(a: tuple[int, int], b: tuple[int, int] | None, radius: int) -> bool:
    return b is not None and abs(a[0] - b[0]) <= radius and abs(a[1] - b[1]) <= radius
