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
from .core.commands import Command, Gather, Move, Stop
from .core.entities import Order, UnitKind
from .core.terrain import is_harvestable
from .core.world import World
from .render.camera import Camera
from .render.hud import Hud
from .render.renderer import Renderer

DRAG_THRESHOLD = 6  # pixels; below this a drag counts as a click


class App:
    def __init__(self, sim_cfg: SimConfig | None = None,
                 render_cfg: RenderConfig | None = None,
                 seed: int = 0, player: int = 0) -> None:
        self.sim_cfg = sim_cfg or SimConfig()
        self.cfg = render_cfg or RenderConfig()
        self.world = World(self.sim_cfg, seed=seed)
        self.player = player

        pygame.init()
        pygame.display.set_caption("EMPIRES")
        self.screen = pygame.display.set_mode(
            (self.cfg.window_width, self.cfg.window_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        self.viewport = self.screen.subsurface(self._viewport_rect())
        self.camera = Camera(
            self.world.width, self.world.height, self.cfg.tile_size,
            self.viewport.get_width(), self.viewport.get_height(),
        )
        self.renderer = Renderer(self.viewport, self.camera)
        self.hud = Hud(self.screen, self._panel_rect())

        self.selected: set[int] = set()
        self.pending: list[Command] = []
        self.paused = False
        self.speed_multiplier = 1
        self.running = True
        self._drag_start: tuple[int, int] | None = None
        self._accumulator = 0.0

        self._centre_on_home()

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

    def _centre_on_home(self) -> None:
        for b in self.world.buildings.values():
            if b.owner == self.player:
                self.camera.centre_on_tile(*b.centre)
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
            self._centre_on_home()
        elif event.key == pygame.K_p:
            self.paused = not self.paused
        elif event.key == pygame.K_f:
            # Fast-forward. Handy for watching a long gathering run, and the
            # same mechanism an eval harness uses to run faster than real time.
            self.speed_multiplier = 1 if self.speed_multiplier > 1 else 8
        elif event.key == pygame.K_e:
            self.selected = {
                u.uid for u in self.world.units_of(self.player)
                if u.order is Order.IDLE and u.kind is UnitKind.WORKER
            }
        elif event.key == pygame.K_x:
            if self.selected:
                self.pending.append(Stop(self.player, tuple(sorted(self.selected))))

    def _on_mouse_down(self, event: pygame.event.Event) -> None:
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

    def _select_click(self, pos: tuple[int, int], additive: bool) -> None:
        tile = self.camera.screen_to_tile(*pos)
        unit = self.world.unit_at(tile, owner=self.player)
        if not additive:
            self.selected.clear()
        if unit is not None:
            self.selected.add(unit.uid)

    def _select_box(self, rect: pygame.Rect, additive: bool) -> None:
        if not additive:
            self.selected.clear()
        for u in self.world.units_of(self.player):
            sx, sy = self.camera.world_to_screen(u.x, u.y)
            if rect.collidepoint(sx, sy):
                self.selected.add(u.uid)

    def _issue_order(self, pos: tuple[int, int]) -> None:
        """Right-click: gather if the tile holds a resource, otherwise move.

        This "smart order" is deliberately the same rule the RL action space
        uses, so a policy and a person express intent identically.
        """
        if not self.selected:
            return
        tile = self.camera.screen_to_tile(*pos)
        if not self.world.in_bounds(tile):
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
                        # Drop selections for units that no longer exist.
                        self.selected &= self.world.units.keys()

                drag_rect = None
                if self._drag_start is not None:
                    end = pygame.mouse.get_pos()
                    drag_rect = pygame.Rect(
                        min(self._drag_start[0], end[0]), min(self._drag_start[1], end[1]),
                        abs(end[0] - self._drag_start[0]), abs(end[1] - self._drag_start[1]),
                    )

                self.renderer.draw(self.world, self.selected, self.player, drag_rect)
                self.hud.draw(self.world, self.camera, self.selected,
                              self.player, self.clock.get_fps())
                pygame.display.flip()
        finally:
            pygame.quit()
