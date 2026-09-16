"""The simulation.

This module is the one that matters. It imports numpy and nothing else from
outside ``empires.core`` -- in particular **it never imports pygame**. That is
what lets the same code run at 60 FPS behind a window and at tens of thousands
of ticks per second inside an RL training loop.

The contract is small::

    world = World(SimConfig(), seed=0)
    world.step([Move(owner=0, unit_ids=(1,), target=(10, 12))])

``step`` advances exactly one tick. Given the same seed and the same command
sequence it produces the same state, every time -- see ``state_hash``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from . import mapgen
from .commands import Command, Gather, Move, Stop
from .entities import Building, Order, Player, Unit, UnitKind
from .pathfinding import find_path, path_to_adjacent
from .terrain import YIELDS, Terrain, is_harvestable, is_passable

Tile = tuple[int, int]


@dataclass
class TickReport:
    """What happened during one tick. The env turns this into rewards."""

    gathered: list[int]  # resource points deposited this tick, per player


class World:
    def __init__(self, cfg: SimConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or SimConfig()
        self.seed = seed
        self.tick = 0

        self.terrain, self.resources, start_tiles = mapgen.generate(self.cfg, seed)

        self.players: list[Player] = [Player(pid) for pid in range(self.cfg.num_players)]
        self.units: dict[int, Unit] = {}
        self.buildings: dict[int, Building] = {}
        self._next_id = 1

        # Cached passability. Invalidated whenever terrain or buildings change.
        self._blocked: np.ndarray | None = None

        # Every town centre goes down first, so worker placement below sees the
        # finished set of obstacles rather than a half-built map.
        for pid, (sx, sy) in enumerate(start_tiles):
            self.add_building(pid, sx, sy)

        for pid, (sx, sy) in enumerate(start_tiles):
            for i in range(self.cfg.start_workers):
                # Fan the starting workers out around the town centre, snapping
                # to open ground so a worker can never begin inside terrain.
                ox = (i % 2) * 3 - 1
                oy = (i // 2) * 3 - 1
                tile = self.nearest_free_tile(sx + ox, sy + oy)
                if tile is not None:
                    self.add_unit(pid, UnitKind.WORKER, *tile)

    # ------------------------------------------------------------------ setup

    @property
    def width(self) -> int:
        return self.cfg.map_width

    @property
    def height(self) -> int:
        return self.cfg.map_height

    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def add_unit(self, owner: int, kind: UnitKind, tx: int, ty: int) -> Unit:
        unit = Unit(uid=self._alloc_id(), owner=owner, kind=kind, x=tx + 0.5, y=ty + 0.5)
        self.units[unit.uid] = unit
        return unit

    def add_building(self, owner: int, tx: int, ty: int, w: int = 2, h: int = 2) -> Building:
        b = Building(bid=self._alloc_id(), owner=owner, x=tx, y=ty, w=w, h=h)
        self.buildings[b.bid] = b
        self._blocked = None
        return b

    # --------------------------------------------------------------- queries

    @property
    def blocked(self) -> np.ndarray:
        """``(H, W)`` bool grid of impassable tiles, rebuilt lazily."""
        if self._blocked is None:
            grid = np.zeros((self.height, self.width), dtype=bool)
            for t in Terrain:
                if not is_passable(t):
                    grid |= self.terrain == int(t)
            for b in self.buildings.values():
                grid[b.y : b.y + b.h, b.x : b.x + b.w] = True
            self._blocked = grid
        return self._blocked

    def units_of(self, owner: int) -> list[Unit]:
        return [u for u in self.units.values() if u.owner == owner]

    def unit_at(self, tile: Tile, owner: int | None = None) -> Unit | None:
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.tile == tile and (owner is None or u.owner == owner):
                return u
        return None

    def building_at(self, tile: Tile) -> Building | None:
        for b in self.buildings.values():
            if b.x <= tile[0] < b.x + b.w and b.y <= tile[1] < b.y + b.h:
                return b
        return None

    def in_bounds(self, tile: Tile) -> bool:
        return 0 <= tile[0] < self.width and 0 <= tile[1] < self.height

    def nearest_free_tile(self, tx: int, ty: int, max_radius: int = 12) -> Tile | None:
        """Closest passable tile to ``(tx, ty)``, searching outward in rings.

        Used for spawning. Anything that places an entity should go through
        this rather than trusting that a computed tile happens to be open.
        """
        if self.in_bounds((tx, ty)) and not self.blocked[ty, tx]:
            return tx, ty
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue  # ring edge only
                    nx, ny = tx + dx, ty + dy
                    if self.in_bounds((nx, ny)) and not self.blocked[ny, nx]:
                        return nx, ny
        return None

    # ------------------------------------------------------------------ step

    def step(self, commands: list[Command] | None = None) -> TickReport:
        """Advance the simulation by exactly one tick."""
        report = TickReport(gathered=[0] * len(self.players))

        for cmd in commands or ():
            self._apply_command(cmd)

        # Iterate in id order so the update does not depend on dict insertion
        # history -- one of the small things that keeps replays reproducible.
        for uid in sorted(self.units):
            self._update_unit(self.units[uid], report)

        self.tick += 1
        return report

    # -------------------------------------------------------------- commands

    def _owned(self, cmd: Command) -> list[Unit]:
        """Resolve a command's unit ids, silently dropping any not owned."""
        out = []
        for uid in cmd.unit_ids:
            u = self.units.get(uid)
            if u is not None and u.owner == cmd.owner:
                out.append(u)
        return out

    def _apply_command(self, cmd: Command) -> None:
        if isinstance(cmd, Stop):
            for u in self._owned(cmd):
                u.order = Order.IDLE
                u.path = []
                u.target = None
                u.gather_timer = 0
            return

        if not self.in_bounds(cmd.target):
            return

        if isinstance(cmd, Move):
            for u in self._owned(cmd):
                u.order = Order.MOVE
                u.target = cmd.target
                u.gather_timer = 0
                u.path = self._route(u.tile, cmd.target) or []
                if not u.path:
                    u.order = Order.IDLE

        elif isinstance(cmd, Gather):
            tx, ty = cmd.target
            if not is_harvestable(self.terrain[ty, tx]) or self.resources[ty, tx] <= 0:
                return
            kind = YIELDS[Terrain(self.terrain[ty, tx])]
            for u in self._owned(cmd):
                if u.kind is not UnitKind.WORKER:
                    continue
                # A worker already hauling something else drops it rather than
                # mixing loads -- the simplest rule that stays predictable.
                if u.carry_kind is not None and u.carry_kind is not kind:
                    u.carrying = 0
                    u.carry_kind = None
                u.order = Order.GATHER
                u.target = cmd.target
                u.gather_timer = 0
                u.path = []

    def _route(self, start: Tile, goal: Tile) -> list[Tile] | None:
        """Path to ``goal``, or beside it when ``goal`` itself is impassable."""
        if self.blocked[goal[1], goal[0]]:
            return path_to_adjacent(self.blocked, start, goal)
        return find_path(self.blocked, start, goal)

    # ----------------------------------------------------------- unit update

    def _update_unit(self, u: Unit, report: TickReport) -> None:
        if u.order is Order.IDLE:
            return
        if u.order is Order.MOVE:
            self._advance(u)
            if not u.path:
                u.order = Order.IDLE
        elif u.order is Order.GATHER:
            self._update_gather(u)
        elif u.order is Order.RETURN:
            self._update_return(u, report)

    def _update_gather(self, u: Unit) -> None:
        assert u.target is not None
        tx, ty = u.target

        if self.resources[ty, tx] <= 0:
            u.order = Order.RETURN if u.carrying else Order.IDLE
            u.path = []
            return

        if _chebyshev(u.tile, u.target) <= 1:
            u.path = []
            u.gather_timer += 1
            if u.gather_timer >= self.cfg.gather_ticks_per_unit:
                u.gather_timer = 0
                u.carry_kind = YIELDS[Terrain(self.terrain[ty, tx])]
                u.carrying += 1
                self.resources[ty, tx] -= 1
                if self.resources[ty, tx] <= 0:
                    # Exhausted: the tile becomes walkable ground.
                    self.terrain[ty, tx] = Terrain.GRASS
                    self._blocked = None
                if (
                    u.carrying >= self.cfg.worker_carry_capacity
                    or self.resources[ty, tx] <= 0
                ):
                    u.order = Order.RETURN
                    u.path = []
            return

        if not u.path:
            path = self._route(u.tile, u.target)
            if not path:
                u.order = Order.RETURN if u.carrying else Order.IDLE
                return
            u.path = path
        self._advance(u)

    def _update_return(self, u: Unit, report: TickReport) -> None:
        dropoff = self._nearest_dropoff(u)
        if dropoff is None:
            u.order = Order.IDLE
            return

        if min(_chebyshev(u.tile, t) for t in dropoff.tiles) <= 1:
            if u.carrying and u.carry_kind is not None:
                self.players[u.owner].add(u.carry_kind, u.carrying)
                report.gathered[u.owner] += u.carrying
            u.carrying = 0
            u.path = []
            # Head straight back if the tile still has anything left.
            if u.target is not None and self.resources[u.target[1], u.target[0]] > 0:
                u.order = Order.GATHER
            else:
                u.order = Order.IDLE
                u.carry_kind = None
            return

        if not u.path:
            path = path_to_adjacent(self.blocked, u.tile, (dropoff.x, dropoff.y))
            if not path:
                u.order = Order.IDLE
                return
            u.path = path
        self._advance(u)

    def _nearest_dropoff(self, u: Unit) -> Building | None:
        best, best_d = None, float("inf")
        for bid in sorted(self.buildings):
            b = self.buildings[bid]
            if b.owner != u.owner or not b.is_dropoff:
                continue
            cx, cy = b.centre
            d = (cx - u.x) ** 2 + (cy - u.y) ** 2
            if d < best_d:
                best, best_d = b, d
        return best

    def _advance(self, u: Unit) -> None:
        """Walk ``u`` along its path by one tick's worth of movement."""
        budget = self.cfg.worker_speed
        while budget > 0 and u.path:
            wx, wy = u.path[0]
            tx, ty = wx + 0.5, wy + 0.5
            dx, dy = tx - u.x, ty - u.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= budget:
                u.x, u.y = tx, ty
                budget -= dist
                u.path.pop(0)
            else:
                u.x += dx / dist * budget
                u.y += dy / dist * budget
                budget = 0.0

    # ------------------------------------------------------------ inspection

    def state_hash(self) -> str:
        """Stable digest of the full simulation state.

        Used by the determinism test, and handy for catching desyncs the moment
        you introduce something non-deterministic (a bare ``random``, a set
        iteration, a dict-ordering assumption).
        """
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(np.ascontiguousarray(self.terrain).tobytes())
        hasher.update(np.ascontiguousarray(self.resources).tobytes())
        hasher.update(str(self.tick).encode())
        for uid in sorted(self.units):
            u = self.units[uid]
            hasher.update(
                f"{u.uid},{u.owner},{int(u.kind)},{u.x:.6f},{u.y:.6f},"
                f"{int(u.order)},{u.carrying},{u.gather_timer},{u.target}".encode()
            )
        for bid in sorted(self.buildings):
            b = self.buildings[bid]
            hasher.update(f"{b.bid},{b.owner},{b.x},{b.y},{b.hp}".encode())
        for p in self.players:
            hasher.update(str(p.resources).encode())
        return hasher.hexdigest()


def _chebyshev(a: Tile, b: Tile) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
