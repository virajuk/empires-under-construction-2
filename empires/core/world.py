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
from .commands import CancelTrain, Command, Gather, Move, Stop, Train
from .entities import (
    BUILDING_SPECS,
    MAX_QUEUE,
    UNIT_SPECS,
    Building,
    BuildingKind,
    Order,
    Player,
    Unit,
    UnitKind,
)
from .pathfinding import find_path, path_to_adjacent
from .terrain import (
    TERRAIN_FOR_RESOURCE,
    YIELDS,
    Resource,
    Terrain,
    is_harvestable,
    is_passable,
)

Tile = tuple[int, int]

# How many candidate tiles a villager will path-test when looking for more
# of its resource. Sorted nearest-first, so the first is almost always the
# answer; the cap stops a walled-off patch turning into a pile of A* runs.
RETARGET_CANDIDATES = 8


@dataclass
class TickReport:
    """What happened during one tick. The env turns this into rewards."""

    gathered: list[int]  # resource points deposited this tick, per player
    trained: list[int]   # units that finished training this tick, per player


@dataclass(frozen=True)
class VillagerActivity:
    """What a player's Villagers are doing right now.

    Two different questions get asked about Villagers, and conflating them is
    how this went wrong twice:

    * **What is it physically doing?** ``harvesting``, ``walking`` and ``idle``
      partition the whole population -- exactly one applies to each Villager,
      and they always sum to ``total``. These are what the HUD shows, because
      they are what you can see happening on screen.
    * **What is it assigned to?** ``gathering`` counts Villagers tasked to a
      resource whether they are walking to it, harvesting it, or hauling a load
      home. It deliberately *overlaps* the three above and is not part of the
      partition.

    Classifying by order alone put every Villager walking to a bush under
    "gathering", so the HUD's movement count sat at zero while four of them
    were plainly crossing the map.

    ``walking`` means "has somewhere left to walk", which is a close proxy for
    motion but not a frame-exact trace of it. At a state change it can differ by
    one tick in either direction:

    * routing is lazy, so a Villager that has just filled its load holds no path
      until the next tick computes one -- for that tick it reads as
      ``harvesting``, which is what it looks like, standing at the bush;
    * a Villager arriving within reach of its target stops with leftover
      waypoints still queued, so for that tick it reads as ``walking`` while
      already harvesting.

    At 20 ticks per second either is 50ms and invisible. It is spelled out only
    so the numbers are not mistaken for an exact motion trace.
    """

    total: int
    harvesting: int  # arrived, working a resource or unloading in place
    walking: int     # in transit -- to a resource, home with a load, or a move order
    idle: int
    gathering: int   # assigned to a resource; overlaps harvesting and walking
    carrying: int    # resource points currently being hauled


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

        # Every Town Center goes down first, so villager placement below sees
        # the finished set of obstacles rather than a half-built map.
        for pid, (sx, sy) in enumerate(start_tiles):
            self.add_building(pid, BuildingKind.TOWN_CENTER, sx, sy)

        for pid, (sx, sy) in enumerate(start_tiles):
            for i in range(self.cfg.start_villagers):
                # Fan the starting villagers out around the Town Center,
                # snapping to open ground so none can begin inside terrain.
                ox = (i % 2) * 3 - 1
                oy = (i // 2) * 3 - 1
                tile = self.nearest_free_tile(sx + ox, sy + oy)
                if tile is not None:
                    self.add_unit(pid, UnitKind.VILLAGER, *tile)

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
        unit = Unit(
            uid=self._alloc_id(), owner=owner, kind=kind,
            x=tx + 0.5, y=ty + 0.5, hp=UNIT_SPECS[kind].hp,
        )
        self.units[unit.uid] = unit
        return unit

    def add_building(self, owner: int, kind: BuildingKind, tx: int, ty: int) -> Building:
        """Place a building. Footprint and hit points come from its spec, so no
        caller has to remember how big a Town Center is."""
        spec = BUILDING_SPECS[kind]
        b = Building(
            bid=self._alloc_id(), owner=owner, kind=kind, x=tx, y=ty,
            w=spec.width, h=spec.height, hp=spec.hp, is_dropoff=spec.is_dropoff,
        )
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

    def villager_activity(self, owner: int) -> VillagerActivity:
        """Economy summary for ``owner``'s Villagers.

        The single source of truth for "how many are working": the HUD and the
        observation encoder both read it, so they cannot drift apart. Only units
        that can gather are counted -- a Soldier standing still is not an idle
        Villager.
        """
        total = harvesting = walking = idle = gathering = carrying = 0
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.owner != owner or not u.spec.can_gather:
                continue
            total += 1
            carrying += u.carrying
            if u.order in (Order.GATHER, Order.RETURN):
                gathering += 1

            # Physical state, keyed off the path rather than the order: a
            # Villager with somewhere left to walk is walking, whatever the
            # reason. Branching this way is exhaustive by construction, so a
            # new Order cannot quietly fall through all three buckets.
            if u.path:
                walking += 1
            elif u.order is Order.IDLE:
                idle += 1
            else:
                # Arrived and working in place: harvesting a tile, or standing
                # at the drop-off about to unload.
                harvesting += 1
        return VillagerActivity(total, harvesting, walking, idle, gathering, carrying)

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
        n = len(self.players)
        report = TickReport(gathered=[0] * n, trained=[0] * n)

        for cmd in commands or ():
            self._apply_command(cmd)

        # Iterate in id order so the update does not depend on dict insertion
        # history -- one of the small things that keeps replays reproducible.
        for bid in sorted(self.buildings):
            self._update_building(self.buildings[bid], report)
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

    def _owned_building(self, owner: int, bid: int) -> Building | None:
        b = self.buildings.get(bid)
        return b if b is not None and b.owner == owner else None

    def _apply_command(self, cmd: Command) -> None:
        if isinstance(cmd, Stop):
            for u in self._owned(cmd):
                u.order = Order.IDLE
                u.path = []
                u.target = None
                u.gather_timer = 0
                u.gather_resource = None
            return

        if isinstance(cmd, Train):
            self._apply_train(cmd)
            return

        if isinstance(cmd, CancelTrain):
            b = self._owned_building(cmd.owner, cmd.building_id)
            if b is None or not b.queue:
                return
            # Cancel from the back, so the one already part-built keeps its
            # progress. Refund in full -- partial refunds punish a misclick
            # twice, once in resources and once in the time already spent.
            self.players[b.owner].refund(UNIT_SPECS[b.queue.pop()].cost)
            if not b.queue:
                b.train_timer = 0
            return

        if not self.in_bounds(cmd.target):
            return

        if isinstance(cmd, Move):
            for u in self._owned(cmd):
                u.order = Order.MOVE
                u.target = cmd.target
                u.gather_timer = 0
                u.gather_resource = None
                u.path = self._route(u.tile, cmd.target) or []
                if not u.path:
                    u.order = Order.IDLE

        elif isinstance(cmd, Gather):
            tx, ty = cmd.target
            if not is_harvestable(self.terrain[ty, tx]) or self.resources[ty, tx] <= 0:
                return
            kind = YIELDS[Terrain(self.terrain[ty, tx])]
            for u in self._owned(cmd):
                if not u.spec.can_gather:
                    continue
                # A villager already hauling something else drops it rather than
                # mixing loads -- the simplest rule that stays predictable.
                if u.carry_kind is not None and u.carry_kind is not kind:
                    u.carrying = 0
                    u.carry_kind = None
                u.order = Order.GATHER
                u.target = cmd.target
                u.gather_resource = kind
                u.gather_timer = 0
                u.path = []

    def _apply_train(self, cmd: Train) -> None:
        """Queue a unit, charging for it up front.

        Charging on queue rather than on completion is the AoE rule, and it is
        the one that behaves: the player sees the cost the moment they commit,
        and a long queue cannot be built for free and then paid for later at
        prices the stockpile can no longer cover.
        """
        b = self._owned_building(cmd.owner, cmd.building_id)
        if b is None:
            return
        try:
            kind = UnitKind(cmd.unit_kind)
        except ValueError:
            return
        if kind not in b.spec.trains:
            return
        if len(b.queue) >= MAX_QUEUE:
            return

        player = self.players[b.owner]
        cost = UNIT_SPECS[kind].cost
        if not player.can_afford(cost):
            return
        player.spend(cost)
        b.queue.append(kind)

    def can_train(self, owner: int, bid: int, kind: UnitKind) -> bool:
        """Whether a Train command would be accepted. The UI asks this to grey
        out a button, so it must stay in step with :meth:`_apply_train`."""
        b = self._owned_building(owner, bid)
        return (
            b is not None
            and kind in b.spec.trains
            and len(b.queue) < MAX_QUEUE
            and self.players[owner].can_afford(UNIT_SPECS[kind].cost)
        )

    def _update_building(self, b: Building, report: TickReport) -> None:
        if not b.queue:
            b.train_timer = 0
            return

        kind = b.queue[0]
        b.train_timer += 1
        if b.train_timer < UNIT_SPECS[kind].train_ticks:
            return

        tile = self.spawn_tile_for(b)
        if tile is None:
            # Ringed in by terrain or buildings. Hold the finished unit at the
            # door rather than dropping it: the timer stays spent, so it pops
            # out the moment a tile frees up.
            return
        self.add_unit(b.owner, kind, *tile)
        b.queue.pop(0)
        b.train_timer = 0
        report.trained[b.owner] += 1

    def spawn_tile_for(self, b: Building) -> Tile | None:
        """Where a unit produced at ``b`` appears: just below it, or the
        nearest open ground to that."""
        return self.nearest_free_tile(b.x, b.y + b.h)

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
            u.path = []
            if u.carrying:
                # Deliver what we have first; the next tile is chosen after
                # unloading, from wherever the drop-off leaves us.
                u.order = Order.RETURN
            elif not self._retarget_gatherer(u):
                self._stop_gathering(u)
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
                    u.carrying >= self.cfg.villager_carry_capacity
                    or self.resources[ty, tx] <= 0
                ):
                    u.order = Order.RETURN
                    u.path = []
            return

        if not u.path:
            path = self._route(u.tile, u.target)
            if not path:
                # Walled off. Another patch of the same resource may still be
                # reachable, so try that before giving up.
                if u.carrying:
                    u.order = Order.RETURN
                elif not self._retarget_gatherer(u):
                    self._stop_gathering(u)
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
            # Head straight back if the tile still has anything left,
            # otherwise move on to the nearest patch of the same resource.
            if u.target is not None and self.resources[u.target[1], u.target[0]] > 0:
                u.order = Order.GATHER
            elif not self._retarget_gatherer(u):
                self._stop_gathering(u)
            return

        if not u.path:
            path = path_to_adjacent(self.blocked, u.tile, (dropoff.x, dropoff.y))
            if not path:
                u.order = Order.IDLE
                return
            u.path = path
        self._advance(u)

    def nearest_resource_tile(self, origin: Tile, resource: Resource,
                              max_radius: int | None = None) -> Tile | None:
        """Closest tile still holding ``resource`` that ``origin`` can reach.

        Reachability is checked rather than assumed: the nearest bush as the
        crow flies may be across a lake, and returning it would send a villager
        idle the moment it failed to path. Candidates are ordered nearest
        first with ties broken by position, so the choice is identical on every
        run -- two villagers freed by the same bush pick the same next one.
        """
        terrains = TERRAIN_FOR_RESOURCE.get(Resource(resource), ())
        if not terrains:
            return None

        # Slice to the search box before scanning. A Chebyshev radius *is* a
        # square, so the crop applies the distance limit exactly -- no separate
        # filter needed -- and keeps the scan off the rest of the map.
        radius = self.cfg.regather_radius if max_radius is None else max_radius
        if radius:
            x0 = max(0, origin[0] - radius)
            y0 = max(0, origin[1] - radius)
            x1 = min(self.width, origin[0] + radius + 1)
            y1 = min(self.height, origin[1] + radius + 1)
        else:
            x0, y0, x1, y1 = 0, 0, self.width, self.height

        sub_terrain = self.terrain[y0:y1, x0:x1]
        mask = self.resources[y0:y1, x0:x1] > 0
        if len(terrains) == 1:
            # The common case by far, and much quicker than np.isin here.
            mask &= sub_terrain == terrains[0]
        else:
            mask &= np.isin(sub_terrain, terrains)

        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None
        xs = xs + x0
        ys = ys + y0

        # Chebyshev, to match how units actually move (8-directional).
        dist = np.maximum(np.abs(xs - origin[0]), np.abs(ys - origin[1]))

        for i in np.lexsort((xs, ys, dist))[:RETARGET_CANDIDATES]:
            tile = (int(xs[i]), int(ys[i]))
            if path_to_adjacent(self.blocked, origin, tile) is not None:
                return tile
        return None

    def _retarget_gatherer(self, u: Unit) -> bool:
        """Send ``u`` to the nearest remaining tile of whatever it was
        collecting. False if there is nothing left within reach."""
        if u.gather_resource is None:
            return False
        tile = self.nearest_resource_tile(u.tile, u.gather_resource)
        if tile is None:
            return False
        u.order = Order.GATHER
        u.target = tile
        u.path = []
        u.gather_timer = 0
        return True

    def _stop_gathering(self, u: Unit) -> None:
        u.order = Order.IDLE
        u.path = []
        u.gather_resource = None
        u.carry_kind = None

    def _nearest_dropoff(self, u: Unit) -> Building | None:
        best, best_d = None, float("inf")
        for bid in sorted(self.buildings):
            b = self.buildings[bid]
            if b.owner != u.owner or not b.is_dropoff:
                continue
            cx, cy = b.center
            d = (cx - u.x) ** 2 + (cy - u.y) ** 2
            if d < best_d:
                best, best_d = b, d
        return best

    def _advance(self, u: Unit) -> None:
        """Walk ``u`` along its path by one tick's worth of movement."""
        budget = self.cfg.villager_speed
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
                f"{int(u.order)},{u.carrying},{u.gather_timer},{u.target},"
                f"{u.gather_resource}".encode()
            )
        for bid in sorted(self.buildings):
            b = self.buildings[bid]
            hasher.update(
                f"{b.bid},{b.owner},{b.x},{b.y},{b.hp},"
                f"{[int(k) for k in b.queue]},{b.train_timer}".encode()
            )
        for p in self.players:
            hasher.update(str(p.resources).encode())
        return hasher.hexdigest()


def _chebyshev(a: Tile, b: Tile) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
