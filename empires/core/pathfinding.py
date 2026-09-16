"""Grid A*.

Pure Python + numpy, no game imports, so it is trivially unit-testable.

Design notes
------------
* 8-directional with octile heuristic, and diagonal moves are forbidden when
  they would cut a blocked corner -- otherwise units slide through wall seams.
* Units do **not** block each other. Only terrain and buildings do. Mutual
  blocking in an RTS needs flow fields plus a push/shove system; until you
  build that, letting units overlap is far better than deadlocking them.
* ``find_path`` returns waypoints *excluding* the start tile. An empty list
  means "already there"; ``None`` means "unreachable".
"""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

Tile = tuple[int, int]

_SQRT2 = 1.4142135623730951

# (dx, dy, cost)
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
)


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)


def find_path(
    blocked: np.ndarray,
    start: Tile,
    goal: Tile,
    max_expansions: int = 20_000,
) -> list[Tile] | None:
    """A* from ``start`` to ``goal`` over a ``(H, W)`` boolean blocked grid.

    ``max_expansions`` bounds the worst case so a unit ordered into an
    unreachable pocket cannot stall the whole tick.
    """
    h, w = blocked.shape
    sx, sy = start
    gx, gy = goal

    if not (0 <= gx < w and 0 <= gy < h) or blocked[gy, gx]:
        return None
    if start == goal:
        return []

    open_heap: list[tuple[float, float, Tile]] = [(_octile(sx, sy, gx, gy), 0.0, start)]
    came_from: dict[Tile, Tile] = {}
    g_score: dict[Tile, float] = {start: 0.0}
    closed: set[Tile] = set()
    expansions = 0

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)

        closed.add(current)
        expansions += 1
        if expansions > max_expansions:
            return None

        cx, cy = current
        for dx, dy, cost in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or blocked[ny, nx]:
                continue
            # No cutting corners diagonally past a blocked tile.
            if dx and dy and (blocked[cy, nx] or blocked[ny, cx]):
                continue

            neighbour = (nx, ny)
            if neighbour in closed:
                continue
            tentative = g + cost
            if tentative < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                heapq.heappush(
                    open_heap,
                    (tentative + _octile(nx, ny, gx, gy), tentative, neighbour),
                )

    return None


def _reconstruct(came_from: dict[Tile, Tile], current: Tile) -> list[Tile]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path[1:]  # drop the start tile


def adjacent_tiles(tile: Tile, w: int, h: int) -> Iterable[Tile]:
    """The 8 in-bounds neighbours of ``tile``."""
    x, y = tile
    for dx, dy, _ in _NEIGHBOURS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def path_to_adjacent(
    blocked: np.ndarray,
    start: Tile,
    goal: Tile,
) -> list[Tile] | None:
    """Path to the cheapest free tile *next to* ``goal``.

    This is how a worker approaches a tree or a gold vein: the target itself is
    impassable, so we route to the best neighbour of it.
    """
    h, w = blocked.shape
    candidates = [t for t in adjacent_tiles(goal, w, h) if not blocked[t[1], t[0]]]
    if not candidates:
        return None
    # Cheap ordering heuristic: try the closest neighbour first, and stop as
    # soon as one of them is actually reachable.
    candidates.sort(key=lambda t: _octile(start[0], start[1], t[0], t[1]))
    for cand in candidates:
        if cand == start:
            return []
        path = find_path(blocked, start, cand)
        if path is not None:
            return path
    return None
