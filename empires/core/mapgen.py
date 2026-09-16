"""Seeded procedural map generation.

Everything derives from a single integer seed, so ``seed=7`` is the same map on
every machine and every run. That matters more than it looks: reproducible maps
are the difference between "my agent got worse" and "my agent got a different
map".
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .terrain import Terrain

Tile = tuple[int, int]

PLAZA_RADIUS = 4  # tiles of guaranteed-clear ground around each start


def _blob_mask(rng: np.random.Generator, w: int, h: int, fill: float, smooth: int) -> np.ndarray:
    """Cellular-automata blobs -- cheap, and it looks like terrain."""
    grid = rng.random((h, w)) < fill
    for _ in range(smooth):
        padded = np.pad(grid.astype(np.int8), 1, mode="edge")
        neighbours = sum(
            padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        )
        grid = neighbours >= 5
    return grid


def generate(cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray, list[Tile]]:
    """Return ``(terrain, resources, start_tiles)``.

    ``terrain`` is ``(H, W)`` uint8 of :class:`Terrain`, ``resources`` is
    ``(H, W)`` int32 of remaining harvestable amount, and ``start_tiles`` is one
    Town Center top-left tile per player.
    """
    width, height = cfg.map_width, cfg.map_height
    rng = np.random.default_rng(seed)
    terrain = np.full((height, width), Terrain.GRASS, dtype=np.uint8)
    resources = np.zeros((height, width), dtype=np.int32)

    water = _blob_mask(rng, width, height, fill=0.42, smooth=3)
    terrain[water] = Terrain.WATER

    forest = _blob_mask(rng, width, height, fill=0.45, smooth=3) & ~water
    terrain[forest] = Terrain.FOREST
    resources[forest] = cfg.forest_amount

    # Start positions, inset from the border so the plaza always fits.
    inset = PLAZA_RADIUS + 2
    corners: list[Tile] = [
        (inset, inset),
        (width - inset - 2, height - inset - 2),
        (width - inset - 2, inset),
        (inset, height - inset - 2),
    ]
    start_tiles = corners[: cfg.num_players]

    # Clear every plaza *before* placing any patch, and keep a mask of them.
    # Patches are then forbidden from overlapping any plaza -- including
    # another player's, which is the case that is easy to forget.
    plazas = np.zeros((height, width), dtype=bool)
    for (sx, sy) in start_tiles:
        x0, x1 = max(0, sx - PLAZA_RADIUS), min(width, sx + PLAZA_RADIUS + 1)
        y0, y1 = max(0, sy - PLAZA_RADIUS), min(height, sy + PLAZA_RADIUS + 1)
        terrain[y0:y1, x0:x1] = Terrain.GRASS
        resources[y0:y1, x0:x1] = 0
        plazas[y0:y1, x0:x1] = True

    for (sx, sy) in start_tiles:
        # Berries first, and nearest: food is the resource you need in the first
        # minute, so every player gets a cluster just outside their plaza.
        _place_cluster(rng, terrain, resources, plazas, sx, sy,
                       Terrain.BERRY, cfg.berry_amount,
                       count=cfg.berry_bushes_per_cluster,
                       min_dist=PLAZA_RADIUS + 2, max_dist=PLAZA_RADIUS + 5)
        _place_patch(rng, terrain, resources, plazas, sx, sy,
                     Terrain.GOLD, cfg.gold_amount, size=4)
        _place_patch(rng, terrain, resources, plazas, sx, sy,
                     Terrain.STONE, cfg.stone_amount, size=3)

    # Neutral berry clusters scattered over the whole map, so food is something
    # players have to expand towards once the home bushes are picked clean.
    for _ in range(cfg.neutral_berry_clusters):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        _place_cluster(rng, terrain, resources, plazas, cx, cy,
                       Terrain.BERRY, cfg.berry_amount,
                       count=cfg.berry_bushes_per_cluster,
                       min_dist=0, max_dist=2)

    # Runs last, once, over the finished map -- see _unseal for why per-cluster
    # repair is not enough.
    _unseal(terrain, resources, Terrain.BERRY)

    return terrain, resources, start_tiles


def _place_patch(rng: np.random.Generator, terrain: np.ndarray, resources: np.ndarray,
                 plazas: np.ndarray, cx: int, cy: int,
                 kind: Terrain, amount: int, size: int) -> None:
    """Drop a solid square resource patch a short walk from ``(cx, cy)``."""
    h, w = terrain.shape
    for _ in range(64):  # bounded retries; give up rather than loop forever
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(PLAZA_RADIUS + size + 1, PLAZA_RADIUS + size + 6)
        px = int(np.clip(cx + np.cos(angle) * dist, 1, w - size - 1))
        py = int(np.clip(cy + np.sin(angle) * dist, 1, h - size - 1))
        block = terrain[py : py + size, px : px + size]
        if np.all(block == Terrain.GRASS) and not plazas[py : py + size, px : px + size].any():
            terrain[py : py + size, px : px + size] = kind
            resources[py : py + size, px : px + size] = amount
            return


def _place_cluster(rng: np.random.Generator, terrain: np.ndarray, resources: np.ndarray,
                   plazas: np.ndarray, cx: int, cy: int,
                   kind: Terrain, amount: int, count: int,
                   min_dist: int, max_dist: int, spread: float = 1.6) -> int:
    """Place ``count`` tiles as one tight clump near ``(cx, cy)``.

    Two stages, and the order matters: first pick a single anchor somewhere in
    the ring ``[min_dist, max_dist]``, then scatter the tiles within ``spread``
    of *that anchor*. Choosing each tile independently from the ring instead
    smears them around its whole circumference, which gives isolated bushes
    rather than a patch you can put four villagers on.

    Unlike :func:`_place_patch` this needs no solid rectangle of free ground, so
    it still succeeds on a cluttered map -- which matters for berries, since a
    player with no food at all is a broken start rather than a hard one.
    Returns how many tiles it actually placed.
    """
    h, w = terrain.shape

    anchor: Tile | None = None
    for _ in range(64):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(min_dist, max_dist)
        ax = int(round(cx + np.cos(angle) * dist))
        ay = int(round(cy + np.sin(angle) * dist))
        if 0 <= ax < w and 0 <= ay < h and terrain[ay, ax] == Terrain.GRASS and not plazas[ay, ax]:
            anchor = (ax, ay)
            break
    if anchor is None:
        return 0

    placed: list[Tile] = []
    for _ in range(count * 40):
        if len(placed) >= count:
            break
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, spread)
        px = int(round(anchor[0] + np.cos(angle) * dist))
        py = int(round(anchor[1] + np.sin(angle) * dist))
        if not (0 <= px < w and 0 <= py < h):
            continue
        if terrain[py, px] != Terrain.GRASS or plazas[py, px]:
            continue
        if _open_neighbours(terrain, px, py) <= 2:
            continue
        terrain[py, px] = kind
        resources[py, px] = amount
        placed.append((px, py))

    return len(placed)


def _unseal(terrain: np.ndarray, resources: np.ndarray, kind: Terrain) -> int:
    """Clear any ``kind`` tile left with no open neighbour. Returns how many.

    This has to run once, globally, after *everything* is placed. Checking as
    each tile goes down is not enough, and neither is a per-cluster pass: a
    later cluster -- or a gold patch dropped next to an earlier one -- can wall
    in a tile whose own repair pass has already finished.

    A villager cannot stand next to a sealed bush, so leaving one in place gives
    the player a resource they can see, click, and never harvest. Reverting a
    tile frees space for its neighbours, so this repeats until it settles.

    Solid patches (:func:`_place_patch`) deliberately keep their buried
    interior -- that is the AoE behaviour, and those tiles become reachable as
    the outer ring is mined out.
    """
    removed = 0
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(terrain == kind)
        for x, y in zip(xs.tolist(), ys.tolist()):
            if _open_neighbours(terrain, x, y) == 0:
                terrain[y, x] = Terrain.GRASS
                resources[y, x] = 0
                removed += 1
                changed = True
    return removed


def _open_neighbours(terrain: np.ndarray, x: int, y: int) -> int:
    """Count walkable tiles around ``(x, y)``, excluding ``(x, y)`` itself.

    Called both before a tile is placed (center still grass) and after (center
    is a resource), so the center has to be discounted conditionally rather
    than with a blanket ``- 1`` -- otherwise a fully sealed tile scores ``-1``
    and slips past an ``== 0`` check.
    """
    h, w = terrain.shape
    x0, x1 = max(0, x - 1), min(w, x + 2)
    y0, y1 = max(0, y - 1), min(h, y + 2)
    count = int((terrain[y0:y1, x0:x1] == Terrain.GRASS).sum())
    if terrain[y, x] == Terrain.GRASS:
        count -= 1
    return count
