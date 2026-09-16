"""Sprite loading.

Two things here are less obvious than they look.

**Variety has to be stable.** The renderer redraws every frame, so picking a
tree at random each time would make the forest flicker. Instead the variant is
derived from a hash of the tile coordinates and the map seed -- so tile (12, 7)
is the same tree on every frame, every run, and on every machine, with no extra
state to keep in sync when terrain changes.

**Missing art must not be fatal.** Every loader returns ``None`` rather than
raising, and the renderer falls back to flat colour. That keeps the game
runnable on a fresh clone without assets, keeps headless RL rendering working,
and means a typo in a filename costs you a missing sprite, not a crash.
"""

from __future__ import annotations

from pathlib import Path

import pygame

GRAPHICS_DIR = Path(__file__).resolve().parent.parent / "graphics"

TREE_DIR = "tree"
BUSH_DIR = "berry_bushes"
TOWN_CENTER_FILE = "home.png"

# Large primes; the standard spatial hash. Any decent mix works -- what matters
# is that it depends only on position and seed, never on frame or wall clock.
_HASH_X = 73856093
_HASH_Y = 19349663
_HASH_SEED = 83492791


def variant_index(x: int, y: int, seed: int, count: int) -> int:
    """Pick one of ``count`` variants for tile ``(x, y)``, deterministically."""
    if count <= 0:
        return 0
    h = (x * _HASH_X) ^ (y * _HASH_Y) ^ (seed * _HASH_SEED)
    return (h & 0x7FFFFFFF) % count


def _convert(surface: pygame.Surface) -> pygame.Surface:
    """``convert_alpha`` needs a display; offscreen renders have none."""
    try:
        return surface.convert_alpha()
    except pygame.error:
        return surface


def _load(path: Path) -> pygame.Surface | None:
    if not path.is_file():
        return None
    try:
        return _convert(pygame.image.load(str(path)))
    except pygame.error:
        return None


def _load_dir(name: str) -> list[pygame.Surface]:
    """Every PNG in a graphics subdirectory, in sorted order.

    Sorted, because the variant index is a position in this list: if the order
    depended on the filesystem, the same seed would grow different forests on
    different machines.
    """
    directory = GRAPHICS_DIR / name
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.png")):
        surf = _load(path)
        if surf is not None:
            out.append(surf)
    return out


def _trim(surface: pygame.Surface) -> pygame.Surface:
    """Crop away transparent margin, so a scale factor means visible size.

    The source art is not padded consistently -- a tree fills its 64x64 canvas
    almost edge to edge while a bush occupies barely half of it. Scaling the
    raw canvas therefore made trees render more than twice the size of bushes
    while the config said 1.4x. Trimming first makes ``tree_scale`` and
    ``bush_scale`` directly comparable, and means dropping in new art with
    different padding does not silently change how big it looks.
    """
    try:
        rects = pygame.mask.from_surface(surface).get_bounding_rects()
    except pygame.error:
        return surface
    if not rects:
        return surface
    bbox = rects[0].unionall(rects[1:])
    return surface.subsurface(bbox).copy()


def _fit(surface: pygame.Surface, box: int) -> pygame.Surface:
    """Scale to fit a ``box`` x ``box`` square, keeping the aspect ratio.

    Trimmed sprites are no longer square, so scaling both axes to the box would
    squash a tall tree. Fitting the longer side keeps the silhouette.
    """
    w, h = surface.get_size()
    longest = max(w, h)
    if longest <= 0:
        return surface
    factor = box / longest
    return pygame.transform.smoothscale(
        surface, (max(1, round(w * factor)), max(1, round(h * factor)))
    )


class Assets:
    """Sprites scaled for one tile size. Cheap to build, so rebuild on zoom."""

    def __init__(self, tile_size: int, tree_scale: float = 1.45,
                 bush_scale: float = 1.15, enabled: bool = True) -> None:
        self.tile_size = tile_size
        self.trees: list[pygame.Surface] = []
        self.bushes: list[pygame.Surface] = []
        self._town_center_src: pygame.Surface | None = None
        self._building_cache: dict[tuple[int, int], pygame.Surface] = {}

        # ``enabled=False`` yields a fully valid Assets whose every lookup
        # returns None, so the renderer needs no second code path to run
        # without art.
        if not enabled:
            return

        tree_px = int(tile_size * tree_scale)
        bush_px = int(tile_size * bush_scale)
        self.trees = [_fit(_trim(s), tree_px) for s in _load_dir(TREE_DIR)]
        self.bushes = [_fit(_trim(s), bush_px) for s in _load_dir(BUSH_DIR)]

        src = _load(GRAPHICS_DIR / TOWN_CENTER_FILE)
        self._town_center_src = _trim(src) if src is not None else None

    @property
    def has_trees(self) -> bool:
        return bool(self.trees)

    @property
    def has_bushes(self) -> bool:
        return bool(self.bushes)

    def tree(self, x: int, y: int, seed: int) -> pygame.Surface | None:
        if not self.trees:
            return None
        return self.trees[variant_index(x, y, seed, len(self.trees))]

    def bush(self, x: int, y: int, seed: int) -> pygame.Surface | None:
        if not self.bushes:
            return None
        return self.bushes[variant_index(x, y, seed, len(self.bushes))]

    def town_center(self, width_px: int, height_px: int) -> pygame.Surface | None:
        """Town Center art fitted to a box, cached per size.

        Fitted rather than stretched: the trimmed source is wider than it is
        tall, so forcing both axes would squash the roof.
        """
        if self._town_center_src is None:
            return None
        key = (max(1, width_px), max(1, height_px))
        cached = self._building_cache.get(key)
        if cached is None:
            cached = _fit(self._town_center_src, min(key))
            self._building_cache[key] = cached
        return cached
