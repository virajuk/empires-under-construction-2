"""Camera: the only place that knows how tiles map to pixels.

Keeping every world<->screen conversion here means the rest of the render code
never does arithmetic on ``tile_size``, and the simulation never sees a pixel.

Projection is isometric (2:1, Age of Empires style): tile ``(x, y)`` maps to
screen position ``((x - y) * tile_w/2, (x + y) * tile_h/2)`` before the camera
offset and an x-shift are applied. Increasing ``x`` moves right and down;
increasing ``y`` moves left and down. The whole tile grid projects onto a
diamond, so ``world_px_w``/``world_px_h`` are that diamond's bounding box, not
a simple ``dimension * tile_size`` like the orthogonal version this replaced.
"""

from __future__ import annotations

import math


class Camera:
    def __init__(self, world_w: int, world_h: int, tile_size: int,
                 view_w: int, view_h: int) -> None:
        self.tile_size = tile_size
        # 2:1 diamond: twice as wide as it is tall, which is what makes it
        # read as ground viewed from an angle rather than a spun square.
        self.tile_w = tile_size * 2
        self.tile_h = tile_size
        self.world_w = world_w
        self.world_h = world_h

        # Bounding box of the projected diamond (see module docstring for the
        # projection). Clamping and culling both measure against this rather
        # than a plain ``dimension * tile_size``.
        self.world_px_w = (world_w + world_h) * self.tile_w // 2
        self.world_px_h = (world_w + world_h) * self.tile_h // 2
        # The projection's raw x can go negative (a tile at x=0, high y sits
        # left of the origin) -- this shifts the whole diamond so its
        # leftmost point lands at pixel 0.
        self._origin_x = world_h * self.tile_w // 2

        self.view_w = view_w
        self.view_h = view_h
        self.x = 0.0  # top-left of the viewport, in projected world pixels
        self.y = 0.0

    def resize(self, view_w: int, view_h: int) -> None:
        self.view_w, self.view_h = view_w, view_h
        self.clamp()

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy
        self.clamp()

    def center_on_tile(self, tx: float, ty: float) -> None:
        px, py = self._project(tx, ty)
        self.x = px - self.view_w / 2
        self.y = py - self.view_h / 2
        self.clamp()

    def clamp(self) -> None:
        # When the map is smaller than the viewport, pin it at 0 rather than
        # letting the clamp invert and drift.
        self.x = max(0.0, min(self.x, max(0.0, self.world_px_w - self.view_w)))
        self.y = max(0.0, min(self.y, max(0.0, self.world_px_h - self.view_h)))

    # ------------------------------------------------------- conversions

    def _project(self, wx: float, wy: float) -> tuple[float, float]:
        """Tile-space -> pixel position, before the camera offset."""
        half_w = self.tile_w / 2
        half_h = self.tile_h / 2
        return (wx - wy) * half_w + self._origin_x, (wx + wy) * half_h

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Tile-space coordinates -> pixel coordinates in the viewport."""
        px, py = self._project(wx, wy)
        return int(px - self.x), int(py - self.y)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        px = sx + self.x - self._origin_x
        py = sy + self.y
        half_w = self.tile_w / 2
        half_h = self.tile_h / 2
        u = px / half_w  # == wx - wy
        v = py / half_h  # == wx + wy
        return (u + v) / 2, (v - u) / 2

    def screen_to_tile(self, sx: float, sy: float) -> tuple[int, int]:
        wx, wy = self.screen_to_world(sx, sy)
        return math.floor(wx), math.floor(wy)

    @property
    def visible_tiles(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` tile range to draw, end-exclusive.

        Culling to this is what keeps the frame cost independent of map size.
        The viewport is a screen-space rectangle, but under an isometric
        projection that back-projects to a rotated quadrilateral in tile
        space -- so unlike the orthogonal camera this used to be, the range
        comes from the *bounding box* of its four corners, not one division.
        The projection is affine, so the corners are exactly where each axis
        is extremal; a straight min/max over them is already a tight bound,
        not an approximation.
        """
        corners = ((0, 0), (self.view_w, 0), (0, self.view_h), (self.view_w, self.view_h))
        xs = []
        ys = []
        for sx, sy in corners:
            wx, wy = self.screen_to_world(sx, sy)
            xs.append(wx)
            ys.append(wy)
        margin = 1
        x0 = math.floor(min(xs)) - margin
        y0 = math.floor(min(ys)) - margin
        x1 = math.ceil(max(xs)) + margin
        y1 = math.ceil(max(ys)) + margin
        return x0, y0, x1, y1
