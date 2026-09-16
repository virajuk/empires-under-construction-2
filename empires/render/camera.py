"""Camera: the only place that knows how tiles map to pixels.

Keeping every world<->screen conversion here means the rest of the render code
never does arithmetic on ``tile_size``, and the simulation never sees a pixel.
"""

from __future__ import annotations


class Camera:
    def __init__(self, world_w: int, world_h: int, tile_size: int,
                 view_w: int, view_h: int) -> None:
        self.tile_size = tile_size
        self.world_px_w = world_w * tile_size
        self.world_px_h = world_h * tile_size
        self.view_w = view_w
        self.view_h = view_h
        self.x = 0.0  # top-left of the viewport, in world pixels
        self.y = 0.0

    def resize(self, view_w: int, view_h: int) -> None:
        self.view_w, self.view_h = view_w, view_h
        self.clamp()

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy
        self.clamp()

    def center_on_tile(self, tx: float, ty: float) -> None:
        self.x = tx * self.tile_size - self.view_w / 2
        self.y = ty * self.tile_size - self.view_h / 2
        self.clamp()

    def clamp(self) -> None:
        # When the map is smaller than the viewport, pin it at 0 rather than
        # letting the clamp invert and drift.
        self.x = max(0.0, min(self.x, max(0.0, self.world_px_w - self.view_w)))
        self.y = max(0.0, min(self.y, max(0.0, self.world_px_h - self.view_h)))

    # ------------------------------------------------------- conversions

    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Tile-space coordinates -> pixel coordinates in the viewport."""
        return int(wx * self.tile_size - self.x), int(wy * self.tile_size - self.y)

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx + self.x) / self.tile_size, (sy + self.y) / self.tile_size

    def screen_to_tile(self, sx: float, sy: float) -> tuple[int, int]:
        wx, wy = self.screen_to_world(sx, sy)
        return int(wx), int(wy)

    @property
    def visible_tiles(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` tile range to draw, end-exclusive.

        Culling to this is what keeps the frame cost independent of map size.
        """
        x0 = int(self.x // self.tile_size)
        y0 = int(self.y // self.tile_size)
        x1 = int((self.x + self.view_w) // self.tile_size) + 1
        y1 = int((self.y + self.view_h) // self.tile_size) + 1
        return x0, y0, x1, y1
