"""Configuration.

Split deliberately into two dataclasses:

- ``SimConfig`` is everything the headless simulation needs. The RL env only
  ever sees this. It is frozen and hashable so it can be logged as part of an
  experiment record.
- ``RenderConfig`` is pixels, frame rates and colours. The simulation must
  never import it.

If you ever find yourself wanting a render field inside ``SimConfig``, that is
a sign that presentation logic has leaked into the sim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # --- world shape ---
    map_width: int = 64
    map_height: int = 48
    num_players: int = 2

    # --- simulation rate ---
    # The sim advances in discrete ticks. Everything below is expressed in
    # ticks, never in seconds, so the sim is wall-clock independent.
    ticks_per_second: int = 20

    # --- units ---
    start_villagers: int = 4
    villager_speed: float = 0.12       # tiles per tick
    villager_carry_capacity: int = 10
    gather_ticks_per_unit: int = 6     # ticks to harvest 1 resource point
    # How far a villager will look for more of the same resource when its tile
    # runs out. Bounded so it moves to the next bush rather than silently
    # trekking across the map; past this it goes idle and waits for orders.
    regather_radius: int = 24

    # --- resources ---
    forest_amount: int = 100
    gold_amount: int = 400
    stone_amount: int = 300
    # Berries are the early food source: a small cluster of bushes near each
    # start, each bush holding relatively little, so food runs out first and
    # forces a move to something else later.
    berry_amount: int = 120
    berry_bushes_per_cluster: int = 6
    neutral_berry_clusters: int = 6

    # --- episode ---
    max_ticks: int = 6000


@dataclass(frozen=True)
class RenderConfig:
    tile_size: int = 22
    window_width: int = 1280
    window_height: int = 720
    panel_height: int = 92
    fps: int = 60
    camera_speed: float = 700.0      # pixels per second
    edge_scroll_margin: int = 0      # set >0 to enable edge scrolling

    # --- sprites ---
    # Set use_sprites False to fall back to flat colour tiles everywhere, which
    # is also what happens automatically when empires/graphics is missing.
    use_sprites: bool = True
    # Multiples of a tile, applied to the *visible* artwork: sprites are
    # trimmed of transparent margin before scaling, so these are directly
    # comparable to each other and stay meaningful when new art is dropped in.
    # Above 1.0 a sprite overlaps its neighbours, which is what makes a patch
    # of forest read as a canopy rather than a checkerboard.
    tree_scale: float = 1.6
    bush_scale: float = 1.4
    # Gold sits in solid patches, so it stays near 1.0 -- much above that and
    # neighbouring tiles overlap into an unreadable mound.
    gold_scale: float = 1.1
    stone_scale: float = 1.1
    building_scale: float = 2.0
