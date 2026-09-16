"""Empires -- a tile-based RTS with a simulation you can train against.

Layout::

    empires/core/     the simulation. pure python + numpy, never imports pygame
    empires/render/   pygame drawing. reads the world, mutates nothing
    empires/env/      gymnasium-style wrapper around the same simulation
    empires/app.py    the playable game: input -> commands -> world -> pixels

The rule that holds the whole thing together: anything that wants to affect the
game emits a :mod:`empires.core.commands` object and hands it to
``World.step``. Human input and RL policies both go through that one door.
"""

from .config import RenderConfig, SimConfig
from .core.world import World

__all__ = ["World", "SimConfig", "RenderConfig"]
__version__ = "0.1.0"
