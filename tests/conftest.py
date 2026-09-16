"""Shared test setup.

pygame needs a video driver the moment anything calls ``display.set_mode``.
Forcing the dummy driver here -- before pygame is imported anywhere -- means
``pytest`` works on a headless box and in CI without the caller having to
remember to export SDL_VIDEODRIVER.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
