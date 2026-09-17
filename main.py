"""Entry point.

Thin on purpose: parse arguments, build config, hand off. All of the game is
in the ``empires`` package, so nothing here is on the import path of a training
run.

    python main.py                  # play
    python main.py --seed 7         # a specific map
    python main.py --bench          # headless throughput check, no window
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="empires", description="Tile-based RTS.")
    p.add_argument("--seed", type=int, default=0, help="map seed (default: 0)")
    p.add_argument("--width", type=int, default=None, help="map width in tiles")
    p.add_argument("--height", type=int, default=None, help="map height in tiles")
    p.add_argument("--players", type=int, default=None, help="number of players")
    p.add_argument("--tile-size", type=int, default=None, help="pixels per tile")
    p.add_argument("--fps", type=int, default=None, help="render frame cap")
    p.add_argument("--sprites", action=argparse.BooleanOptionalAction, default=None,
                   help="draw the sprite art instead of flat colour tiles")
    p.add_argument("--tps", type=int, default=None, help="simulation ticks per second")
    p.add_argument("--bench", action="store_true",
                   help="run the sim headless and report ticks/sec, then exit")
    p.add_argument("--bench-ticks", type=int, default=20_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from dataclasses import replace

    from empires.config import RenderConfig, SimConfig

    sim_cfg = SimConfig()
    overrides = {
        "map_width": args.width,
        "map_height": args.height,
        "num_players": args.players,
        "ticks_per_second": args.tps,
    }
    sim_cfg = replace(sim_cfg, **{k: v for k, v in overrides.items() if v is not None})

    if args.bench:
        from scripts.bench import run_bench

        run_bench(sim_cfg, seed=args.seed, ticks=args.bench_ticks)
        return 0

    render_cfg = RenderConfig()
    render_overrides = {
        "tile_size": args.tile_size,
        "fps": args.fps,
        "use_sprites": args.sprites,
    }
    render_cfg = replace(
        render_cfg, **{k: v for k, v in render_overrides.items() if v is not None}
    )

    from empires.app import App

    App(sim_cfg, render_cfg, seed=args.seed).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
