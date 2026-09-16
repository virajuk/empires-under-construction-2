"""Headless throughput check and a random-agent smoke test.

Run this whenever you change the sim. Two numbers matter for RL:

* **raw ticks/sec** -- the ceiling on how fast any agent can experience the
  game.
* **env steps/sec** -- the same thing including observation encoding, which is
  usually where the time actually goes.

    python main.py --bench
    python -m scripts.bench --agent
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from empires.config import SimConfig
from empires.core.commands import Gather
from empires.core.terrain import Terrain
from empires.core.world import World


def run_bench(cfg: SimConfig | None = None, seed: int = 0, ticks: int = 20_000) -> float:
    cfg = cfg or SimConfig()
    world = World(cfg, seed=seed)

    # Put every villager to work first -- an idle sim is a meaningless benchmark.
    ys, xs = np.where(world.terrain == Terrain.FOREST)
    commands = []
    if len(xs):
        for pid in range(cfg.num_players):
            tc = next(b for b in world.buildings.values() if b.owner == pid)
            nearest = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
            commands.append(
                Gather(pid, tuple(u.uid for u in world.units_of(pid)),
                       (int(xs[nearest]), int(ys[nearest])))
            )

    world.step(commands)
    start = time.perf_counter()
    for i in range(ticks):
        # Re-task idle villagers periodically. A sim full of idle units is a
        # meaningless benchmark -- it measures the early-out, not the game.
        world.step(_retask(world) if i % 400 == 399 else None)
    elapsed = time.perf_counter() - start

    rate = ticks / elapsed
    sim_seconds = ticks / cfg.ticks_per_second
    print(f"map          {cfg.map_width}x{cfg.map_height}, {len(world.units)} units")
    print(f"ticks        {ticks:,} in {elapsed:.2f}s")
    print(f"throughput   {rate:,.0f} ticks/sec  ({rate / cfg.ticks_per_second:,.0f}x real time)")
    print(f"simulated    {sim_seconds / 60:.1f} minutes of game time")
    print(f"resources    {[p.resources for p in world.players]}")
    return rate


def _retask(world: World) -> list:
    ys, xs = np.where(world.resources > 0)
    if not len(xs):
        return []
    out = []
    for pid in range(len(world.players)):
        idle = [u.uid for u in world.units_of(pid) if int(u.order) == 0]
        if not idle:
            continue
        tc = next((b for b in world.buildings.values() if b.owner == pid), None)
        if tc is None:
            continue
        nearest = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
        out.append(Gather(pid, tuple(sorted(idle)), (int(xs[nearest]), int(ys[nearest]))))
    return out


def run_agent(episodes: int = 2, seed: int = 0) -> None:
    """Random policy through the full env, to time observation encoding too."""
    from empires.env.empires_env import EmpiresEnv

    env = EmpiresEnv(SimConfig(max_ticks=2000), seed=seed)
    rng = np.random.default_rng(seed)
    w, h = env.sim_cfg.map_width, env.sim_cfg.map_height

    total_steps = 0
    start = time.perf_counter()
    for ep in range(episodes):
        _, reset_info = env.reset()
        total_reward = 0.0
        while True:
            action = (
                rng.integers(0, env.max_units + 1),
                rng.integers(0, w),
                rng.integers(0, h),
            )
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if terminated or truncated:
                break
        print(f"episode {ep}  seed={reset_info['world_seed']}  "
              f"return={total_reward:7.2f}  resources={info['resources']}")
    elapsed = time.perf_counter() - start
    print(f"\n{total_steps / elapsed:,.0f} env steps/sec "
          f"({total_steps / elapsed * env.ticks_per_action:,.0f} ticks/sec)")
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="store_true", help="run a random agent through the env")
    ap.add_argument("--ticks", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.agent:
        run_agent(seed=args.seed)
    else:
        run_bench(seed=args.seed, ticks=args.ticks)
