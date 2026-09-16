# Empires

A tile-based RTS, built so that the game you play and the environment you train
against are the same code.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python main.py                  # play
python main.py --seed 7         # a specific map
python main.py --bench          # headless throughput, no window
pytest                          # 91 tests
```

## The one design decision that matters

Everything hangs off a single boundary:

```
                    ┌──────────────────────────┐
   mouse/keys ─────▶│                          │
                    │   Command objects        │
   RL policy  ─────▶│                          │
                    └────────────┬─────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  World.step()            │   pure python + numpy
                    │  empires/core/           │   never imports pygame
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      empires/render/  (pixels)        empires/env/  (tensors)
```

`empires/core/` has no pygame import anywhere in it. That is what lets the same
simulation run at 60 FPS behind a window and at tens of thousands of ticks per
second inside a training loop. If you ever need a colour, a sprite or a screen coordinate
inside `core/`, something has gone wrong.

The second half of the rule: **nothing mutates the world directly.** Human
input and RL actions both produce `Command` objects (`Move`, `Gather`, `Stop`)
and hand them to `World.step`. The simulation cannot tell which one it is
talking to, so an agent plays the game through exactly the interface a person
does.

## Layout

| Path | What lives there |
| --- | --- |
| `empires/core/world.py` | The simulation. One `step()`, one tick. Start here. |
| `empires/core/commands.py` | The action vocabulary — the door into the sim. |
| `empires/core/pathfinding.py` | A*, 8-directional, no corner cutting. |
| `empires/core/mapgen.py` | Seeded terrain + resource generation. |
| `empires/core/terrain.py` | Terrain types and which resource each yields. |
| `empires/render/` | Camera, tile renderer, HUD, minimap. |
| `empires/env/` | Gymnasium-style env and the observation encoder. |
| `empires/app.py` | The playable game: input → commands → world → pixels. |
| `main.py` | Argument parsing. Thin, so it is never on a training import path. |

## Resources

| Source | Yields | Notes |
| --- | --- | --- |
| Berry bushes | Food | A tight cluster just outside each start, plus neutral clusters to expand towards. Small stock each, so they run dry first. |
| Forest | Wood | Large blobs scattered over the map. |
| Gold | Gold | One 4×4 patch per start. |
| Stone | Stone | One 3×3 patch per start. |

All resource tiles are impassable and harvested from an adjacent tile, as in
AoE. An exhausted tile reverts to grass and becomes walkable, which is why a
solid gold patch is mined from the outside in.

`Resource` values are indices into `Player.resources` and into the observation
scalar vector, so they must stay distinct and contiguous from 0 — a duplicated
value makes `IntEnum` silently alias two names and merge their stockpiles.

## Controls

| | |
| --- | --- |
| Left click / drag | Select a unit, or box-select. Hold **Shift** to add. |
| Right click | Smart order: harvest if the tile holds a resource, else move. |
| **WASD** / arrows | Pan |
| **Space** | Centre on your town centre |
| **E** | Select all idle workers |
| **X** | Stop |
| **P** / **F** | Pause / fast-forward ×8 |

## Determinism

`World` is deterministic given a seed and a command stream. `tests/test_world.py`
runs two worlds side by side for 400 ticks and asserts their `state_hash()`
matches. Keep that test passing — replays, debugging, and anything off-policy
all depend on it. The usual ways it breaks:

- iterating a `set` or an unsorted `dict` in `step()`
- a bare `random` / `np.random` call instead of the seeded generator
- wall-clock time leaking into the simulation

Update timing is fixed-tick and decoupled from the frame rate (`App.run` uses an
accumulator), so the game plays identically on a 60 Hz and a 144 Hz display.

## RL

```python
from empires.env.empires_env import EmpiresEnv

env = EmpiresEnv(ticks_per_action=4)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step((1, 12, 9))
```

Gymnasium is optional — with it installed you get real `spaces`; without it the
env is a plain object with the same five-tuple API.

**Observation** — a dict, the shape a conv trunk + MLP head expects:

- `grid`: `(13, H, W)` float32 — terrain one-hot (grass, water, forest, gold,
  stone, berry), resource stock, own/enemy units, own/enemy buildings, carrying,
  idle. Egocentric, so a policy trained as player 0 transfers to player 1
  unchanged and self-play needs one network. The width is derived from the
  `Terrain` enum, so adding a terrain type widens it automatically.
- `scalars`: `(6,)` — stockpile, episode progress, idle fraction.

**Action** — `MultiDiscrete([max_units + 1, W, H])`, i.e. `(unit_slot, x, y)`.
Slot 0 is a no-op; slots 1..N index your units by id. The order issued is the
same smart order a right-click produces.

**Reward** — resources deposited, scaled. This is a placeholder: it rewards a
worker economy and nothing else.

Measured on this machine, 64x48 map with a fully active economy: **~39k sim
ticks/sec** headless (~2,000x real time), and **~5.7k env steps/sec** including
observation encoding. Observation encoding, not the simulation, is the
bottleneck -- `encode_grid` loops over units in Python. Look there first if you
need more. Run `python -m scripts.bench --agent` to measure yours.

### Known next steps

- **Reward is sparse.** A random policy scores exactly 0: a worker only pays out
  when it completes walk -> harvest -> haul -> deposit, and random `(slot, x, y)`
  actions essentially never chain that. `tests/test_env.py::test_gathering_earns_reward`
  confirms the plumbing works when a *scripted* policy issues sensible orders, so
  this is an exploration problem, not a bug. Two cheap fixes before you reach for
  a better algorithm: pay out on each harvest tick rather than only on deposit,
  and add the auto-assign action below to shorten the credit-assignment path.

- **Action space.** One unit per step scales badly — with 50 workers the policy
  burns 50 decisions on one round of orders. Notes on the three usual upgrades
  (auto-assign, spatial action head, shared-weight per-unit policy) are at the
  bottom of `empires/env/empires_env.py`.
- **No win condition.** Reward is resource delta and nothing else, so nothing
  pushes the agent past economy.
- **Not implemented yet:** combat, unit production, building construction, fog
  of war, tech tree. Each is additive — a new `Command` type, a branch in
  `World.step`, an observation plane.
- **Units do not collide.** Deliberate: mutual blocking needs flow fields and a
  push/shove system, and until then overlapping units beat deadlocked ones.
