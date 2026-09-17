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
| `empires/core/entities.py` | Units, buildings, and the specs carrying their names. |
| `empires/render/` | Camera, tile renderer, HUD, minimap. |
| `empires/render/assets.py` | Sprite loading and per-tile variety. |
| `empires/graphics/` | The art: `home.png`, `tree/`, `berry_bushes/`. |
| `empires/env/` | Gymnasium-style env and the observation encoder. |
| `empires/app.py` | The playable game: input → commands → world → pixels. |
| `main.py` | Argument parsing. Thin, so it is never on a training import path. |

## Vocabulary

The game's domain language lives in [`empires/core/entities.py`](empires/core/entities.py).
Display names come from the spec tables rather than being typed out at each call
site, so the HUD, logs and any future tooltip all read the same word.

| Kind | Name | Notes |
| --- | --- | --- |
| `UnitKind.VILLAGER` | Villager | The resource gatherer. `can_gather=True`. |
| `UnitKind.SOLDIER` | Soldier | Defined but not yet implemented — cannot gather. |
| `BuildingKind.TOWN_CENTER` | Town Center | 2×2, accepts every resource. |

`UNIT_SPECS` and `BUILDING_SPECS` carry label, plural, hit points and footprint.
`World.add_building(owner, kind, x, y)` reads the size and hp from the spec, so
no caller has to remember how big a Town Center is. `unit_label(kind, n)` and
`building_label(kind, n)` handle pluralisation.

Adding a kind means adding an enum member and a spec entry — `test_entities.py`
asserts every kind has one, so a missing spec fails at test time rather than at
render time.

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

**Villagers keep working.** A `Gather` order assigns a *resource*, not just a
tile (`Unit.gather_resource`). When the bush runs dry the villager delivers
whatever it is carrying, then walks to the nearest remaining bush and carries
on — one order keeps a crew fed for thousands of ticks. It only stops when
nothing of that resource is left within `regather_radius` (24 tiles). The
assignment has to live on the unit because an exhausted tile has already
reverted to grass and can no longer say what it used to be.

`World.nearest_resource_tile()` orders candidates nearest-first and **checks
each is actually reachable** before returning it — the closest bush as the crow
flies may be across a lake, and returning it would send the villager idle the
moment it failed to path. Each villager searches from its own position, so a
crew freed by one bush spreads across neighbouring bushes rather than queueing
on a single tile. A `Move` or `Stop` order clears the assignment.

`Resource` values are indices into `Player.resources` and into the observation
scalar vector, so they must stay distinct and contiguous from 0 — a duplicated
value makes `IntEnum` silently alias two names and merge their stockpiles.

## Controls

| | |
| --- | --- |
| Left click | Select the unit under the cursor. Clicking the same spot again cycles through units stacked there. |
| Double click | Select every unit of that kind currently on screen. |
| Shift + click | Add to the selection, or remove a unit already in it. |
| Left drag | Box-select. Hold **Shift** to add the box to the selection. |
| Right click | Smart order: harvest if the tile holds a resource, else move. |
| Click a Town Center | Select it, and show its production panel. |
| **V** / Train button | Queue a Villager (50 Food). |
| **WASD** / arrows | Pan |
| **Space** | Center on your Town Center |
| **E** | Select all idle Villagers |
| **X** | Stop units, or cancel the last queued unit |
| **P** / **F** | Pause / fast-forward ×8 |

## Art

Sprites live in `empires/graphics/`:

| Path | Used for |
| --- | --- |
| `home.png` | Town Center |
| `tree/*.png` | Forest tiles |
| `berry_bushes/*.png` | Berry tiles |
| `gold/*.png` | Gold tiles |
| `stone/*.png` | Stone tiles |

Drop more PNGs into any of those directories and they join the rotation
automatically — the loader takes every PNG in the directory, sorted. Removing
one is how you retire it; nothing else needs editing.

Adding a decorated terrain is three lines: a directory in `DECOR_DIRS`
([`assets.py`](empires/render/assets.py)), a `*_scale` on `RenderConfig`, and
the line in `Renderer.__init__` that ties the two together. `DECOR_TERRAIN` and
the per-terrain sprite tables are derived from `DECOR_DIRS`, so they cannot
drift out of step. The art tests are parametrised over it, so a new resource
gets coverage for presence, scaling, trimming and ground colour for free.

Two things are less obvious than they look:

**Variety has to be stable.** The renderer redraws every frame, so picking a
tree at random per call would make the forest flicker. The variant is instead
derived from a hash of the tile coordinates and the map seed, so tile (12, 7)
grows the same tree on every frame and every machine, with no extra state to
keep in sync when terrain changes. `test_assets.py` asserts consecutive frames
are pixel-identical.

**Decor is a second draw pass.** Trees are drawn taller than their tile and
anchored to its bottom edge so a patch of forest overlaps into a canopy. Drawn
inline with the terrain loop, the next row of ground would paint over each
trunk — so all ground goes down first, then decor, top row to bottom, giving
nearer trees the overlap.

Missing art is never fatal: every sprite lookup can return `None` and each has a
flat-colour fallback, so a fresh clone without `graphics/` still runs. Tune or
disable in `RenderConfig`:

```python
use_sprites = True     # False falls back to flat colour tiles
tree_scale = 1.6       # multiples of a tile; >1 overlaps neighbours
bush_scale = 1.4
gold_scale = 1.1       # solid patches, so these stay near 1.0
stone_scale = 1.1
building_scale = 2.0
```

Water is the only terrain still drawn as flat colour.

Source PNGs are normalised on load: palettised 8-bit images are widened to
32-bit, because `smoothscale` rejects anything narrower and `convert_alpha`
cannot run without a display (which is the case for the RL env's `rgb_array`
render).

## Production

Select your Town Center and press **V** (or the panel button) to queue a
Villager for **50 Food**. It takes 100 ticks — five seconds at the default 20
ticks per second — and appears on open ground just below the building.

Costs and training times live on the specs in
[`empires/core/entities.py`](empires/core/entities.py), next to hit points and
labels:

```python
UnitKind.VILLAGER: UnitSpec(..., cost=costs(food=50), train_ticks=100)
BuildingKind.TOWN_CENTER: BuildingSpec(..., trains=(UnitKind.VILLAGER,))
```

`BuildingSpec.trains` is the single source of truth for what a building can
make: the UI asks it what buttons to offer, and the simulation asks it whether
an order is legal.

Rules worth knowing:

- **Cost is charged when the unit joins the queue**, not when it pops out. The
  player sees the cost the moment they commit, and a long queue cannot be built
  for free and paid for later at prices the stockpile no longer covers.
- **Cancelling refunds in full** and takes from the *back* of the queue, so the
  unit already part-built keeps its progress.
- **Queues are capped** at `MAX_QUEUE` (10), so a misclick — or an agent
  spamming the action — cannot drain a stockpile into minutes of production.
- **A finished unit with nowhere to stand waits at the door** rather than being
  dropped; it appears the moment a tile frees up.
- `World.can_train()` is what greys out the button. It has a test asserting it
  agrees with what `step()` actually accepts, so the two cannot drift.

The RL action space does **not** yet include training — `(unit_slot, x, y)`
only issues move and gather orders. Adding it means extending the action space,
not the simulation; `TickReport.trained` is already there to reward from.

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
Villager economy and nothing else.

Measured on this machine, 64x48 map with a fully active economy: **~28k sim
ticks/sec** headless (~1,400x real time), and **~5.7k env steps/sec** including
observation encoding. (It was ~39k before villagers re-targeted automatically;
most of the drop is simply that they now work continuously instead of going
idle, with the resource search itself accounting for ~6%.) Observation encoding, not the simulation, is the
bottleneck -- `encode_grid` loops over units in Python. Look there first if you
need more. Run `python -m scripts.bench --agent` to measure yours.

### Known next steps

- **Reward is sparse.** A random policy scores exactly 0: a Villager only pays out
  when it completes walk -> harvest -> haul -> deposit, and random `(slot, x, y)`
  actions essentially never chain that. `tests/test_env.py::test_gathering_earns_reward`
  confirms the plumbing works when a *scripted* policy issues sensible orders, so
  this is an exploration problem, not a bug. Two cheap fixes before you reach for
  a better algorithm: pay out on each harvest tick rather than only on deposit,
  and add the auto-assign action below to shorten the credit-assignment path.

- **Action space.** One unit per step scales badly — with 50 Villagers the policy
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
