import numpy as np
import pytest

from empires.config import SimConfig
from empires.env import observations as obs
from empires.env.empires_env import EmpiresEnv

CFG = SimConfig(max_ticks=200)


@pytest.fixture
def env():
    e = EmpiresEnv(CFG, seed=4, ticks_per_action=4)
    yield e
    e.close()


def test_reset_returns_well_formed_observation(env):
    o, info = env.reset()
    assert set(o) == {"grid", "scalars"}
    assert o["grid"].shape == (obs.NUM_CHANNELS, CFG.map_height, CFG.map_width)
    assert o["grid"].dtype == np.float32
    assert o["scalars"].shape == (obs.NUM_SCALARS,)
    assert "world_seed" in info


def test_observations_stay_in_range(env):
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(25):
        action = (rng.integers(0, env.max_units + 1),
                  rng.integers(0, CFG.map_width),
                  rng.integers(0, CFG.map_height))
        o, _, _, _, _ = env.step(action)
        assert o["grid"].min() >= 0.0 and o["grid"].max() <= 1.0
        assert o["scalars"].min() >= 0.0 and o["scalars"].max() <= 1.0


def test_step_returns_the_five_tuple(env):
    env.reset()
    o, r, terminated, truncated, info = env.step((0, 0, 0))
    assert isinstance(r, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["tick"] == env.ticks_per_action


def test_frame_skip_advances_that_many_ticks():
    e = EmpiresEnv(CFG, seed=1, ticks_per_action=7)
    e.reset()
    e.step((0, 0, 0))
    assert e.world.tick == 7
    e.close()


def test_episode_truncates_at_max_ticks(env):
    env.reset()
    truncated = False
    for _ in range(CFG.max_ticks):
        _, _, terminated, truncated, _ = env.step((0, 0, 0))
        if terminated or truncated:
            break
    assert truncated
    assert env.world.tick >= CFG.max_ticks


def test_same_seed_gives_the_same_episode():
    a, b = EmpiresEnv(CFG, seed=9), EmpiresEnv(CFG, seed=9)
    a.reset()
    b.reset()
    for i in range(20):
        action = (1, 5 + i % 3, 6)
        a.step(action)
        b.step(action)
    assert a.world.state_hash() == b.world.state_hash()
    a.close()
    b.close()


def test_slot_zero_is_a_noop(env):
    env.reset()
    before = [u.order for u in env.world.units_of(0)]
    env.step((0, 10, 10))
    assert [u.order for u in env.world.units_of(0)] == before


def test_out_of_range_slot_is_a_noop(env):
    env.reset()
    n = len(env.controllable_units())
    _, r, _, _, _ = env.step((n + 5, 10, 10))
    assert r == 0.0


def test_gathering_earns_reward():
    """A scripted 'send everyone to the nearest tree' policy must score."""
    from empires.core.terrain import Terrain

    e = EmpiresEnv(SimConfig(max_ticks=4000), seed=3, ticks_per_action=4)
    e.reset()
    ys, xs = np.where(e.world.terrain == Terrain.FOREST)
    tc = next(b for b in e.world.buildings.values() if b.owner == 0)
    i = int(np.argmin((xs - tc.x) ** 2 + (ys - tc.y) ** 2))
    target = (int(xs[i]), int(ys[i]))

    total = 0.0
    for slot in range(1, len(e.controllable_units()) + 1):
        _, r, _, _, _ = e.step((slot, *target))
        total += r
    for _ in range(300):
        _, r, terminated, truncated, info = e.step((0, 0, 0))
        total += r
        if terminated or truncated:
            break
    assert total > 0, "a worker sent to a tree should deposit something"
    assert sum(info["resources"]) > 0
    e.close()


def test_render_rgb_array_has_the_right_shape():
    e = EmpiresEnv(CFG, seed=0, render_mode="rgb_array")
    e.reset()
    frame = e.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    e.close()
