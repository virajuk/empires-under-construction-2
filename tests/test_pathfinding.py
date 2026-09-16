import numpy as np
import pytest

from empires.core.pathfinding import find_path, path_to_adjacent


def grid(rows: list[str]) -> np.ndarray:
    """Build a blocked grid from ASCII art; '#' is impassable."""
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


def test_straight_line():
    g = grid(["....."])
    path = find_path(g, (0, 0), (4, 0))
    assert path == [(1, 0), (2, 0), (3, 0), (4, 0)]


def test_start_equals_goal_is_empty_not_none():
    g = grid(["..."])
    assert find_path(g, (1, 0), (1, 0)) == []


def test_routes_around_a_wall():
    g = grid([
        ".....",
        ".###.",
        ".....",
    ])
    path = find_path(g, (0, 1), (4, 1))
    assert path is not None
    assert path[-1] == (4, 1)
    assert not any(g[y, x] for x, y in path)


def test_unreachable_returns_none():
    g = grid([
        "..#..",
        "..#..",
        "..#..",
    ])
    assert find_path(g, (0, 1), (4, 1)) is None


def test_blocked_goal_returns_none():
    g = grid(["..#.."])
    assert find_path(g, (0, 0), (2, 0)) is None


def test_goal_out_of_bounds_returns_none():
    g = grid(["....."])
    assert find_path(g, (0, 0), (9, 0)) is None


def test_does_not_cut_blocked_corners():
    # The only diagonal from (0,0) to (1,1) squeezes between two walls.
    g = grid([
        ".#",
        "#.",
    ])
    assert find_path(g, (0, 0), (1, 1)) is None


def test_path_to_adjacent_stops_beside_a_blocked_target():
    g = grid([
        ".....",
        "..#..",
        ".....",
    ])
    path = path_to_adjacent(g, (0, 1), (2, 1))
    assert path is not None
    last = path[-1] if path else (0, 1)
    assert max(abs(last[0] - 2), abs(last[1] - 1)) == 1


def test_path_to_adjacent_none_when_fully_walled():
    g = grid([
        "#####",
        "#####",
        "#####",
    ])
    assert path_to_adjacent(g, (0, 0), (2, 1)) is None


@pytest.mark.parametrize("seed", range(5))
def test_random_maze_paths_are_valid(seed):
    rng = np.random.default_rng(seed)
    g = rng.random((20, 20)) < 0.25
    g[0, 0] = g[19, 19] = False
    path = find_path(g, (0, 0), (19, 19))
    if path is None:
        return  # legitimately walled off
    prev = (0, 0)
    for step in path:
        assert max(abs(step[0] - prev[0]), abs(step[1] - prev[1])) == 1
        assert not g[step[1], step[0]]
        prev = step
    assert prev == (19, 19)
