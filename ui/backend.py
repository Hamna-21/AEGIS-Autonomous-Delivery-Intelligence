"""Frontend adapters for the existing AEGIS backend."""
import csv
from pathlib import Path

import numpy as np

from common import load_instance
from part_b.search import (DeliveryProblem, a_star, bfs, depth_limited_search,
                           greedy_best_first_search, iterative_deepening_search,
                           uniform_cost_search)
from part_d.interface import default_grid
from world.grid import load_grid
from world.transitions import legal_neighbors

from .config import ROOT

INSTANCE = load_instance()
PLANNERS = {"BFS": bfs, "Uniform Cost Search": uniform_cost_search,
            "Depth-Limited Search": lambda p: depth_limited_search(p, 80),
            "Iterative Deepening Search": lambda p: iterative_deepening_search(p, 80),
            "Greedy Best-First": greedy_best_first_search, "A*": a_star}


def load_ui_grid():
    try:
        return load_grid(ROOT / "data" / f"grid_{INSTANCE['instance_id']}.npy"), "Part A generated NumPy grid"
    except FileNotFoundError:
        return default_grid(), "Deterministic fallback grid"


def make_edges(grid):
    edges = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] >= 0:
                for nxt in legal_neighbors(grid, (r, c)):
                    if (r + c) % 3 != 0 or nxt[0] >= r:
                        edges.add(((r, c), nxt))
    return edges


def make_problem(grid, start, stops, flood_enabled=False, flooded=(), one_way=False):
    return DeliveryProblem(grid, tuple(start), [tuple(x) for x in stops], INSTANCE["terrain_cost"],
                           allowed_edges=make_edges(grid) if one_way else None,
                           flooded_cells=flooded, flood_enabled=flood_enabled, flood_interval=5)


def run_search(problem, label):
    return PLANNERS[label](problem)


def plan(grid, start, stops, name, flood=False, flooded=(), one_way=False):
    return run_search(make_problem(grid, start, stops, flood, flooded, one_way), name)


def read_csv(path: Path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))
