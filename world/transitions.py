"""Movement rules for the grid world.

One-way streets are represented by directed edges `(from_cell, to_cell)`. A
missing edge set means ordinary four-neighbour movement. If an edge set is
provided, every move must be present in that set, including moves over terrain
that is not a road. This conservative rule keeps the supplied generator's
unspecified one-way details configurable instead of inventing them.
"""
from dataclasses import dataclass
import hashlib
from typing import Iterable
import random
import numpy as np

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


def in_bounds(grid, cell):
    row, col = cell
    return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]


def is_passable(grid, cell):
    return in_bounds(grid, cell) and int(grid[cell]) >= 0


def legal_neighbors(grid, cell, allowed_edges: set[Edge] | None = None):
    result = []
    for dr, dc in DELTAS:
        nxt = (cell[0] + dr, cell[1] + dc)
        if not is_passable(grid, nxt):
            continue
        if allowed_edges is not None and (tuple(cell), nxt) not in allowed_edges:
            continue
        result.append(nxt)
    return result


def is_legal_move(grid, source, target, allowed_edges=None):
    return tuple(target) in legal_neighbors(grid, tuple(source), allowed_edges)


def directed_edges_from_instance(grid, instance_id, start=None, stops=()):
    """Create deterministic directed edges and add simple reachability corridors."""
    edges = set()
    seed = str(instance_id).encode()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            source = (row, col)
            if not is_passable(grid, source):
                continue
            for target in legal_neighbors(grid, source):
                digest = hashlib.sha256(seed + f":{row},{col}:{target[0]},{target[1]}".encode()).digest()
                if digest[0] % 3 != 0:
                    edges.add((source, target))
    if start is not None:
        for stop in stops:
            frontier = [tuple(start)]
            parent = {tuple(start): None}
            while frontier:
                current = frontier.pop(0)
                if current == tuple(stop):
                    break
                for nxt in legal_neighbors(grid, current):
                    if nxt not in parent:
                        parent[nxt] = current
                        frontier.append(nxt)
            if tuple(stop) in parent:
                current = tuple(stop)
                while parent[current] is not None:
                    edges.add((parent[current], current))
                    current = parent[current]
    return edges


@dataclass(frozen=True)
class SlipResult:
    intended: Cell
    actual: Cell
    slipped: bool


def apply_slip(grid, source, intended, slip_probability, rng=None, allowed_edges=None):
    """Apply a stochastic movement slip without changing the original grid.

    A slip chooses uniformly from legal neighbours of the source. If no legal
    neighbour exists, the agent remains in place. Invalid intended moves are
    rejected and leave the agent in place rather than bypassing obstacles.
    """
    if not 0 <= slip_probability <= 1:
        raise ValueError("slip_probability must be between 0 and 1")
    source = tuple(source)
    intended = tuple(intended)
    if not is_legal_move(grid, source, intended, allowed_edges):
        return SlipResult(intended, source, False)
    rng = rng or random.Random()
    if rng.random() >= slip_probability:
        return SlipResult(intended, intended, False)
    choices = legal_neighbors(grid, source, allowed_edges)
    actual = rng.choice(choices) if choices else source
    return SlipResult(intended, actual, actual != intended)
