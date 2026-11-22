"""From-scratch multi-delivery search algorithms for AEGIS Part B.

The search state is (position, delivered stops, time, flooded cells).  Time and
flooded cells are deliberately part of the state because the legal terrain and
therefore the action cost can change during a route.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count, permutations
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from world.flood import advance_flood
from world.grid import TERRAIN_NAMES
from world.transitions import legal_neighbors
from world.fog import visible_mask

Cell = tuple[int, int]


@dataclass(frozen=True)
class SearchState:
    position: Cell
    delivered: frozenset[int]
    time: int
    flooded: frozenset[Cell]
    known_cells: frozenset[Cell] = frozenset()


@dataclass
class SearchResult:
    algorithm: str
    path: list[Cell]
    states: list[SearchState]
    cost: float
    nodes_expanded: int
    runtime_seconds: float
    max_frontier: int
    max_memory_states: int
    found: bool


class DeliveryProblem:
    def __init__(self, grid, start, delivery_stops, terrain_cost,
                 allowed_edges=None, flooded_cells=(), flood_interval=5,
                 flood_enabled=False, fog_enabled=False, fog_radius=4,
                 deadlines=None, late_penalty=10.0):
        self.grid = np.asarray(grid, dtype=int)
        self.start = tuple(start)
        self.stops = tuple(tuple(x) for x in delivery_stops)
        self.terrain_cost = dict(terrain_cost)
        self.allowed_edges = allowed_edges
        self.flood_interval = flood_interval
        self.flood_enabled = flood_enabled
        self.fog_enabled = bool(fog_enabled)
        self.fog_radius = int(fog_radius)
        self.deadlines = tuple(deadlines or ())
        self.late_penalty = float(late_penalty)
        self.initial_flooded = frozenset(tuple(x) for x in flooded_cells)
        if not self._passable(self.start):
            raise ValueError("start must be passable")

    def initial_state(self):
        delivered = frozenset(i for i, cell in enumerate(self.stops)
                              if cell == self.start)
        known = frozenset(map(tuple, np.argwhere(visible_mask(self.grid.shape, self.start, self.fog_radius)))) if self.fog_enabled else frozenset()
        return SearchState(self.start, delivered, 0, self.initial_flooded, known)

    def is_goal(self, state):
        return len(state.delivered) == len(self.stops)

    def _passable(self, cell):
        r, c = cell
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and self.grid[cell] >= 0

    def successors(self, state):
        for nxt in legal_neighbors(self.grid, state.position, self.allowed_edges):
            new_time = state.time + 1
            new_grid, new_flooded = advance_flood(
                self.grid, state.flooded, new_time, self.flood_interval,
                self.flood_enabled)
            if new_grid[nxt] < 0:
                continue
            terrain = TERRAIN_NAMES[int(new_grid[nxt])]
            delivered = set(state.delivered)
            delivery_cost = float(self.terrain_cost[terrain])
            was_known = (not self.fog_enabled) or nxt in state.known_cells
            for i, stop in enumerate(self.stops):
                if stop == nxt:
                    delivered.add(i)
                    if i < len(self.deadlines) and new_time > self.deadlines[i]:
                        delivery_cost += self.late_penalty
            known = state.known_cells
            if self.fog_enabled:
                known = known | frozenset(map(tuple, np.argwhere(visible_mask(self.grid.shape, nxt, self.fog_radius))))
                if not was_known:
                    delivery_cost = min(float(v) for v in self.terrain_cost.values())
            yield SearchState(nxt, frozenset(delivered), new_time,
                              frozenset(new_flooded), frozenset(known)), delivery_cost

    def heuristic(self, state, inadmissible=False):
        remaining = [self.stops[i] for i in range(len(self.stops))
                     if i not in state.delivered]
        if not remaining:
            return 0.0
        unit = min(float(v) for v in self.terrain_cost.values())
        # Minimum cost Hamiltonian completion in an obstacle-free Manhattan
        # relaxation.  It is a lower bound because every real move changes one
        # coordinate by one and all terrain costs are at least unit.
        best = min(sum(abs(a[0] - b[0]) + abs(a[1] - b[1])
                       for a, b in zip((state.position,) + order, order))
                   for order in permutations(remaining))
        value = float(best * unit)
        if inadmissible:
            # Deliberately overestimates by pricing every relaxed step as the
            # maximum terrain cost. Used only for the requested experiment.
            value = float(best * max(float(v) for v in self.terrain_cost.values()))
        return value


def _reconstruct(parent, goal):
    states = [goal]
    while states[-1] in parent:
        states.append(parent[states[-1]])
    states.reverse()
    return states, [s.position for s in states]


def _result(name, started, expanded, frontier, memory, cost, path_states, found):
    states, path = path_states if path_states else ([], [])
    return SearchResult(name, path, states, cost if found else float("inf"),
                        expanded, perf_counter() - started, frontier, memory, found)


def bfs(problem):
    from collections import deque
    started = perf_counter(); root = problem.initial_state(); q = deque([root])
    parent = {}; cost = {root: 0.0}; seen = {root}; expanded = 0; peak = 1
    while q:
        state = q.popleft(); expanded += 1
        if problem.is_goal(state): return _result("BFS", started, expanded, peak, len(seen), cost[state], _reconstruct(parent, state), True)
        for nxt, step in problem.successors(state):
            if nxt not in seen:
                seen.add(nxt); parent[nxt] = state; cost[nxt] = cost[state] + step; q.append(nxt)
        peak = max(peak, len(q))
    return _result("BFS", started, expanded, peak, len(seen), 0, None, False)


def _best_first(problem, mode, inadmissible=False):
    started = perf_counter(); root = problem.initial_state(); serial = count()
    heap = [(problem.heuristic(root, inadmissible) if mode == "greedy" else 0.0, next(serial), root)]
    g = {root: 0.0}; parent = {}; expanded = 0; peak = 1; closed = set()
    while heap:
        _, _, state = heappop(heap)
        if state in closed: continue
        closed.add(state); expanded += 1
        if problem.is_goal(state): return _result("Greedy Best-First" if mode == "greedy" else "Uniform Cost Search" if mode == "ucs" else "A*", started, expanded, peak, len(closed) + len(heap), g[state], _reconstruct(parent, state), True)
        for nxt, step in problem.successors(state):
            new_g = g[state] + step
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g; parent[nxt] = state
                h = problem.heuristic(nxt, inadmissible) if mode in ("greedy", "astar") else 0
                heappush(heap, (h if mode == "greedy" else new_g + h, next(serial), nxt))
        peak = max(peak, len(heap))
    return _result("Greedy Best-First" if mode == "greedy" else "Uniform Cost Search" if mode == "ucs" else "A*", started, expanded, peak, len(closed) + len(heap), 0, None, False)


def uniform_cost_search(problem): return _best_first(problem, "ucs")
def greedy_best_first_search(problem, inadmissible=False): return _best_first(problem, "greedy", inadmissible)
def a_star(problem, inadmissible=False): return _best_first(problem, "astar", inadmissible)


def depth_limited_search(problem, limit):
    started = perf_counter(); root = problem.initial_state(); expanded = 0; peak = 1; memory = 0
    def visit(state, depth, path, cost):
        nonlocal expanded, peak, memory
        expanded += 1; memory = max(memory, len(path)); peak = max(peak, len(path))
        if problem.is_goal(state): return state, cost, [state]
        if depth == limit: return None
        for nxt, step in problem.successors(state):
            if nxt in path: continue
            found = visit(nxt, depth + 1, path | {nxt}, cost + step)
            if found: return found[0], found[1], [state] + found[2]
        return None
    found = visit(root, 0, {root}, 0.0)
    if found:
        goal, cost, chain = found
        return _result("Depth-Limited Search", started, expanded, peak, memory, cost, (chain, [s.position for s in chain]), True)
    return _result("Depth-Limited Search", started, expanded, peak, memory, 0, None, False)


def iterative_deepening_search(problem, max_depth=100):
    started = perf_counter(); total_expanded = total_memory = total_frontier = 0
    for depth in range(max_depth + 1):
        result = depth_limited_search(problem, depth)
        total_expanded += result.nodes_expanded; total_memory = max(total_memory, result.max_memory_states); total_frontier = max(total_frontier, result.max_frontier)
        if result.found:
            result.algorithm = "Iterative Deepening Search"; result.nodes_expanded = total_expanded; result.max_memory_states = total_memory; result.max_frontier = total_frontier; result.runtime_seconds = perf_counter() - started; return result
    return _result("Iterative Deepening Search", started, total_expanded, total_frontier, total_memory, 0, None, False)


ALGORITHMS = {"bfs": bfs, "ucs": uniform_cost_search, "dls": depth_limited_search, "ids": iterative_deepening_search, "greedy": greedy_best_first_search, "astar": a_star}
