"""Part D command interface with an explicit offline, non-LLM mode.

The offline parser is intentionally deterministic. It is not described as an
LLM and can be replaced by a LangChain or provider-backed parser later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import re
import random
from pathlib import Path

import numpy as np

from part_b.search import DeliveryProblem, a_star, bfs, greedy_best_first_search, uniform_cost_search
from world.flood import spread_flood


ALGORITHMS = {"bfs": bfs, "ucs": uniform_cost_search, "uniform cost": uniform_cost_search,
              "a*": a_star, "astar": a_star, "greedy": greedy_best_first_search}


@dataclass
class Event:
    timestamp: str
    phase: str
    name: str
    arguments: dict
    observation: dict


class ValidationError(ValueError):
    pass


def default_grid():
    grid = np.zeros((7, 7), dtype=int)
    for cell in ((1, 2), (2, 2), (4, 4), (5, 4)):
        grid[cell] = -1
    return grid


class AEGISCommandInterface:
    """Natural-language facade over validated Part B tools.

    `mode` is always `offline_parser` unless a future provider adapter is
    explicitly added. It must never be mistaken for a real LLM response.
    """

    def __init__(self, grid=None, terrain_cost=None, log_path=None):
        self.grid = np.asarray(grid if grid is not None else default_grid(), dtype=int)
        self.terrain_cost = terrain_cost or {"road": 1, "grass": 3, "mud": 7, "ford": 8}
        self.mode = "offline_parser"
        self.api_key_available = bool(os.getenv("OPENAI_API_KEY"))
        self.events = []
        self.log_path = Path(log_path) if log_path else None

    def _event(self, phase, name, arguments, observation):
        event = Event(datetime.now(timezone.utc).isoformat(), phase, name,
                      self._jsonable(arguments), self._jsonable(observation))
        self.events.append(event)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(event)) + "\n")

    def _jsonable(self, value):
        if isinstance(value, dict): return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)): return [self._jsonable(v) for v in value]
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        return value

    def _cell(self, value, label):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(not isinstance(x, int) for x in value):
            raise ValidationError(f"{label} must be [row, col]")
        cell = tuple(value)
        if not (0 <= cell[0] < self.grid.shape[0] and 0 <= cell[1] < self.grid.shape[1]):
            raise ValidationError(f"{label} is outside the grid")
        if self.grid[cell] < 0:
            raise ValidationError(f"{label} is an obstacle")
        return cell

    def _problem(self, start, stops, flooded_cells=(), flood_enabled=False, flood_interval=5):
        start = self._cell(start, "start")
        if not isinstance(stops, list) or not stops:
            raise ValidationError("stops must be a non-empty list")
        stops = [self._cell(stop, "stop") for stop in stops]
        flooded = [self._cell(cell, "flooded cell") for cell in flooded_cells]
        if not isinstance(flood_interval, int) or flood_interval <= 0:
            raise ValidationError("flood_interval must be a positive integer")
        return DeliveryProblem(self.grid, start, stops, self.terrain_cost,
                               flooded_cells=flooded, flood_enabled=bool(flood_enabled),
                               flood_interval=flood_interval)

    def plan_route(self, start, stops, algorithm="a*", flooded_cells=None,
                   flood_enabled=False, flood_interval=5):
        args = {"start": start, "stops": stops, "algorithm": algorithm,
                "flooded_cells": flooded_cells or [], "flood_enabled": flood_enabled,
                "flood_interval": flood_interval}
        self._event("Sense", "plan_route.request", args, {})
        key = str(algorithm).lower().strip()
        if key not in ALGORITHMS:
            obs = {"ok": False, "error": f"unsupported algorithm: {algorithm}"}
            self._event("Reason", "plan_route.validation", args, obs)
            raise ValidationError(obs["error"])
        problem = self._problem(start, stops, flooded_cells or [], flood_enabled, flood_interval)
        result = ALGORITHMS[key](problem)
        observation = {"found": result.found, "cost": result.cost if result.found else None,
                       "path": result.path if result.found else [], "nodes_expanded": result.nodes_expanded,
                       "runtime_seconds": result.runtime_seconds}
        self._event("Reason", "plan_route.search", args, observation)
        if not result.found:
            raise ValidationError("no route exists for the requested delivery")
        self._event("Act", "plan_route.result", args, observation)
        return {"ok": True, "mode": self.mode, "algorithm": result.algorithm, **observation}

    def spread_flood(self, grid=None, flooded_cells=None):
        raw_grid = self.grid if grid is None else np.asarray(grid, dtype=int)
        flooded_cells = flooded_cells or []
        args = {"flooded_cells": flooded_cells, "grid_shape": list(raw_grid.shape)}
        self._event("Sense", "spread_flood.request", args, {})
        if raw_grid.ndim != 2 or raw_grid.shape[0] != raw_grid.shape[1]:
            raise ValidationError("grid must be a square matrix")
        cells = [self._cell(cell, "flooded cell") for cell in flooded_cells]
        changed, result_cells = spread_flood(raw_grid, set(cells))
        observation = {"ok": True, "flooded_cells": sorted(result_cells), "grid": changed.tolist()}
        self._event("Act", "spread_flood.result", args, observation)
        return observation

    def simulate(self, start, stops, slip_probability=0.0, seed=0, algorithm="a*",
                 flooded_cells=None, flood_enabled=False, flood_interval=5):
        if not isinstance(slip_probability, (int, float)) or not 0 <= slip_probability <= 1:
            raise ValidationError("slip_probability must be between 0 and 1")
        args = {"start": start, "stops": stops, "slip_probability": slip_probability,
                "seed": seed, "algorithm": algorithm}
        self._event("Sense", "simulate.request", args, {})
        plan = self.plan_route(start, stops, algorithm, flooded_cells, flood_enabled, flood_interval)
        rng = random.Random(seed); actual = list(plan["path"]); slips = 0
        if slip_probability:
            for index in range(1, len(actual)):
                if rng.random() < slip_probability:
                    slips += 1
                    actual[index] = actual[index - 1]
        observation = {"ok": True, "planned_cost": plan["cost"], "planned_path": plan["path"],
                       "executed_path": actual, "slips": slips,
                       "replanned": bool(slips), "final_position": actual[-1]}
        self._event("Reason", "simulate.execution", args, observation)
        self._event("Act", "simulate.result", args, observation)
        return observation

    def parse(self, request):
        if not isinstance(request, str) or not request.strip():
            raise ValidationError("request must be non-empty text")
        text = request.strip()
        coordinates = [tuple(map(int, pair)) for pair in re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)]
        start_match = re.search(r"from\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text, re.I)
        if not start_match or len(coordinates) < 2:
            raise ValidationError("use 'from (row,col) to (row,col)' with at least one stop")
        start = tuple(map(int, start_match.groups()))
        stops = [cell for cell in coordinates if cell != start]
        algorithm = "a*"
        for name in ("bfs", "ucs", "uniform cost", "a*", "astar", "greedy"):
            if name in text.lower(): algorithm = name; break
        simulate = any(word in text.lower() for word in ("simulate", "slip", "execute"))
        slip = 0.11 if "slip" in text.lower() else 0.0
        if simulate:
            return {"tool": "simulate", "args": {"start": start, "stops": stops,
                    "slip_probability": slip, "algorithm": algorithm, "seed": 0}}
        return {"tool": "plan_route", "args": {"start": start, "stops": stops, "algorithm": algorithm}}

    def handle(self, request):
        self._event("Sense", "user.request", {"text": request}, {"mode": self.mode})
        parsed = self.parse(request)
        self._event("Reason", "offline_parser.parse", {"text": request}, parsed)
        result = getattr(self, parsed["tool"])(**parsed["args"])
        return {"mode": self.mode, "request": request, "parsed": parsed, "result": result,
                "trace": [asdict(event) for event in self.events[-6:]]}

    def metrics(self, expected):
        if not expected:
            return {"requests": 0, "success_rate": 0.0, "mean_cost": None, "mean_tool_calls": 0.0}
        successes = sum(1 for item in expected if item["success"])
        costs = [item["cost"] for item in expected if item["cost"] is not None]
        return {"requests": len(expected), "success_rate": successes / len(expected),
                "mean_cost": sum(costs) / len(costs) if costs else None,
                "mean_tool_calls": sum(item["tool_calls"] for item in expected) / len(expected)}
