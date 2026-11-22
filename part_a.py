"""Part A world report generated from the official CMS-derived instance."""
import json
import time

import numpy as np

from config import ACTIVE_TWISTS, GRID, GROUP_STAMP, INSTANCE, INSTANCE_ID
from world.grid import build_cost_map_loop, build_cost_map_vectorized


def benchmark(grid, terrain_cost, repeats=100):
    start = time.perf_counter()
    for _ in range(repeats):
        vectorized = build_cost_map_vectorized(grid, terrain_cost)
    vector_seconds = (time.perf_counter() - start) / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        loop = build_cost_map_loop(grid, terrain_cost)
    loop_seconds = (time.perf_counter() - start) / repeats
    if not np.allclose(vectorized, np.asarray(loop), equal_nan=True):
        raise AssertionError("vectorized and loop cost maps differ")
    return vector_seconds, loop_seconds, loop_seconds / vector_seconds if vector_seconds else float("inf")


def main():
    vector_s, loop_s, speedup = benchmark(GRID, INSTANCE["terrain_cost"])
    report = {
        "instance_id": INSTANCE_ID, "grid_shape": list(GRID.shape),
        "blocked_cells": int(np.count_nonzero(GRID == -1)),
        "passable_cells": int(np.count_nonzero(GRID >= 0)),
        "vectorized_seconds_per_run": vector_s,
        "python_loop_seconds_per_run": loop_s, "speedup": speedup,
        "active_twists": ACTIVE_TWISTS,
    }
    print(GROUP_STAMP)
    print(json.dumps(report, indent=2))
    print("PEAS")
    print("Performance: minimize computed delivery cost while completing every generated stop safely.")
    print(f"Environment: generated discrete grid with twists {', '.join(ACTIVE_TWISTS)}.")
    print("Actuators: move, wait, deliver, and replan after new observations or slips.")
    print("Sensors: position, terrain, obstacles, delivered stops, time, fog observations, and rival state.")
    print("Environment classification")
    print("observable: partially observable because fog hides cells outside the configured view radius.")
    print("deterministic: stochastic because slip execution changes the intended movement outcome.")
    print("episodic: sequential because each action changes the next delivery state and elapsed time.")
    print("static: dynamic because time, deadlines, fog knowledge, and rival position change during execution.")
    print("discrete: discrete because positions, actions, stops, and time are discrete values.")
    print("multi-agent: multi-agent because the generated instance includes an adversary.")
    print("agent_type: utility-based because it trades delivery progress, cost, deadlines, and safety.")
    print("simple_reflex_limit: a reflex rule cannot order multiple stops or reason about future time and rival risk.")


if __name__ == "__main__":
    main()
