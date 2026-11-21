"""AEGIS Part A report for the environment and world model."""
import json
import time
from pathlib import Path
import numpy as np

from common import DATA_DIR, load_instance, stamp
from world.grid import build_cost_map_loop, build_cost_map_vectorized, load_grid


def benchmark(grid, terrain_cost, repeats=100):
    build_cost_map_vectorized(grid, terrain_cost)
    build_cost_map_loop(grid, terrain_cost)
    start = time.perf_counter()
    for _ in range(repeats):
        vectorized = build_cost_map_vectorized(grid, terrain_cost)
    vector_seconds = (time.perf_counter() - start) / repeats
    start = time.perf_counter()
    for _ in range(repeats):
        loop = build_cost_map_loop(grid, terrain_cost)
    loop_seconds = (time.perf_counter() - start) / repeats
    if not np.allclose(vectorized, np.asarray(loop), equal_nan=True):
        raise AssertionError("Vectorized and loop cost maps differ")
    return vector_seconds, loop_seconds, loop_seconds / vector_seconds if vector_seconds else float("inf")


def main():
    instance = load_instance()
    grid = load_grid(DATA_DIR / f"grid_{instance['instance_id']}.npy")
    vector_s, loop_s, speedup = benchmark(grid, instance["terrain_cost"])
    report = {
        "instance_id": instance["instance_id"],
        "grid_shape": list(grid.shape),
        "blocked_cells": int(np.count_nonzero(grid == -1)),
        "passable_cells": int(np.count_nonzero(grid >= 0)),
        "vectorized_seconds_per_run": vector_s,
        "python_loop_seconds_per_run": loop_s,
        "speedup": speedup,
        "active_twists": instance["twists"],
        "flood_status": "disabled; awaiting instructor approval",
    }
    print(stamp())
    print("\nPART A - NUMPY COST MAP BENCHMARK")
    print(json.dumps(report, indent=2))
    print("\nPEAS")
    print("Performance: deliver all parcels safely, legally, and with low cost and time.")
    print("Environment: grid city with terrain costs, obstacles, fog, one-way streets, slips, and a rival.")
    print("Actuators: move, wait, recompute route, and deliver.")
    print("Sensors: visible terrain, obstacles, position, time, delivered stops, and visible rival state.")
    print("\nENVIRONMENT CLASSIFICATION")
    print("Partially observable, stochastic, sequential, dynamic, discrete, and multi-agent.")
    print("\nAGENT TYPE")
    print("A utility-based agent can trade off delivery progress, cost, time, safety, and rival risk.")
    print("A simple reflex agent is insufficient because it cannot plan multi-stop routes or anticipate future risks.")


if __name__ == "__main__":
    main()
