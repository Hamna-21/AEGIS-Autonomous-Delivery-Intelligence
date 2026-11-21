"""Optional proposed flood twist.

This module is disabled by default and is awaiting instructor approval. It is
kept separate from the core world model because flood is not in the supplied
generated instance's active twists.
"""
import numpy as np

DEGRADE = {0: 3, 3: 2, 2: -1, 1: 1}


def spread_flood(grid, flooded_cells):
    next_grid = grid.copy()
    next_flooded = set(flooded_cells)
    for row, col in flooded_cells:
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] >= 0:
                next_flooded.add((nr, nc))
    for row, col in next_flooded:
        if next_grid[row, col] >= 0:
            next_grid[row, col] = DEGRADE[int(next_grid[row, col])]
    return next_grid, next_flooded


def advance_flood(grid, flooded_cells, time_step, interval=5, enabled=False):
    if interval <= 0:
        raise ValueError("interval must be positive")
    if not enabled or time_step == 0 or time_step % interval:
        return grid.copy(), set(flooded_cells)
    return spread_flood(grid, flooded_cells)
