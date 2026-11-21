"""Configurable fog-of-war observations."""
import numpy as np


def visible_mask(shape, position, radius, metric="chebyshev"):
    if radius < 0:
        raise ValueError("radius must be non-negative")
    row, col = position
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    distance = np.maximum(np.abs(rr - row), np.abs(cc - col)) if metric == "chebyshev" else np.abs(rr - row) + np.abs(cc - col)
    if metric not in {"chebyshev", "manhattan"}:
        raise ValueError("metric must be chebyshev or manhattan")
    return distance <= radius


def observe(full_grid, position, radius, unknown=-2, metric="chebyshev"):
    mask = visible_mask(full_grid.shape, position, radius, metric)
    observed = np.full(full_grid.shape, unknown, dtype=int)
    observed[mask] = full_grid[mask]
    return observed, mask
