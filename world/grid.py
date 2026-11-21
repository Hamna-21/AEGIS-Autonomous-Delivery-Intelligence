"""Grid loading and vectorized or loop-based terrain costs."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import numpy as np

TERRAIN_NAMES = {0: "road", 1: "grass", 2: "mud", 3: "ford"}
BLOCKED = -1


def load_grid(path):
    path = Path(path)
    if path.exists():
        grid = np.load(path, allow_pickle=False)
    else:
        # Development fallback while the binary starter asset is still in the
        # supplied ZIP. If a local copy exists, it always takes precedence.
        candidates = [
            Path(r"C:\Users\Dell\Downloads\Autonomous-Delivery-Agent-Part-A-Starter.zip"),
            Path(r"C:\Users\Dell\Downloads\Autonomous-Delivery-Agent-Part-A-Starter (1).zip"),
        ]
        member = f"Autonomous-Delivery-Agent/data/{path.name}"
        for archive_path in candidates:
            if archive_path.exists():
                with ZipFile(archive_path) as archive:
                    with archive.open(member) as stream:
                        grid = np.load(BytesIO(stream.read()), allow_pickle=False)
                break
        else:
            raise FileNotFoundError(f"Grid not found: {path}")
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("Expected a square 2-D grid.")
    if not np.isin(grid, [BLOCKED, *TERRAIN_NAMES]).all():
        raise ValueError("Grid contains an unknown terrain code.")
    return grid


def build_cost_map_vectorized(grid, terrain_cost):
    costs = np.full(grid.shape, np.inf, dtype=float)
    for code, name in TERRAIN_NAMES.items():
        costs[grid == code] = float(terrain_cost[name])
    return costs


def build_cost_map_loop(grid, terrain_cost):
    costs = [[float("inf") for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            code = int(grid[row, col])
            if code >= 0:
                costs[row][col] = float(terrain_cost[TERRAIN_NAMES[code]])
    return costs
