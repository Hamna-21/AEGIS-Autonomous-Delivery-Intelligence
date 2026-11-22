"""Part A environment facade."""
from world.flood import advance_flood, spread_flood
from world.fog import observe, visible_mask
from world.grid import (BLOCKED, TERRAIN_NAMES, build_cost_map_loop,
                        build_cost_map_vectorized, load_grid)
from world.transitions import (apply_slip, is_legal_move, legal_neighbors)

__all__ = ["advance_flood", "spread_flood", "observe", "visible_mask",
           "BLOCKED", "TERRAIN_NAMES", "build_cost_map_loop",
           "build_cost_map_vectorized", "load_grid", "apply_slip",
           "is_legal_move", "legal_neighbors"]
