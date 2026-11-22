"""Canonical Part A facade driven by config.INSTANCE and config.GRID."""
from part_a.environment import (BLOCKED, TERRAIN_NAMES, advance_flood,
                                apply_slip, build_cost_map_loop,
                                build_cost_map_vectorized, is_legal_move,
                                legal_neighbors, load_grid, observe,
                                spread_flood, visible_mask)
from config import ACTIVE_TWISTS, GRID, GROUP_STAMP, INSTANCE, INSTANCE_ID

__all__ = ["INSTANCE", "INSTANCE_ID", "GRID", "GROUP_STAMP", "ACTIVE_TWISTS",
           "BLOCKED", "TERRAIN_NAMES", "advance_flood", "apply_slip",
           "build_cost_map_loop", "build_cost_map_vectorized", "is_legal_move",
           "legal_neighbors", "load_grid", "observe", "spread_flood",
           "visible_mask"]
