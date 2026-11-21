import random
import unittest
import numpy as np

from world.flood import advance_flood, spread_flood
from world.fog import observe, visible_mask
from world.grid import build_cost_map_loop, build_cost_map_vectorized, load_grid
from world.transitions import apply_slip, is_legal_move, legal_neighbors


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([[0, 1, -1], [2, 3, 0], [-1, 0, 0]], dtype=int)
        self.costs = {"road": 1, "grass": 3, "mud": 7, "ford": 8}

    def test_vectorized_and_loop_costs_match(self):
        vector = build_cost_map_vectorized(self.grid, self.costs)
        loop = np.asarray(build_cost_map_loop(self.grid, self.costs))
        self.assertTrue(np.allclose(vector, loop, equal_nan=True))
        self.assertTrue(np.isinf(vector[0, 2]))

    def test_grid_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            load_grid(__file__)

    def test_obstacles_and_bounds_are_respected(self):
        self.assertEqual(set(legal_neighbors(self.grid, (0, 0))), {(0, 1), (1, 0)})
        self.assertFalse(is_legal_move(self.grid, (0, 0), (0, 2)))

    def test_one_way_edge_restriction(self):
        edges = {((0, 0), (0, 1))}
        self.assertTrue(is_legal_move(self.grid, (0, 0), (0, 1), edges))
        self.assertFalse(is_legal_move(self.grid, (0, 1), (0, 0), edges))

    def test_slip_probability_zero_keeps_intended_move(self):
        result = apply_slip(self.grid, (0, 0), (0, 1), 0.0, random.Random(1))
        self.assertEqual(result.actual, (0, 1))
        self.assertFalse(result.slipped)

    def test_fog_hides_cells_outside_radius(self):
        observed, mask = observe(self.grid, (1, 1), 0)
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(observed[1, 1], 3)
        self.assertEqual(observed[0, 0], -2)
        self.assertTrue(visible_mask(self.grid.shape, (1, 1), 1).sum() > 1)

    def test_flood_is_disabled_by_default(self):
        flooded = {(0, 0)}
        unchanged, cells = advance_flood(self.grid, flooded, 5)
        self.assertTrue(np.array_equal(unchanged, self.grid))
        self.assertEqual(cells, flooded)

    def test_flood_changes_grid_only_when_enabled(self):
        changed, cells = advance_flood(self.grid, {(0, 0)}, 5, enabled=True)
        self.assertNotEqual(changed[0, 0], self.grid[0, 0])
        self.assertIn((0, 1), cells)
        direct, _ = spread_flood(self.grid, {(0, 0)})
        self.assertTrue(np.array_equal(changed, direct))


if __name__ == "__main__":
    unittest.main()
