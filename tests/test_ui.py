import unittest

import numpy as np

from ui.app import load_ui_grid, make_edges, make_problem, run_search


class UITests(unittest.TestCase):
    def test_grid_source_and_edges(self):
        grid, source = load_ui_grid()
        self.assertEqual(grid.ndim, 2)
        self.assertTrue(source)
        self.assertTrue(make_edges(grid))

    def test_live_search_controls_use_real_algorithms(self):
        grid = np.zeros((5, 5), dtype=int)
        for label in ("BFS", "Uniform Cost Search", "Depth-Limited Search", "Iterative Deepening Search", "Greedy Best-First", "A*"):
            result = run_search(make_problem(grid, (0, 0), [(4, 4)]), label)
            self.assertTrue(result.found, label)


if __name__ == "__main__":
    unittest.main()
