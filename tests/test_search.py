import unittest
import numpy as np

from part_b.search import (DeliveryProblem, a_star, bfs,
                           depth_limited_search, greedy_best_first_search,
                           iterative_deepening_search, uniform_cost_search)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((4, 5), dtype=int)
        self.costs = {"road": 1, "grass": 3, "mud": 7, "ford": 8}
        self.problem = DeliveryProblem(self.grid, (0, 0), [(0, 2), (3, 4)], self.costs)

    def test_all_algorithms_deliver_all_stops(self):
        results = [bfs(self.problem), uniform_cost_search(self.problem),
                   depth_limited_search(self.problem, 20),
                   iterative_deepening_search(self.problem, 20),
                   greedy_best_first_search(self.problem), a_star(self.problem)]
        for result in results:
            self.assertTrue(result.found, result.algorithm)
            self.assertEqual(result.states[-1].delivered, frozenset({0, 1}))
            self.assertEqual(result.path[0], (0, 0))

    def test_obstacle_and_directed_edge_are_respected(self):
        grid = self.grid.copy(); grid[0, 1] = -1
        edges = {((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (1, 2))}
        problem = DeliveryProblem(grid, (0, 0), [(1, 2)], self.costs, edges)
        result = uniform_cost_search(problem)
        self.assertTrue(result.found)
        self.assertEqual(result.path, [(0, 0), (1, 0), (1, 1), (1, 2)])
        self.assertFalse(any(a == (0, 1) or b == (0, 1) for a, b in zip(result.path, result.path[1:])))

    def test_flood_is_time_dependent_and_blocking(self):
        grid = self.grid.copy(); grid[0, 1] = 2
        problem = DeliveryProblem(grid, (0, 0), [(0, 2)], self.costs,
                                 flooded_cells={(0, 1)}, flood_interval=1,
                                 flood_enabled=True)
        # The cell is still a mud cell in the static map, but the time-one
        # flood transition turns it into an obstacle before it can be entered.
        self.assertFalse(any(state.position == (0, 1)
                             for state, _ in problem.successors(problem.initial_state())))

    def test_heuristic_is_admissible_on_small_instances(self):
        for state, _ in self.problem.successors(self.problem.initial_state()):
            h = self.problem.heuristic(state)
            optimum = uniform_cost_search(DeliveryProblem(
                self.grid, state.position,
                [self.problem.stops[i] for i in range(len(self.problem.stops)) if i not in state.delivered],
                self.costs)).cost
            self.assertLessEqual(h, optimum)

    def test_heuristic_is_empirically_consistent(self):
        pending = [self.problem.initial_state()]
        seen = set(pending)
        while pending:
            state = pending.pop()
            for nxt, step in self.problem.successors(state):
                self.assertLessEqual(self.problem.heuristic(state),
                                     step + self.problem.heuristic(nxt) + 1e-9)
                if nxt not in seen and nxt.time < 5:
                    seen.add(nxt)
                    pending.append(nxt)

    def test_cycle_avoidance_and_ids_memory(self):
        edges = {((0, 0), (0, 1)), ((0, 1), (0, 0)), ((0, 1), (0, 2))}
        problem = DeliveryProblem(np.zeros((1, 3), dtype=int), (0, 0), [(0, 2)], self.costs, edges)
        result = iterative_deepening_search(problem, 5)
        self.assertTrue(result.found)
        self.assertLessEqual(result.max_memory_states, 6)
        self.assertEqual(result.path, [(0, 0), (0, 1), (0, 2)])


if __name__ == "__main__":
    unittest.main()
