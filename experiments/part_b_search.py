"""Run a small Part B comparison without external search libraries."""
import numpy as np

from part_b.search import (DeliveryProblem, a_star, bfs,
                           greedy_best_first_search, uniform_cost_search)


def main():
    grid = np.zeros((10, 10), dtype=int)
    problem = DeliveryProblem(grid, (0, 0), [(2, 7), (8, 9), (9, 1)],
                              {"road": 1, "grass": 3, "mud": 7, "ford": 8})
    results = [bfs(problem), uniform_cost_search(problem),
               greedy_best_first_search(problem), a_star(problem),
               a_star(problem, inadmissible=True)]
    print("algorithm\tfound\tcost\texpanded\truntime_s\tmemory")
    for r in results:
        print(f"{r.algorithm}\t{r.found}\t{r.cost:.1f}\t{r.nodes_expanded}\t"
              f"{r.runtime_seconds:.6f}\t{r.max_memory_states}")


if __name__ == "__main__":
    main()
