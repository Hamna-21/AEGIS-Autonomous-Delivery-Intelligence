# Part B search

`part_b.search` implements BFS, Uniform Cost Search, Depth-Limited Search,
Iterative Deepening Search, Greedy Best-First Search, and A* from scratch.

The state is `(position, delivered, time, flooded)`. The delivered field is a
set of stop indexes. Time advances once per move. At each new time step the
existing Part A flood transition is applied, so a newly flooded cell is not
silently treated as traversable. `legal_neighbors` enforces bounds, obstacles,
and directed one-way edges.

## Plain-language algorithm guide

* BFS explores all routes with 0 moves, then 1 move, then 2 moves. It is a good
  baseline when every move costs the same.
* Uniform Cost Search always expands the cheapest route so far. It is the
  cost-optimal baseline for positive movement costs.
* Depth-Limited Search is depth-first search with a hard move limit. Its stack
  is small, but it can miss a solution beyond the limit.
* Iterative Deepening repeats depth-limited search with limits 0, 1, 2, and so
  on. It gets BFS-like shallow-solution behaviour with DFS-like memory.
* Greedy Best-First Search expands the state that appears closest to the
  remaining deliveries. It is often fast, but the path need not be cheapest.
* A* combines route cost so far with the same estimate of remaining cost. With
  the default heuristic it is cost-optimal.

## Heuristic argument and experiment

The default heuristic ignores obstacles, one-way restrictions, and flood
detours. It computes the cheapest Manhattan-distance ordering of the remaining
stops and multiplies the number of unit moves by the minimum terrain cost.
Every real route must make at least those coordinate changes, and every move
costs at least the minimum terrain cost. Therefore this relaxed value cannot
be greater than the true remaining cost, so it is admissible. The experiment
flag `inadmissible=True` replaces the minimum terrain cost with the maximum
terrain cost. This intentionally overestimates on cheap roads and lets A* be
compared with an inadmissible heuristic.

The returned `SearchResult` records solution cost, nodes expanded, runtime,
peak frontier size, and peak search-memory states. BFS and UCS are the primary
baselines. The tests also verify directed movement, flood timing, cycle
avoidance, and the bounded memory behaviour of IDS.
