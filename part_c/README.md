# Part C adversarial planning

Part C uses a reduced game representation: agent position, rival position,
delivery progress, and ply count. The agent maximizes the score and the rival
minimizes it. A delivered route is a win, capture is a loss, and reaching the
horizon is a draw.

`minimax` and `alpha_beta` are implemented directly in `part_c/game.py`.
Alpha-Beta uses custom move ordering. Agent moves are ordered by distance to the
next delivery and safety from the rival. Rival moves are ordered by distance to
the agent.

The evaluation function has five weighted features: delivery progress, agent
distance to the next target, distance between rival and agent, agent mobility,
and rival mobility. Terminal outcomes receive a large score so that winning,
losing, and drawing are preferred correctly.

The experiment runs rival depths 2 through the generated depth, records node
counts for both algorithms, and calculates pruning percentage as
`100 * (minimax_nodes - alpha_beta_nodes) / minimax_nodes`.

Stochastic execution applies slip to the chosen agent movement. The rival then
moves, and the agent replans from the actual post-slip state on the next turn.
The experiment runs 30 games for deterministic planning and 30 games for slip
execution, reporting wins, draws, losses, and normal-approximation 95 percent
confidence intervals for each outcome. The SVG output marks candidate moves in
light blue and the selected move in dark blue.
