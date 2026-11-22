"""Part C benchmark: search comparison, visualization, and 30-game evaluation."""
from pathlib import Path
import random

from part_c.game import (ReducedDeliveryGame, alpha_beta, confidence_interval,
                         minimax, render_move_svg)
from config import GRID, INSTANCE, GROUP_STAMP, INSTANCE_ID


def make_game(seed, horizon=14):
    rng = random.Random(seed)
    passable = {(r, c) for r in range(GRID.shape[0]) for c in range(GRID.shape[1]) if GRID[r, c] >= 0}
    start = tuple(INSTANCE["start"])
    rival = sorted(passable, key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]), reverse=True)[seed % 20]
    targets = (tuple(INSTANCE["delivery_stops"][seed % len(INSTANCE["delivery_stops"])]),)
    return ReducedDeliveryGame(passable, start, rival, targets, horizon)


def execute(game, depth, slip_probability, seed):
    rng = random.Random(seed); state = game.start_state
    while game.terminal_value(state) is None:
        decision = alpha_beta(game, state, depth)
        intended = decision.move
        actual = intended
        if rng.random() < slip_probability:
            actual = rng.choice([action for action, _ in game.legal_moves(state.agent)])
        state = game.move(state, "agent", actual)
        if game.terminal_value(state) is not None:
            break
        # Adversarial rival move, then replan on the next turn from the actual
        # state. This is the replanning response to unexpected slip.
        state = min((game.move(state, "rival", action)
                     for action, _ in game.legal_moves(state.rival)),
                    key=game.evaluation)
    value = game.terminal_value(state)
    return "win" if value == 1 else "loss" if value == -1 else "draw"


def summarize(outcomes):
    n = len(outcomes); result = {}
    for label in ("win", "draw", "loss"):
        count = outcomes.count(label); rate = count / n
        result[label] = {"count": count, "rate": rate,
                         "ci95": confidence_interval(rate, n)}
    return result


def main():
    root = Path("outputs"); root.mkdir(exist_ok=True)
    game = make_game(0)
    print("depth\tminimax_nodes\talpha_beta_nodes\tpruning_percent\tvalue")
    for depth in range(2, INSTANCE["adversary_search_depth"] + 1):
        mm = minimax(game, depth=depth); ab = alpha_beta(game, depth=depth)
        pruning = 100.0 * (mm.stats.nodes - ab.stats.nodes) / max(1, mm.stats.nodes)
        print(f"{depth}\t{mm.stats.nodes}\t{ab.stats.nodes}\t{pruning:.2f}\t{ab.value:.2f}")
    render_move_svg(game, game.start_state, alpha_beta(game, depth=INSTANCE["adversary_search_depth"]),
                    root / "part_c_candidate_moves.svg")
    for slip in (0.0, INSTANCE["slip_probability"]):
        # Depth 2 keeps the 60-game evaluation reproducible and quick while
        # the table above benchmarks every depth supported by the generated instance.
        outcomes = [execute(make_game(i), 2, slip, 1000 + i) for i in range(30)]
        print(GROUP_STAMP, f"instance_id: {INSTANCE_ID}", ("deterministic" if slip == 0 else "stochastic_slip"), summarize(outcomes))


if __name__ == "__main__":
    main()
