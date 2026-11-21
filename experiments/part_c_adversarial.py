"""Part C benchmark: search comparison, visualization, and 30-game evaluation."""
from pathlib import Path
import random

from part_c.game import (ReducedDeliveryGame, alpha_beta, confidence_interval,
                         minimax, render_move_svg)


def make_game(seed, horizon=14):
    rng = random.Random(seed)
    passable = {(r, c) for r in range(7) for c in range(7)}
    # Keep a few deterministic obstacle cells while preserving a connected
    # open corridor for repeatable experiments.
    for cell in ((1, 2), (2, 2), (4, 4), (5, 4)):
        passable.discard(cell)
    start = (6, 0)
    # Mix a safe and a pressure scenario so the 30-game report contains
    # meaningful outcomes instead of an all-draw distribution.
    rival = (5, 1) if seed % 3 == 0 else (0, 6)
    targets = ((0, 1),) if seed % 2 else ((0, 0),)
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
    for depth in range(2, 7):
        mm = minimax(game, depth=depth); ab = alpha_beta(game, depth=depth)
        pruning = 100.0 * (mm.stats.nodes - ab.stats.nodes) / max(1, mm.stats.nodes)
        print(f"{depth}\t{mm.stats.nodes}\t{ab.stats.nodes}\t{pruning:.2f}\t{ab.value:.2f}")
    render_move_svg(game, game.start_state, alpha_beta(game, depth=6),
                    root / "part_c_candidate_moves.svg")
    for slip in (0.0, 0.11):
        # Depth 2 keeps the 60-game evaluation reproducible and quick while
        # the table above still benchmarks every supported depth through 6.
        outcomes = [execute(make_game(i), 2, slip, 1000 + i) for i in range(30)]
        print(("deterministic" if slip == 0 else "stochastic_slip"), summarize(outcomes))


if __name__ == "__main__":
    main()
