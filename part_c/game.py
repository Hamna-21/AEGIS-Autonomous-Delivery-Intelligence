"""Part C adversarial delivery game and from-scratch game-tree search."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

Cell = tuple[int, int]
PASS = "wait"
MOVES = ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"),
         (0, 1, "right"), (0, 0, PASS))


@dataclass(frozen=True)
class GameState:
    agent: Cell
    rival: Cell
    target_index: int
    ply: int = 0


@dataclass
class SearchStats:
    nodes: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0


@dataclass
class Decision:
    move: str
    value: float
    candidates: list[tuple[str, float]]
    stats: SearchStats
    algorithm: str
    depth: int


class ReducedDeliveryGame:
    """Small deterministic game used by planning and stochastic execution.

    The agent moves first. After an agent move, the rival chooses a move.
    Reaching the target advances delivery. A collision is a loss for the
    agent. Delivery of all targets is a win. A depth limit produces a draw.
    """

    def __init__(self, passable: Iterable[Cell], start: Cell, rival_start: Cell,
                 targets: Iterable[Cell], horizon=12):
        self.passable = frozenset(tuple(x) for x in passable)
        self.targets = tuple(tuple(x) for x in targets)
        self.horizon = int(horizon)
        self.start_state = GameState(tuple(start), tuple(rival_start), 0, 0)
        if self.start_state.agent not in self.passable or self.start_state.rival not in self.passable:
            raise ValueError("start positions must be passable")

    def legal_moves(self, position: Cell):
        moves = []
        for dr, dc, name in MOVES:
            nxt = (position[0] + dr, position[1] + dc)
            if nxt in self.passable:
                moves.append((name, nxt))
        return moves

    def move(self, state: GameState, actor: str, action: str):
        source = state.agent if actor == "agent" else state.rival
        options = dict(self.legal_moves(source))
        nxt = options.get(action, source)
        if actor == "agent":
            index = state.target_index
            if index < len(self.targets) and nxt == self.targets[index]:
                index += 1
            return GameState(nxt, state.rival, index, state.ply + 1)
        return GameState(state.agent, nxt, state.target_index, state.ply)

    def terminal_value(self, state: GameState):
        if state.agent == state.rival:
            return -1.0
        if state.target_index == len(self.targets):
            return 1.0
        if state.ply >= self.horizon:
            return 0.0
        return None

    def rival_successors(self, state: GameState):
        for action, _ in self.legal_moves(state.rival):
            yield action, self.move(state, "rival", action)

    def after_agent_move(self, state: GameState, action: str):
        moved = self.move(state, "agent", action)
        if self.terminal_value(moved) is not None:
            return [(PASS, moved)]
        return list(self.rival_successors(moved))

    def _distance(self, a: Cell, b: Cell):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def evaluation(self, state: GameState):
        """Rival evaluation with five weighted features, from agent view."""
        terminal = self.terminal_value(state)
        if terminal is not None:
            return terminal * 1000.0
        target = self.targets[state.target_index]
        agent_distance = self._distance(state.agent, target)
        rival_distance = self._distance(state.rival, state.agent)
        agent_mobility = len(self.legal_moves(state.agent))
        rival_mobility = len(self.legal_moves(state.rival))
        progress = state.target_index / max(1, len(self.targets))
        return (100.0 * progress - 2.0 * agent_distance +
                3.0 * rival_distance + 1.0 * agent_mobility -
                1.5 * rival_mobility)

    def ordered_agent_moves(self, state: GameState):
        target = self.targets[state.target_index] if state.target_index < len(self.targets) else state.agent
        return sorted(self.legal_moves(state.agent),
                      key=lambda item: (self._distance(item[1], target),
                                        self._distance(item[1], state.rival), item[0]))

    def ordered_rival_moves(self, state: GameState):
        return sorted(self.legal_moves(state.rival),
                      key=lambda item: (self._distance(item[1], state.agent), item[0]))


def minimax(game: ReducedDeliveryGame, state=None, depth=4):
    return _search(game, state or game.start_state, depth, False, False)


def alpha_beta(game: ReducedDeliveryGame, state=None, depth=4):
    return _search(game, state or game.start_state, depth, True, True)


def _search(game, root, depth, pruning, ordering):
    stats = SearchStats()

    def value(state, remaining, maximizing):
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth - remaining)
        terminal = game.terminal_value(state)
        if terminal is not None or remaining == 0:
            return game.evaluation(state)
        if maximizing:
            best = -float("inf")
            moves = game.ordered_agent_moves(state) if ordering else game.legal_moves(state.agent)
            for action, _ in moves:
                children = game.after_agent_move(state, action)
                child_value = min(value(child, remaining - 1, False) for _, child in children)
                best = max(best, child_value)
            return best
        best = float("inf")
        moves = game.ordered_rival_moves(state) if ordering else game.legal_moves(state.rival)
        for action, _ in moves:
            child = game.move(state, "rival", action)
            best = min(best, value(child, remaining - 1, True))
        return best

    def ab_value(state, remaining, maximizing, alpha, beta):
        stats.nodes += 1
        stats.max_depth_reached = max(stats.max_depth_reached, depth - remaining)
        terminal = game.terminal_value(state)
        if terminal is not None or remaining == 0:
            return game.evaluation(state)
        if maximizing:
            best = -float("inf")
            moves = game.ordered_agent_moves(state) if ordering else game.legal_moves(state.agent)
            for action, _ in moves:
                children = game.after_agent_move(state, action)
                child_value = min(ab_value(child, remaining - 1, False, alpha, beta) for _, child in children)
                best = max(best, child_value); alpha = max(alpha, best)
                if beta <= alpha:
                    stats.cutoffs += 1; break
            return best
        best = float("inf")
        moves = game.ordered_rival_moves(state) if ordering else game.legal_moves(state.rival)
        for action, _ in moves:
            child = game.move(state, "rival", action)
            best = min(best, ab_value(child, remaining - 1, True, alpha, beta)); beta = min(beta, best)
            if beta <= alpha:
                stats.cutoffs += 1; break
        return best

    candidates = []
    moves = game.ordered_agent_moves(root) if ordering else game.legal_moves(root.agent)
    for action, _ in moves:
        children = game.after_agent_move(root, action)
        if pruning:
            score = min(ab_value(child, depth - 1, False, -float("inf"), float("inf")) for _, child in children)
        else:
            score = min(value(child, depth - 1, False) for _, child in children)
        candidates.append((action, score))
    # Prefer making progress over waiting when scores tie. This keeps the
    # reduced benchmark from selecting PASS merely because it sorts later.
    chosen, score = max(candidates,
                        key=lambda item: (item[1], item[0] != PASS, item[0]))
    return Decision(chosen, score, candidates, stats,
                    "Alpha-Beta" if pruning else "Minimax", depth)


def confidence_interval(rate, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    margin = z * sqrt(rate * (1.0 - rate) / n)
    return max(0.0, rate - margin), min(1.0, rate + margin)


def render_move_svg(game, state, decision, path: str | Path):
    """Write a dependency-free SVG showing candidate and selected moves."""
    rows = max(r for r, _ in game.passable) + 1
    cols = max(c for _, c in game.passable) + 1
    size = 44; width, height = cols * size, rows * size
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    for r in range(rows):
        for c in range(cols):
            fill = "#f2f2f2" if (r, c) in game.passable else "#333333"
            parts.append(f'<rect x="{c*size}" y="{r*size}" width="{size}" height="{size}" fill="{fill}" stroke="#bbbbbb"/>')
    for action, _ in decision.candidates:
        target = dict(game.legal_moves(state.agent)).get(action, state.agent)
        x1, y1 = state.agent[1] * size + size // 2, state.agent[0] * size + size // 2
        x2, y2 = target[1] * size + size // 2, target[0] * size + size // 2
        color = "#1976d2" if action == decision.move else "#90caf9"
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{5 if action == decision.move else 2}"/>')
    for pos, color in ((state.agent, "#2e7d32"), (state.rival, "#c62828")):
        parts.append(f'<circle cx="{pos[1]*size+size//2}" cy="{pos[0]*size+size//2}" r="12" fill="{color}"/>')
    for target in game.targets[state.target_index:]:
        parts.append(f'<circle cx="{target[1]*size+size//2}" cy="{target[0]*size+size//2}" r="8" fill="#ffb300"/>')
    parts.append("</svg>")
    Path(path).write_text("".join(parts), encoding="utf-8")
