"""Canonical Part C game facade."""
from part_c.adversarial import (Decision, GameState, ReducedDeliveryGame,
                                alpha_beta, confidence_interval, minimax,
                                render_move_svg)
from config import GROUP_STAMP, INSTANCE, INSTANCE_ID

__all__ = ["Decision", "GameState", "ReducedDeliveryGame", "minimax",
           "alpha_beta", "confidence_interval", "render_move_svg",
           "GROUP_STAMP", "INSTANCE", "INSTANCE_ID"]
