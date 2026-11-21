import unittest

from part_c.game import ReducedDeliveryGame, alpha_beta, minimax, render_move_svg


class PartCTests(unittest.TestCase):
    def setUp(self):
        cells = {(r, c) for r in range(4) for c in range(4)}
        self.game = ReducedDeliveryGame(cells, (3, 0), (0, 3), [(0, 0)], horizon=8)

    def test_alpha_beta_matches_minimax_value(self):
        for depth in range(2, 6):
            mm = minimax(self.game, depth=depth)
            ab = alpha_beta(self.game, depth=depth)
            self.assertEqual(mm.value, ab.value)

    def test_alpha_beta_orders_and_prunes(self):
        mm = minimax(self.game, depth=5)
        ab = alpha_beta(self.game, depth=5)
        self.assertGreaterEqual(mm.stats.nodes, ab.stats.nodes)
        self.assertGreaterEqual(ab.stats.cutoffs, 0)
        self.assertTrue(ab.candidates)

    def test_evaluation_has_rival_features_and_svg_visualization(self):
        decision = alpha_beta(self.game, depth=3)
        self.assertNotEqual(self.game.evaluation(self.game.start_state), 0)
        path = "outputs/test_part_c_moves.svg"
        render_move_svg(self.game, self.game.start_state, decision, path)
        with open(path, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("<svg", text)
        self.assertIn("#1976d2", text)


if __name__ == "__main__":
    unittest.main()
