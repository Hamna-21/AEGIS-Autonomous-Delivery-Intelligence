import json
import unittest

from part_d import AEGISCommandInterface, ValidationError


class PartDTests(unittest.TestCase):
    def setUp(self):
        self.interface = AEGISCommandInterface()

    def test_twenty_natural_language_requests_dispatch(self):
        valid_targets = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                         (0, 6), (1, 0), (1, 1), (1, 3), (1, 4)]
        requests = [f"Plan a route from (0,0) to {valid_targets[i % 10]} using A*"
                    for i in range(20)]
        successes = 0
        for request in requests:
            try:
                response = self.interface.handle(request)
                successes += int(response["result"]["ok"])
            except ValidationError:
                pass
        self.assertGreaterEqual(successes, 18)
        self.assertTrue(all(event.phase in {"Sense", "Reason", "Act"}
                            for event in self.interface.events))

    def test_invalid_and_impossible_requests_are_rejected(self):
        with self.assertRaises(ValidationError):
            self.interface.plan_route([0, 0], [[99, 99]])
        with self.assertRaises(ValidationError):
            self.interface.plan_route([0, 0], [[1, 2]], algorithm="made-up")
        with self.assertRaises(ValidationError):
            self.interface.plan_route([0, 0], [[1, 2]], flood_interval=0)
        with self.assertRaises(ValidationError):
            self.interface.handle("please do something vague")

    def test_tools_and_trace_are_real(self):
        plan = self.interface.plan_route([0, 0], [[6, 6]])
        self.assertTrue(plan["ok"])
        simulation = self.interface.simulate([0, 0], [[6, 6]], slip_probability=1.0, seed=2)
        self.assertTrue(simulation["replanned"])
        flood = self.interface.spread_flood(flooded_cells=[[0, 0]])
        self.assertTrue(flood["ok"])
        names = [event.name for event in self.interface.events]
        self.assertIn("plan_route.search", names)
        self.assertIn("simulate.execution", names)
        self.assertIn("spread_flood.result", names)

    def test_offline_label_and_json_log(self):
        response = self.interface.handle("Plan a route from (0,0) to (0,1)")
        self.assertEqual(response["mode"], "offline_parser")
        json.dumps(response)


if __name__ == "__main__":
    unittest.main()
