import csv
import tempfile
import unittest
from pathlib import Path

from experiments.run_all import ci95, team_fields


class ResearchPipelineTests(unittest.TestCase):
    def test_confidence_interval_and_team_metadata(self):
        low, high = ci95([1, 0, 1, 1])
        self.assertLessEqual(low, 0.75)
        self.assertGreaterEqual(high, 0.75)
        self.assertIn("team_members", team_fields())
        self.assertIn("cms_ids", team_fields())

    def test_csv_round_trip_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "team_members"])
                writer.writeheader(); writer.writerow({"seed": 1, "team_members": team_fields()["team_members"]})
            with path.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["seed"], "1")
            self.assertTrue(rows[0]["team_members"])


if __name__ == "__main__":
    unittest.main()
