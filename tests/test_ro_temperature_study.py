import unittest

import polars as pl

from scripts.run_ro_temperature_study import replace_graph_metrics


class RoTemperatureStudyTest(unittest.TestCase):
    def test_stable_graph_metrics_replace_raw_metrics(self):
        raw = pl.DataFrame(
            {
                "temperature": [25],
                "bit_id": [0],
                "physical_bit_id": [10],
                "flip_rate": [0.1],
                "out_strength": [9.0],
                "stability": [0.8],
            }
        )
        stable = pl.DataFrame(
            {
                "temperature": [25],
                "bit_id": [0],
                "physical_bit_id": [10],
                "flip_rate": [0.1],
                "out_strength": [2.0],
                "graph_role": ["source"],
            }
        )

        result = replace_graph_metrics(raw, stable)

        self.assertEqual(result["out_strength"][0], 2.0)
        self.assertEqual(result["graph_role"][0], "source")
        self.assertEqual(result["stability"][0], 0.8)


if __name__ == "__main__":
    unittest.main()
