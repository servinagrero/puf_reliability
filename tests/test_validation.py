import unittest

import numpy as np

from pufre.data import PUFObservations
from pufre.error_metrics.validation import validate_error_graph_split


class ErrorGraphValidationTest(unittest.TestCase):
    def test_split_reports_paired_scores_for_each_held_out_device(self):
        rng = np.random.default_rng(17)
        reference = np.zeros((8, 4), dtype=np.uint8)
        responses = rng.binomial(
            1,
            np.array([0.05, 0.10, 0.20, 0.30]),
            size=(8, 12, 4),
        ).astype(np.uint8)
        observations = PUFObservations.from_binary(responses, reference)

        independent, graph, trial, devices = validate_error_graph_split(
            observations,
            family="test",
            split=0,
            C=0.1,
            random_state=3,
            max_iter=100,
            n_jobs=1,
        )

        self.assertEqual(trial.n_test_devices, 2)
        self.assertEqual(len(devices), trial.n_test_devices)
        self.assertAlmostEqual(
            trial.log_loss_gain,
            independent.log_loss - graph.log_loss,
        )
        self.assertTrue(
            all(
                result.log_loss_gain
                == result.independent_log_loss - result.error_graph_log_loss
                for result in devices
            )
        )


if __name__ == "__main__":
    unittest.main()
