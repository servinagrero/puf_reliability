import unittest

import numpy as np

from pufre.analysis.temperature import fit_temperature_graph_study


class TemperatureGraphStudyTest(unittest.TestCase):
    def test_room_regularization_is_reused_at_other_temperatures(self):
        rng = np.random.default_rng(7)
        reference = rng.integers(0, 2, size=(4, 4))
        responses = {
            temperature: np.bitwise_xor(
                reference[:, np.newaxis, :],
                rng.binomial(1, probability, size=(4, 20, 4)),
            )
            for temperature, probability in [(25, 0.1), (80, 0.2)]
        }
        value1 = {
            temperature: response.astype(float)
            for temperature, response in responses.items()
        }
        value2 = {
            temperature: 1.0 - response
            for temperature, response in responses.items()
        }

        result = fit_temperature_graph_study(
            reference=reference,
            responses_by_temperature=responses,
            value1_by_temperature=value1,
            value2_by_temperature=value2,
            baseline_temperature=25,
            graph_C=None,
            max_tuning_bits=1,
            cv_splits=2,
            max_iter=100,
        )

        self.assertEqual(result.models[25].regularization, result.regularization)
        self.assertEqual(result.models[80].regularization, result.regularization)


if __name__ == "__main__":
    unittest.main()
