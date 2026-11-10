import unittest

import numpy as np

from pufre.error_metrics.independence import (
    conditional_independence_test,
    holm_adjusted_p_values,
    permute_columns_within_devices,
)


class ConditionalIndependenceTest(unittest.TestCase):
    def test_holm_adjustment_preserves_order_and_monotonicity(self):
        adjusted = holm_adjusted_p_values([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_permutation_preserves_device_bit_counts(self):
        errors = np.random.default_rng(1).integers(0, 2, size=(4, 20, 8))
        permuted = permute_columns_within_devices(
            errors,
            np.random.default_rng(2),
        )
        np.testing.assert_array_equal(
            errors.sum(axis=1),
            permuted.sum(axis=1),
        )

    def test_detects_duplicated_error_bits(self):
        errors = np.random.default_rng(3).integers(0, 2, size=(12, 40, 8))
        errors[:, :, 1] = errors[:, :, 0]
        result = conditional_independence_test(
            errors,
            n_permutations=39,
            random_state=4,
        )
        correlation = result.summary_rows()[0]
        self.assertGreater(correlation["observed_to_null_ratio"], 1.0)
        self.assertLessEqual(correlation["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
