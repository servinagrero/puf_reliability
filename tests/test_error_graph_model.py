import unittest

import numpy as np

from pufre.graph.bootstrap import (
    EdgeStability,
    _permute_columns_within_groups,
    edge_stability_frame,
)
from pufre.graph.comparison import (
    graph_bootstrap_metric_frame,
    paired_graph_distance_frame,
)
from pufre.graph.model import DirectedErrorGraphModel, NodewiseLogisticGraph


class DirectedErrorGraphModelTest(unittest.TestCase):
    def test_matrix_orientation_and_probability(self):
        model = DirectedErrorGraphModel(
            coefficients=np.array([[0.0, 2.0], [-1.0, 0.0]]),
            intercepts=np.zeros(2),
            regularization=0.1,
        )
        errors = np.array([[1.0, -1.0]])
        probability = model.conditional_flip_probability(errors)

        self.assertEqual(model.adjacency[0, 1], 2.0)
        self.assertTrue(np.all(np.diag(model.adjacency) == 0.0))
        np.testing.assert_allclose(
            probability,
            np.array([[0.88079708, 0.73105858]]),
        )

    def test_constant_target_has_no_incoming_edges(self):
        errors = np.array(
            [
                [1, 1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [1, -1, -1],
            ],
            dtype=float,
        )
        model = NodewiseLogisticGraph(C=0.1, max_iter=100, n_jobs=2).fit(
            errors
        ).model_

        self.assertTrue(np.all(model.adjacency[0] == 0.0))
        self.assertTrue(model.fit_details[0]["is_constant"])

    def test_grouped_cv_selects_family_regularization(self):
        rng = np.random.default_rng(11)
        errors = rng.choice([-1.0, 1.0], size=(24, 4))
        groups = np.repeat(np.arange(6), 4)
        candidates = np.array([0.01, 0.1])

        model = NodewiseLogisticGraph(
            C=None,
            Cs=candidates,
            max_tuning_bits=2,
            cv_splits=2,
            max_iter=100,
        ).fit(errors, groups=groups).model_

        self.assertIn(model.regularization, candidates)

    def test_null_permutation_preserves_group_bit_counts(self):
        errors = np.array(
            [
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1],
                [1, 1, 1],
                [-1, -1, -1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ]
        )
        groups = np.repeat([0, 1], 4)
        permuted = _permute_columns_within_groups(
            errors,
            groups,
            np.random.default_rng(7),
        )

        for group in np.unique(groups):
            original = errors[groups == group]
            surrogate = permuted[groups == group]
            self.assertTrue(
                np.array_equal(
                    np.sort(original, axis=0),
                    np.sort(surrogate, axis=0),
                )
            )
        self.assertFalse(np.array_equal(errors, permuted))

    def test_empty_edge_stability_frame_retains_csv_schema(self):
        zeros = np.zeros((2, 2))
        stability = EdgeStability(zeros, zeros, zeros, zeros, zeros)

        frame = edge_stability_frame(stability, min_probability=0.5)

        self.assertTrue(frame.is_empty())
        self.assertEqual(
            frame.columns,
            [
                "target",
                "predictor",
                "selection_probability",
                "mean_weight",
                "std_weight",
                "mean_abs_weight",
                "sign_stability",
            ],
        )

    def test_bootstrap_graph_metrics_and_distances(self):
        samples = np.array(
            [
                [[0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, 0.0]],
            ]
        )
        metrics = graph_bootstrap_metric_frame(samples)
        self.assertEqual(metrics["directed_edges"].to_list(), [1, 1])
        np.testing.assert_allclose(
            paired_graph_distance_frame(samples, samples).select(
                "normalized_frobenius_distance",
                "cosine_distance",
                "edge_jaccard_distance",
            ).to_numpy(),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
