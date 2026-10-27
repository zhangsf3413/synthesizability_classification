# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pure NumPy checks; these tests never load or run a model."""

import unittest

import numpy as np

from deepmd.utils.classification import (
    classification_metrics,
    classification_probabilities,
    classification_statistics,
    merge_classification_statistics,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_stable_softmax_and_cross_entropy(self) -> None:
        logits = np.array([[1001.0, 1000.0], [-1001.0, -1000.0]])
        labels = np.array([[0], [1]], dtype=np.int32)
        probability = classification_probabilities(logits)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        np.testing.assert_array_equal(probability.argmax(axis=1), logits.argmax(axis=1))
        metrics = classification_metrics(classification_statistics(logits, labels))
        self.assertAlmostEqual(metrics["ce"], np.log1p(np.exp(-1)))
        self.assertAlmostEqual(metrics["mae"], 1 / (1 + np.exp(1)))
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_merge_unequal_system_sizes_matches_concatenation(self) -> None:
        logits = np.array([[2, 0], [0, 1], [0, 2], [1, 0], [0, 3]], dtype=float)
        labels = np.array([0, 0, 1, 1, 1])
        combined = classification_metrics(classification_statistics(logits, labels))
        merged = classification_metrics(
            merge_classification_statistics(
                [
                    classification_statistics(logits[:1], labels[:1]),
                    classification_statistics(logits[1:], labels[1:]),
                ]
            )
        )
        for key, value in combined.items():
            np.testing.assert_allclose(merged[key], value, rtol=1e-14, atol=1e-14)
        self.assertEqual(merged["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertAlmostEqual(merged["macro_f1"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(merged["balanced_accuracy"], (0.5 + 2 / 3) / 2)
        self.assertEqual(merged["false_positive_rate"], 0.5)

    def test_all_positive_predictions_are_not_high_macro_f1(self) -> None:
        labels = np.array([0, 1, 1, 1])
        metrics = classification_metrics(
            classification_statistics(np.tile([0.0, 2.0], (4, 1)), labels)
        )
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 3 / 7)
        self.assertEqual(metrics["false_positive_rate"], 1.0)

    def test_tie_predicts_first_class(self) -> None:
        metrics = classification_metrics(
            classification_statistics(np.zeros((2, 2)), np.array([0, 1]))
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 0]])

    def test_rejects_bad_label_shapes_values_and_nonfinite_logits(self) -> None:
        logits = np.zeros((2, 2))
        for labels in (
            np.eye(2),
            np.array([0.5, 1]),
            np.array([0, 2]),
            np.array([0, np.nan]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                classification_statistics(logits, labels)
        with self.assertRaises(ValueError):
            classification_probabilities(np.array([[np.inf, 0.0]]))

    def test_multiclass_metrics(self) -> None:
        metrics = classification_metrics(
            classification_statistics(np.eye(3) * 3, np.arange(3))
        )
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["support_per_class"], [1, 1, 1])
        self.assertNotIn("false_positive_rate", metrics)
