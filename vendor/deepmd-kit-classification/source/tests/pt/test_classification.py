# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch
import torch.nn.functional as F

from deepmd.pt.loss.classification import (
    ClassificationLoss,
)
from deepmd.pt.model.task.classification import (
    ClassificationFittingNet,
)


class TestClassificationLoss(unittest.TestCase):
    def test_cross_entropy_and_accuracy(self) -> None:
        logits = torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]], device="cpu")
        labels = torch.tensor([[0.0], [2.0]], device="cpu")
        loss_module = ClassificationLoss(num_classes=3, var_name="class")

        _, loss, metrics = loss_module(
            {},
            lambda: {"class": logits},
            {"class": labels},
            natoms=4,
        )

        expected = F.cross_entropy(logits, labels.squeeze(1).long())
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(metrics["accuracy"], torch.tensor(0.5, device="cpu"))

    def test_rejects_out_of_range_label(self) -> None:
        loss_module = ClassificationLoss(num_classes=2, var_name="class")
        with self.assertRaisesRegex(ValueError, "Class labels must be"):
            loss_module(
                {},
                lambda: {"class": torch.zeros((1, 2), device="cpu")},
                {"class": torch.tensor([[2]], device="cpu")},
                natoms=1,
            )

    def test_rejects_fractional_label(self) -> None:
        loss_module = ClassificationLoss(num_classes=3, var_name="class")
        with self.assertRaisesRegex(ValueError, "Class labels must be integers"):
            loss_module(
                {},
                lambda: {"class": torch.zeros((1, 3), device="cpu")},
                {"class": torch.tensor([[1.5]], device="cpu")},
                natoms=1,
            )

    def test_rejects_nonfinite_label(self) -> None:
        loss_module = ClassificationLoss(num_classes=3, var_name="class")
        with self.assertRaisesRegex(ValueError, "Class labels must be finite"):
            loss_module(
                {},
                lambda: {"class": torch.zeros((1, 3), device="cpu")},
                {"class": torch.tensor([[float("nan")]], device="cpu")},
                natoms=1,
            )

    def test_label_requirement(self) -> None:
        requirement = ClassificationLoss(3, "class").label_requirement[0]
        self.assertEqual(requirement.key, "class")
        self.assertEqual(requirement.ndof, 1)
        self.assertFalse(requirement.atomic)


class TestClassificationFitting(unittest.TestCase):
    def test_serialize_roundtrip(self) -> None:
        fitting = ClassificationFittingNet(
            ntypes=2,
            dim_descrpt=8,
            num_classes=3,
            label_name="phase",
            seed=7,
        )
        restored = ClassificationFittingNet.deserialize(fitting.serialize())

        self.assertEqual(restored.num_classes, 3)
        self.assertEqual(restored.var_name, "phase")
        output = restored.output_def()["phase"]
        self.assertEqual(output.shape, [3])
        self.assertTrue(output.intensive)


if __name__ == "__main__":
    unittest.main()
