"""Checks with historical predictions; these tests do not import/execute a model."""

from __future__ import annotations

import csv
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from predict_remote import remap_types
from repro_common import (
    ROOT,
    load_json,
    metrics_module,
    parse_dptest,
    recipe_records,
    safe_extract,
    summarize,
)
from reproduce import prepare, verify


class ScientificReproduction(unittest.TestCase):
    def test_checkpoint_ranking_matches_original_selection(self):
        for name, spec in load_json(ROOT / "metadata/experiments.json").items():
            with (ROOT / "experiments" / name / "reference/checkpoint_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = [r for r in csv.DictReader(stream) if r["split"] == "valid"]
            self.assertEqual(len(rows), 40)
            selected = max(
                rows,
                key=lambda row: (
                    float(row["macro_f1"]),
                    float(row["balanced_accuracy"]),
                    -float(row["mae"]),
                    -int(row["step"]),
                ),
            )
            self.assertEqual(int(selected["step"]), spec["selected_step"])
            self.assertEqual(selected["family"], spec["selected_family"])

    def test_all_historical_native_outputs(self):
        for name, spec in load_json(ROOT / "metadata/experiments.json").items():
            for split in ("train", "valid", "test"):
                directory = ROOT / "experiments" / name / "reference" / split
                records = parse_dptest(directory / "dptest.classification.out", spec["classes"])
                metrics = summarize(records)
                native = load_json(directory / "metrics.json")
                self.assertEqual(metrics["confusion_matrix"], native["confusion_matrix"])
                for key in (
                    "ce",
                    "accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "mae",
                    "mse",
                    "rmse",
                    "false_positive_rate",
                ):
                    self.assertAlmostEqual(metrics[key], native[key], places=11)

    def test_three_class_merge_is_argmax_before_merge(self):
        # P(class 1 or 2) > 0.5 does not imply that either is the largest class.
        row = {"recipe_id": "example", "true": 0, "scores": np.log([0.4, 0.3, 0.3]).tolist()}
        result = summarize([row])
        self.assertEqual(result["binary_confusion_matrix"], [[1, 0], [0, 0]])

    def test_recipe_aggregation_uses_mean_logits(self):
        records = [
            {"recipe_id": "a", "true": 1, "scores": scores}
            for scores in ([0.0, 1.0], [0.0, 1.0], [10.0, 0.0])
        ]
        averaged = recipe_records(records)
        self.assertEqual(np.argmax(averaged[0]["scores"]), 0)
        self.assertEqual(sum(np.argmax(r["scores"]) == 1 for r in records), 2)

    def test_invalid_labels_and_stable_softmax(self):
        module = metrics_module()
        for label in ([-1], [2], [0.5], [np.nan]):
            with self.assertRaises(ValueError):
                module.classification_statistics(np.asarray([[1000, 999]]), np.asarray(label))
        probability = module.classification_probabilities(np.asarray([[1000.0, 999.0]]))
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        self.assertEqual(probability.argmax(), 0)

    def test_element_remapping_is_by_symbol(self):
        np.testing.assert_array_equal(
            remap_types(np.asarray([[0, 1, 0]]), ["O", "Sr"], ["H", "Sr", "O"]), [[2, 1, 2]]
        )
        with self.assertRaises(ValueError):
            remap_types(np.asarray([0]), ["Unknown"], ["H"])

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("../escaped")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                safe_extract(archive_path, Path(temporary) / "output")

    def test_training_payload_excludes_data_and_trained_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "job"
            prepare(
                "three_natural",
                "train",
                directory,
                "/mounted/dataset",
                "/mounted/background",
                "/mounted/pretrained.pt",
            )
            with tarfile.open(directory / "payload.tar.gz") as archive:
                names = archive.getnames()
            self.assertFalse(
                any(n.endswith((".npy", ".pt", ".ckpt")) or n.startswith("data/") for n in names)
            )
            self.assertIn("deepmd-kit-classification/deepmd/pt/loss/classification.py", names)
            self.assertNotIn("best_checkpoint.pt", names)

    def test_pinned_artifacts(self):
        self.assertEqual(verify()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
