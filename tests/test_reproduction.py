"""Checks with historical predictions; these tests do not import/execute a model."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import warnings
from collections import defaultdict
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bohrium import machine_for
from deepmd_commands import test_command, train_command
from evaluate_remote import evaluate_split
from predict_remote import remap_types
from repro_common import (
    ROOT,
    load_json,
    metrics_module,
    parse_dptest,
    recipe_records,
    safe_extract,
    summarize,
    write_json,
)
from reproduce import prepare, verify
from run_remote import choose_best


def native_parser():
    """Run the pinned CLI parser without importing any backend or model code."""
    source = ROOT / "vendor/deepmd-kit-classification/deepmd/main.py"
    names = {
        "get_ll",
        "RawTextArgumentDefaultsHelpFormatter",
        "BackendOption",
        "DeprecateAction",
        "main_parser",
    }
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    scope = {
        "argparse": argparse,
        "logging": logging,
        "os": os,
        "textwrap": textwrap,
        "warnings": warnings,
        "defaultdict": defaultdict,
        "Any": Any,
        "BACKEND_TABLE": {"pt": "pytorch", "pytorch": "pytorch", "tensorflow": "tensorflow"},
        "available_model_names": lambda: [],
        "__version__": "parser-check-only",
    }
    exec(compile(tree, str(source), "exec"), scope)
    return scope["main_parser"]()


class ScientificReproduction(unittest.TestCase):
    def test_checkpoint_ranking_matches_original_selection(self):
        for name, spec in load_json(ROOT / "metadata/experiments.json").items():
            with (ROOT / "experiments" / name / "reference/checkpoint_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = [
                    {
                        **r,
                        **{key: float(r[key]) for key in ("macro_f1", "balanced_accuracy", "mae")},
                        "step": int(r["step"]),
                    }
                    for r in csv.DictReader(stream)
                    if r["split"] == "valid"
                ]
            self.assertEqual(len(rows), 40)
            selected = choose_best(rows)
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
                self.assertEqual(archive.getmember("bin/dp").mode, 0o755)
            self.assertFalse(
                any(n.endswith((".npy", ".pt", ".ckpt")) or n.startswith("data/") for n in names)
            )
            self.assertIn("deepmd-kit-classification/deepmd/pt/loss/classification.py", names)
            self.assertNotIn("best_checkpoint.pt", names)
            self.assertIn("run_remote.py", names)
            self.assertNotIn("train_multitask_remote.py", names)
            self.assertNotIn("runtime_checks.py", names)
            self.assertIn('export PATH="$PWD/bin:$PATH"', (directory / "run_job.sh").read_text())
            subprocess.run(["bash", "-n", str(directory / "run_job.sh")], check=True)

    def test_commands_parse_with_the_pinned_upstream_parser(self):
        parser = native_parser()
        train = parser.parse_args(train_command()[1:])
        self.assertEqual(train.backend, "pytorch")
        self.assertEqual(train.command, "train")
        self.assertEqual(train.INPUT, "input.json")
        self.assertEqual(train.finetune, "pretrained/model.ckpt-200000.pt")
        self.assertIsNone(train.restart)
        self.assertTrue(train.skip_neighbor_stat)
        test = parser.parse_args(
            test_command("best_checkpoint.pt", "data/phase/valid", "a b/out")[1:]
        )
        self.assertEqual(test.command, "test")
        self.assertEqual(test.model, "best_checkpoint.pt")
        self.assertEqual(test.system, "data/phase/valid")
        self.assertEqual(test.numb_test, 0)
        self.assertEqual(test.head, "synthesizability")
        self.assertEqual(test.detail_file, "a b/out")
        self.assertFalse(test.shuffle_test)
        self.assertFalse(test.atomic)
        self.assertIsNone(test.rand_seed)

    def test_native_test_adapter_replays_historical_output_without_a_model(self):
        experiment = ROOT / "experiments/three_natural"
        spec = load_json(ROOT / "metadata/experiments.json")["three_natural"]
        reference = experiment / "reference/valid"

        def replay(command, check):
            self.assertTrue(check)
            self.assertEqual(command[:3], ["dp", "--pt", "test"])
            prefix = command[command.index("-d") + 1]
            for suffix in (".classification.out", ".classification.metrics.json"):
                shutil.copyfile(reference / f"dptest{suffix}", f"{prefix}{suffix}")

        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            shutil.copytree(reference, "reference/valid")
            shutil.copyfile(experiment / "experiment_manifest.json", "experiment_manifest.json")
            with patch("deepmd_commands.subprocess.run", side_effect=replay) as call:
                result = evaluate_split("valid", Path("evaluation/valid"), spec)
            call.assert_called_once()
            self.assertEqual(result["structure"]["n"], 192)
            self.assertTrue(result["checks"]["reference_classes_identical"])
            self.assertEqual(result["checks"]["max_abs_logit_difference"], 0)
            self.assertEqual(
                result["structure"]["confusion_matrix"],
                load_json(reference / "metrics.json")["confusion_matrix"],
            )

    def test_submission_preserves_external_resources_and_redacts_credentials(self):
        class MachineStub:
            def __init__(self, data):
                self.data = data

            def serialize(self, if_empty_remote_profile=False):
                return {
                    "remote_profile": {} if if_empty_remote_profile else self.data["remote_profile"]
                }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / "machine.json"
            write_json(
                config,
                {
                    "remote_profile": {
                        "password": "not-a-real-credential",
                        "input_data": {
                            "job_resources": ["example-background-resource"],
                            "dataset_mounts": ["example-mount"],
                            "backward_files": ["stale-output"],
                        },
                    }
                },
            )
            write_json(directory / "task_group.json", {"job_group_id": 123})
            with patch.dict(
                sys.modules,
                {
                    "dpdispatcher": SimpleNamespace(
                        Machine=SimpleNamespace(load_from_dict=MachineStub)
                    )
                },
            ):
                machine = machine_for(config, directory)
            settings = machine.data["remote_profile"]["input_data"]
            self.assertEqual(settings["job_resources"], ["example-background-resource"])
            self.assertEqual(settings["dataset_mounts"], ["example-mount"])
            self.assertNotIn("backward_files", settings)
            self.assertTrue(settings["grouped"])
            self.assertEqual(machine.group_id, 123)
            self.assertEqual(machine.serialize()["remote_profile"], {})

    def test_pinned_artifacts(self):
        self.assertEqual(verify()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
