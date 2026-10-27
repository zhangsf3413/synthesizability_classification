"""Native dp test of a fixed checkpoint; never run on a local workstation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from deepmd_commands import run, test_command
from repro_common import load_json, parse_dptest, save_reports, write_json


def evaluate_split(split, output, spec, check_reference=True, checkpoint="best_checkpoint.pt"):
    output.mkdir(parents=True)
    root = Path("heldout" if split == "test" else "data") / spec["data_name"]
    run(test_command(checkpoint, root / split, output / "dptest"))
    records = parse_dptest(output / "dptest.classification.out", spec["classes"])
    expected = load_json("experiment_manifest.json")["dataset_counts"][split]
    if len(records) != expected["structures"]:
        raise ValueError("dp test frame count mismatch")
    structure, recipe = save_reports(output, records)
    native = load_json(output / "dptest.classification.metrics.json")
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "mae", "mse", "rmse", "ce"):
        if not np.isclose(native[key], structure[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Native metric mismatch: {key}")
    check = {"split": split, "structures": len(records), "native_metrics_verified": True}
    if check_reference:
        reference = parse_dptest(
            Path("reference") / split / "dptest.classification.out", spec["classes"]
        )
        keys = [(r["recipe_id"], r["frame"], r["true"]) for r in records]
        reference_keys = [(r["recipe_id"], r["frame"], r["true"]) for r in reference]
        if keys != reference_keys:
            raise ValueError("Reference structure/label order mismatch")
        actual = np.asarray([r["scores"] for r in records])
        original = np.asarray([r["scores"] for r in reference])
        # Minor GPU/library nondeterminism is allowed in scores; classes must agree.
        if not np.allclose(actual, original, rtol=1e-4, atol=2e-3):
            raise ValueError("Scores differ from the historical checkpoint reference")
        if not np.array_equal(actual.argmax(axis=1), original.argmax(axis=1)):
            raise ValueError("Predicted classes differ from historical outputs")
        check.update(
            reference_classes_identical=True,
            max_abs_logit_difference=float(np.abs(actual - original).max()),
        )
    write_json(output / "reproduction_checks.json", check)
    return {"structure": structure, "recipe": recipe, "checks": check}
