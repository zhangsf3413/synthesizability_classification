"""Native dp test of a fixed checkpoint; never run on a local workstation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from repro_common import load_json, parse_dptest, save_reports, sha256, write_json


def evaluate_split(split, output, spec, check_reference=True, checkpoint="best_checkpoint.pt"):
    from deepmd.entrypoints.test import test as dp_test

    output.mkdir(parents=True)
    kwargs = {}
    if split == "train":
        kwargs["train_json"] = "input.json"
    elif split == "valid":
        kwargs["valid_json"] = "input.json"
    else:
        systems = sorted((Path("heldout") / spec["data_name"] / "test").glob("*/type.raw"))
        system_list = output / "systems.txt"
        system_list.write_text("\n".join(str(p.parent) for p in systems) + "\n")
        kwargs["datafile"] = str(system_list)
    dp_test(
        model=str(checkpoint),
        system=None,
        datafile=kwargs.pop("datafile", None),
        numb_test=0,
        rand_seed=42,
        shuffle_test=False,
        detail_file=str(output / "dptest"),
        atomic=False,
        head="synthesizability",
        **kwargs,
    )
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


def main():
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise SystemExit("Model execution is restricted to Bohrium")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "evaluate"), required=True)
    args = parser.parse_args()
    spec = load_json("reproduction_spec.json")
    if sha256("best_checkpoint.pt") != spec["model_sha256"]:
        raise ValueError("Requested selected checkpoint hash mismatch")
    selection = load_json("fixed_checkpoint.json")
    if (
        not selection["selection_complete"]
        or selection["selection_split"] != "valid"
        or selection["test_used"]
    ):
        raise ValueError("Checkpoint must be fixed using validation alone")
    from runtime_checks import initialize

    initialize()
    from stage_data import stage

    stage()
    splits = ("valid",) if args.mode == "smoke" else ("train", "valid", "test")
    results = {}
    for split in splits:
        if split == "test":
            write_json(
                "evaluation_reproduced/fixed_selection_before_test.json",
                {
                    "model_sha256": sha256("best_checkpoint.pt"),
                    "selection": selection,
                    "test_used_for_selection": False,
                },
            )
            stage(("test",))
        results[split] = evaluate_split(split, Path("evaluation_reproduced") / split, spec)
    write_json(
        "evaluation_reproduced/summary.json",
        {
            "status": "passed",
            "mode": args.mode,
            "model_sha256": sha256("best_checkpoint.pt"),
            "training_executed": False,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
