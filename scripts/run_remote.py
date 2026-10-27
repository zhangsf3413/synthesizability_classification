"""Stage external labels, use native dp train/test, and select on full validation."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
from pathlib import Path

from deepmd_commands import run, train_command
from evaluate_remote import evaluate_split
from repro_common import load_json, sha256, write_csv, write_json
from stage_data import stage, stage_background


def choose_best(rows):
    return max(rows, key=lambda r: (r["macro_f1"], r["balanced_accuracy"], -r["mae"], -r["step"]))


def select_checkpoint(spec):
    training = load_json("input.json")["training"]
    prefix = Path(training["save_ckpt"])
    # Pinned DeepMD inserts _ema before the prefix's final suffix.
    ema_prefix = prefix.with_name(f"{prefix.stem}_ema{prefix.suffix}")
    rows = []
    for step in range(training["save_freq"], training["numb_steps"] + 1, training["save_freq"]):
        for family, name in (("ordinary", prefix), ("ema", ema_prefix)):
            checkpoint = Path(f"{name}-{step}.pt")
            directory = Path("evaluation") / f"{family}_step_{step:06d}_valid"
            metrics = evaluate_split("valid", directory, spec, False, checkpoint)["structure"]
            rows.append(
                {
                    "step": step,
                    "family": family,
                    "checkpoint": checkpoint.name,
                    "split": "valid",
                    **{
                        key: metrics[key]
                        for key in (
                            "n",
                            "accuracy",
                            "macro_f1",
                            "balanced_accuracy",
                            "mae",
                            "ce",
                            "binary_accuracy",
                            "binary_macro_f1",
                            "false_positive_rate",
                        )
                    },
                }
            )
            write_csv("evaluation/checkpoint_metrics.csv", rows)
    best = choose_best(rows)
    shutil.copyfile(best["checkpoint"], "best_checkpoint.pt")
    selection = {
        "selection_complete": True,
        "selection_split": "valid",
        "evaluation_unit": "structure",
        "primary_metric": "macro_f1",
        "test_used": False,
        "best": best,
        "tie_breakers": ["balanced_accuracy", "lower_mae", "earlier_step"],
    }
    write_json("evaluation/best_checkpoint.json", selection)
    return selection


def main():
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise SystemExit("Model execution is restricted to Bohrium")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for this Bohrium job")
    write_json(
        "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "deepmd": importlib.metadata.version("deepmd-kit"),
            "entrypoint": "native dp CLI",
        },
    )
    spec = load_json("reproduction_spec.json")
    if spec["mode"] == "predict":
        from predict_remote import main as predict

        predict()
        return
    stage()
    output = Path("evaluation" if spec["mode"] == "train" else "evaluation_reproduced")
    if spec["mode"] == "train":
        stage_background()
        run(train_command())
        selection = select_checkpoint(spec)
    else:
        if sha256("best_checkpoint.pt") != spec["model_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        selection = load_json("fixed_checkpoint.json")
    if (
        not selection["selection_complete"]
        or selection["selection_split"] != "valid"
        or selection["test_used"]
    ):
        raise ValueError("Select the checkpoint on validation before testing")
    model_hash = sha256("best_checkpoint.pt")
    splits = ("valid",) if spec["mode"] == "smoke" else ("train", "valid", "test")
    results = {}
    for split in splits:
        if split == "test":
            write_json(
                output / "fixed_selection_before_test.json",
                {
                    "model_sha256": model_hash,
                    "selection": selection,
                    "test_used_for_selection": False,
                },
            )
            stage(("test",))
        results[split] = evaluate_split(split, output / split, spec, spec["mode"] != "train")
    write_json(
        output / "summary.json",
        {
            "status": "passed",
            "model_sha256": model_hash,
            "selection": selection,
            "training_executed": spec["mode"] == "train",
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
