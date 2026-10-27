"""Full-validation selection for native CE classifiers; no alternate training method."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

from repro_common import load_json, write_csv, write_json


def select():
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise RuntimeError("Checkpoint evaluation is restricted to Bohrium")
    from deepmd.pt.train.ema import get_ema_checkpoint_prefix
    from evaluate_remote import evaluate_split

    spec = load_json("reproduction_spec.json")
    training = load_json("input.json")["training"]
    prefixes = [
        ("ordinary", training["save_ckpt"]),
        ("ema", get_ema_checkpoint_prefix(training["save_ckpt"])),
    ]
    rows = []
    for step in range(training["save_freq"], training["numb_steps"] + 1, training["save_freq"]):
        for family, prefix in prefixes:
            checkpoint = Path(f"{prefix}-{step}.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            directory = Path("evaluation") / f"{family}_step_{step:06d}_valid"
            metrics = evaluate_split(
                "valid", directory, spec, check_reference=False, checkpoint=checkpoint
            )["structure"]
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
                            "rmse",
                            "ce",
                            "binary_accuracy",
                            "binary_macro_f1",
                            "false_positive_rate",
                        )
                    },
                }
            )
            write_csv("evaluation/checkpoint_metrics.csv", rows)
            import torch

            gc.collect()
            torch.cuda.empty_cache()
    best = max(
        rows, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], -row["mae"], -row["step"])
    )
    shutil.copy2(best["checkpoint"], "best_checkpoint.pt")
    directory = Path("evaluation") / f"{best['family']}_step_{best['step']:06d}_train"
    train = evaluate_split("train", directory, spec, check_reference=False)["structure"]
    selection = {
        "selection_complete": True,
        "selection_split": "valid",
        "evaluation_unit": "structure",
        "primary_metric": "macro_f1",
        "tie_breakers": ["balanced_accuracy", "lower_mae", "earlier_step"],
        "class_order": spec["class_order"],
        "test_used": False,
        "best": best,
        "selected_checkpoint_train_metrics": train,
        "selected_model_file": "best_checkpoint.pt",
    }
    write_json("evaluation/best_checkpoint.json", selection)
    return selection
