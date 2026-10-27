"""Fresh 100k native classification/CE multitask training on Bohrium."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from repro_common import load_json, sha256, write_json


def main():
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise SystemExit("Training is restricted to Bohrium")
    from runtime_checks import initialize

    initialize()
    from stage_data import stage, stage_background

    stage()
    stage_background()
    spec = load_json("reproduction_spec.json")
    from validate_classification import validate

    write_json("runtime_preparation_checks.json", validate())
    if Path("best_checkpoint.pt").exists() or list(Path(".").glob("model*.pt")):
        raise RuntimeError("Fresh training must not contain a trained classification checkpoint")
    import train_multitask_remote

    sys.argv = ["train_multitask_remote.py", "--input", "input.json", "--gpu", "V100"]
    train_multitask_remote.main()
    from select_checkpoints_remote import select

    selection = select()
    model_hash = sha256("best_checkpoint.pt")
    write_json(
        "evaluation/fixed_selection_before_test.json",
        {"model_sha256": model_hash, "selection": selection, "test_used_for_selection": False},
    )
    stage(("test",))
    from evaluate_remote import evaluate_split

    result = evaluate_split("test", Path("evaluation/test"), spec, check_reference=False)
    if sha256("best_checkpoint.pt") != model_hash:
        raise ValueError("Selected model changed during test evaluation")
    write_json(
        "evaluation/final_metrics.json",
        {
            "status": "passed",
            "selection": selection,
            "model_sha256": model_hash,
            "test_used_for_selection": False,
            "test_results": result,
            "original_pretrained_restart": True,
        },
    )


if __name__ == "__main__":
    main()
