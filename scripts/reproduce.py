#!/usr/bin/env python3
"""Verify code/results or prepare Bohrium jobs; never package training data."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from repro_common import (
    ROOT,
    load_json,
    parse_dptest,
    save_reports,
    sha256,
    write_csv,
    write_json,
)


def catalog():
    return load_json(ROOT / "metadata/experiments.json")


def verify():
    source = ROOT / "vendor/deepmd-kit-classification"
    hashes = load_json(ROOT / "upstream/runtime_source_sha256.json")
    for name, expected in hashes.items():
        if sha256(source / name) != expected:
            raise ValueError(f"Runtime source changed: {name}")
    lock = load_json(ROOT / "upstream/lock.json")
    if sha256(ROOT / "upstream" / lock["patch"]) != lock["patch_sha256"]:
        raise ValueError("Upstream patch changed")
    for name, spec in catalog().items():
        exp = ROOT / "experiments" / name
        if sha256(exp / "input.json") != spec["input_sha256"]:
            raise ValueError(f"Original config changed: {name}")
        if sha256(exp / "best_checkpoint.pt") != spec["model_sha256"]:
            raise ValueError(f"Selected model changed: {name}")
        config = load_json(exp / "input.json")
        training = config["training"]
        if training["numb_steps"] != 100000 or training["save_freq"] != 5000:
            raise ValueError("Unexpected training length/checkpoint cadence")
        for data_key in ("training_data", "validation_data"):
            path = training["data_dict"]["synthesizability"][data_key]["systems"]
            if Path(path).is_absolute() or Path(path).parts[0] != "data":
                raise ValueError("Expected relative dataset paths")
        manifest = load_json(exp / "experiment_manifest.json")
        import csv

        with (ROOT / "metadata" / spec["kind"] / "development_index.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        if {r["split"] for r in rows} != {"train", "valid"}:
            raise ValueError("Development index contains test data")
        index = ROOT / "metadata" / spec["kind"] / "development_index.csv"
        if sha256(index) != manifest["development_index_sha256"]:
            raise ValueError("Historical development index changed")
        training_rows = [r for r in rows if r["split"] == "train"]
        data = training["data_dict"]["synthesizability"]["training_data"]
        if spec["sampling"] == "balanced":
            recipe_class = {r["recipe_id"]: int(r["synthesizable_id"]) for r in training_rows}
            counts = np.bincount(list(recipe_class.values()), minlength=2)
            expected = [0.5 / counts[recipe_class[r]] for r in sorted(recipe_class)]
            if not np.allclose(data["sys_probs"], expected, rtol=0, atol=1e-15):
                raise ValueError("sys_probs order/values mismatch")
        elif "sys_probs" in data or data["auto_prob"] != "prob_sys_size":
            raise ValueError("Natural sampling config mismatch")
    result = {
        "status": "passed",
        "runtime_source_files": len(hashes),
        "experiments": len(catalog()),
        "upstream_patch_files": len(lock["changed_files"]),
        "training_data_bundled": False,
        "local_model_execution": False,
        "scope": "source/patch/config/model hashes, relative paths, split indices and sampling order",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def report(output):
    rows = []
    for name, spec in catalog().items():
        for split in ("train", "valid", "test"):
            reference = ROOT / "experiments" / name / "reference" / split
            records = parse_dptest(reference / "dptest.classification.out", spec["classes"])
            structure, recipe = save_reports(output / name / split, records)
            native = load_json(reference / "metrics.json")
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "mae", "mse", "rmse", "ce"):
                if not np.isclose(structure[key], native[key], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"Saved native metric differs: {name}/{split}/{key}")
            for unit, result in (("structure", structure), ("recipe", recipe)):
                rows.append(
                    {
                        "experiment": name,
                        "unit": unit,
                        "split": split,
                        "selected_step": spec["selected_step"],
                        **{
                            k: result[k]
                            for k in (
                                "n",
                                "accuracy",
                                "macro_f1",
                                "balanced_accuracy",
                                "binary_accuracy",
                                "binary_macro_f1",
                                "false_positive_rate",
                                "mae",
                                "rmse",
                            )
                        },
                    }
                )
    write_csv(output / "summary.csv", rows)
    write_json(
        output / "verification.json",
        {"status": "passed", "native_outputs_recomputed": 9, "local_model_execution": False},
    )
    print(f"Reports written to {output}")


def add_file(source, target, archive):
    info = tarfile.TarInfo(str(target))
    content = Path(source).read_bytes()
    info.size = len(content)
    info.mode = 0o755 if str(target) == "bin/dp" else 0o644
    archive.addfile(info, io.BytesIO(content))


def prepare(experiment, mode, output, dataset, background, pretrained):
    spec = {
        **catalog()[experiment],
        "experiment": experiment,
        "mode": mode,
        "external_dataset_root": dataset,
        "external_background_root": background,
        "external_pretrained": pretrained,
    }
    if output.exists():
        raise ValueError(f"Output exists; use a new run directory: {output}")
    if mode == "train" and (not background or not pretrained):
        raise ValueError("Training requires remote --background-root and --pretrained paths")
    output.mkdir(parents=True)
    exp = ROOT / "experiments" / experiment
    with tarfile.open(output / "payload.tar.gz", "w:gz") as archive:
        for file in sorted((ROOT / "vendor/deepmd-kit-classification").rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts:
                add_file(
                    file,
                    Path("deepmd-kit-classification")
                    / file.relative_to(ROOT / "vendor/deepmd-kit-classification"),
                    archive,
                )
        for file in sorted((ROOT / "workflow").iterdir()):
            if file.is_file():
                add_file(file, Path("bin") / file.name, archive)
        for name in ("input.json", "experiment_manifest.json"):
            add_file(exp / name, name, archive)
        for name in (
            "repro_common.py",
            "stage_data.py",
            "evaluate_remote.py",
            "run_remote.py",
            "deepmd_commands.py",
            "predict_remote.py",
        ):
            add_file(ROOT / "scripts" / name, name, archive)
        add_file(
            ROOT / "upstream/overlay/deepmd/utils/classification.py",
            "classification_metrics.py",
            archive,
        )
        for name in ("development_index.csv", "holdout_index.csv", "dataset_index.csv"):
            add_file(ROOT / "metadata" / spec["kind"] / name, Path("metadata") / name, archive)
        if mode != "train":
            add_file(exp / "best_checkpoint.pt", "best_checkpoint.pt", archive)
            add_file(exp / "reference/best_checkpoint.json", "fixed_checkpoint.json", archive)
            for split in ("train", "valid", "test"):
                add_file(
                    exp / "reference" / split / "dptest.classification.out",
                    f"reference/{split}/dptest.classification.out",
                    archive,
                )
        encoded = (json.dumps(spec, indent=2) + "\n").encode()
        info = tarfile.TarInfo("reproduction_spec.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    script = """#!/usr/bin/env bash
set -Eeuo pipefail
exec 2>&1
export BOHRIUM_MULTITASK_REMOTE=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 NUM_WORKERS=0
export OMP_NUM_THREADS=8 DP_INTRA_OP_PARALLELISM_THREADS=8 DP_INTER_OP_PARALLELISM_THREADS=1
export PYTHONUNBUFFERED=1
collect() {
    rc=$?
    trap - EXIT
    set +e
    mkdir -p repro_outputs
    for f in bohrium_reproduction.log bohrium_reproduction.err reproduction_spec.json input.json \\
             environment.json pip-freeze.txt lcurve.out best_checkpoint.pt; do
        [[ ! -f "$f" ]] || cp "$f" repro_outputs/
    done
    for d in evaluation evaluation_reproduced predictions; do
        [[ ! -d "$d" ]] || cp -a "$d" repro_outputs/
    done
    for f in model*.pt; do [[ ! -f "$f" ]] || cp "$f" repro_outputs/; done
    printf '{"exit_code": %s}\\n' "$rc" > repro_outputs/exit_status.json
    tar -czf reproduction_result.tar.gz repro_outputs
    exit "$rc"
}
trap collect EXIT
tar -xzf payload.tar.gz
export PATH="$PWD/bin:$PATH"
python -m pip freeze > pip-freeze.txt
python -u run_remote.py
"""
    (output / "run_job.sh").write_text(script)
    forward = ["payload.tar.gz", "run_job.sh"]
    backward = [
        "bohrium_reproduction.log",
        "bohrium_reproduction.err",
        "reproduction_result.tar.gz",
    ]
    if mode == "train":
        backward.append("lcurve.out")
    write_json(
        output / "submission_spec.json",
        {
            "experiment": experiment,
            "mode": mode,
            "forward_files": forward,
            "backward_files": backward,
            "log_file": "bohrium_reproduction.log",
            "training_data_in_payload": False,
            "forward_sha256": {f: sha256(output / f) for f in forward},
        },
    )
    print(f"Prepared {mode}, code/config/models only: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify")
    rep = sub.add_parser("report")
    rep.add_argument("--output", type=Path, default=ROOT / "runs/reports")
    prep = sub.add_parser("prepare")
    prep.add_argument("--experiment", choices=list(catalog()), required=True)
    prep.add_argument("--mode", choices=("train", "evaluate", "smoke", "predict"), required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--dataset-root", required=True, help="Dataset path visible inside the Bohrium job"
    )
    prep.add_argument(
        "--background-root", help="Mixed background root (train/valid), visible in Bohrium"
    )
    prep.add_argument("--pretrained", help="Original DPA4 checkpoint path visible in Bohrium")
    args = parser.parse_args()
    if args.action == "verify":
        verify()
    elif args.action == "report":
        report(args.output.resolve())
    else:
        prepare(
            args.experiment,
            args.mode,
            args.output.resolve(),
            args.dataset_root,
            args.background_root,
            args.pretrained,
        )


if __name__ == "__main__":
    main()
