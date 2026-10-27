"""Shared packaging and pure NumPy reporting; this module never imports a model."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_extract(archive_path, destination, prefixes=None):
    """Extract only regular files/directories; reject links and path traversal."""
    destination = Path(destination).resolve()
    with tarfile.open(archive_path) as archive:
        selected = []
        for member in archive:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
            if prefixes is None or any(member.name.startswith(p) for p in prefixes):
                selected.append(member)
        archive.extractall(destination, members=selected, filter="data")


def metrics_module():
    """Load the exact historical NumPy implementation without importing DeepMD."""
    path = ROOT / "upstream/overlay/deepmd/utils/classification.py"
    if not path.is_file():  # Prepared remote jobs carry the same file here.
        path = Path(__file__).with_name("classification_metrics.py")
    spec = importlib.util.spec_from_file_location("historical_classification_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_dptest(path, classes):
    records = []
    system = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            system = line.lstrip("# ").split(":", 1)[0]
            continue
        values = np.asarray([float(v) for v in line.split()])
        if system is None or len(values) != 3 + 2 * classes:
            raise ValueError("Unexpected classification dp test format")
        if not np.isfinite(values).all() or not np.equal(values[:3], values[:3].astype(int)).all():
            raise ValueError("Expected finite scores and integer frame/true/predicted class")
        logits = values[3 : 3 + classes]
        if int(values[2]) != int(logits.argmax()):
            raise ValueError("Saved prediction disagrees with argmax")
        records.append(
            {
                "recipe_id": Path(system).name,
                "frame": int(values[0]),
                "true": int(values[1]),
                "scores": logits.tolist(),
            }
        )
    if not records:
        raise ValueError("No classification records found")
    keys = [(r["recipe_id"], r["frame"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate recipe/frame in classification output")
    return records


def summarize(records):
    module = metrics_module()
    logits = np.asarray([r["scores"] for r in records])
    labels = np.asarray([r["true"] for r in records])
    result = module.classification_metrics(module.classification_statistics(logits, labels))
    predicted = logits.argmax(axis=1)
    matrix = np.bincount((labels != 0).astype(int) * 2 + (predicted != 0), minlength=4)
    tn, fp, fn, tp = map(int, matrix)
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result.update(
        binary_accuracy=(tn + tp) / len(records),
        binary_macro_f1=(f1_0 + f1_1) / 2,
        binary_confusion_matrix=matrix.reshape(2, 2).tolist(),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        false_positive_rate=fp / (tn + fp) if tn + fp else None,
        binary_rule="argmax among original classes, then map class 0 to 0 and others to 1",
        error_units="probability",
    )
    return result


def recipe_records(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["recipe_id"]].append(row)
    result = []
    for recipe, rows in sorted(groups.items()):
        if len({r["true"] for r in rows}) != 1:
            raise ValueError(f"Conflicting labels within recipe: {recipe}")
        result.append(
            {
                "recipe_id": recipe,
                "true": rows[0]["true"],
                "n_structures": len(rows),
                "scores": np.mean([r["scores"] for r in rows], axis=0).tolist(),
            }
        )
    return result


def prediction_rows(records):
    module = metrics_module()
    probabilities = module.classification_probabilities(np.asarray([r["scores"] for r in records]))
    output = []
    for row, probability in zip(records, probabilities, strict=True):
        prediction = int(np.argmax(row["scores"]))
        output.append(
            {
                "recipe_id": row["recipe_id"],
                "frame": row.get("frame", ""),
                "n_structures": row.get("n_structures", 1),
                "true_class": row["true"],
                "predicted_class": prediction,
                "binary_synthesizable": int(prediction != 0),
                **{f"logit_{c}": v for c, v in enumerate(row["scores"])},
                **{f"prob_{c}": float(v) for c, v in enumerate(probability)},
            }
        )
    return output


def save_reports(directory, records):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    recipes = recipe_records(records)
    write_csv(directory / "structure_predictions.csv", prediction_rows(records))
    write_csv(directory / "recipe_predictions.csv", prediction_rows(recipes))
    structure = summarize(records)
    recipe = summarize(recipes)
    structure["evaluation_unit"] = "structure"
    recipe.update(evaluation_unit="recipe", aggregation="mean raw logits, then argmax")
    write_json(directory / "metrics.json", structure)
    write_json(directory / "recipe_metrics.json", recipe)
    return structure, recipe
