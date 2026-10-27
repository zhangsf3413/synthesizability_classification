"""Predict external unlabeled DeepMD/Numpy systems on Bohrium, structure by structure."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from repro_common import load_json, metrics_module, sha256, write_csv, write_json


def remap_types(types, local_map, model_map):
    types = np.asarray(types)
    if (
        not np.equal(types, types.astype(int)).all()
        or (types < 0).any()
        or (types >= len(local_map)).any()
    ):
        raise ValueError("Invalid local atom types")
    missing = set(local_map) - set(model_map)
    if missing:
        raise ValueError(f"Elements unsupported by the model: {sorted(missing)}")
    remap = np.asarray([model_map.index(symbol) for symbol in local_map], dtype=np.int64)
    return remap[types.astype(int)]


def main():
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise SystemExit("Model execution is restricted to Bohrium")
    spec = load_json("reproduction_spec.json")
    if sha256("best_checkpoint.pt") != spec["model_sha256"]:
        raise ValueError("Selected model hash mismatch")
    sys.path.insert(0, str(Path("bin").resolve()))
    from use_classification_source import activate

    activate()
    from deepmd.infer.deep_classification import DeepClassification

    model = DeepClassification("best_checkpoint.pt", head="synthesizability")
    if model.get_task_dim() != spec["classes"] or model.get_var_name() != spec["label_name"]:
        raise ValueError("Unexpected classification head")
    root = Path(spec["external_dataset_root"]).resolve()
    index = root / "prediction_index.csv"
    if index.is_file():
        with index.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = []
        for folder in sorted(root.rglob("set.*")):
            if not (folder / "coord.npy").is_file():
                continue
            for i in range(len(np.load(folder / "coord.npy", mmap_mode="r"))):
                rows.append(
                    {
                        "recipe_id": folder.parent.name,
                        "system": str(folder.parent.relative_to(root)),
                        "set": folder.name,
                        "frame": str(i),
                    }
                )
    if not rows:
        raise ValueError("No prediction structures found")
    output, cache, groups = [], {}, defaultdict(list)
    module = metrics_module()

    def fields(logits):
        probability = module.classification_probabilities(np.asarray(logits).reshape(1, -1))[0]
        prediction = int(np.argmax(logits))
        return {
            "predicted_class": prediction,
            "predicted_class_name": spec["class_order"][prediction],
            "binary_synthesizable": int(prediction != 0),
            **{f"logit_{i}": float(v) for i, v in enumerate(logits)},
            **{f"prob_{i}": float(v) for i, v in enumerate(probability)},
        }

    all_logits = []
    for row in rows:
        relative = Path(row["system"]) / row.get("set", "set.000")
        folder = (root / relative).resolve()
        if not folder.is_relative_to(root):
            raise ValueError("Prediction path escapes dataset root")
        if folder not in cache:
            coord = np.load(folder / "coord.npy", allow_pickle=False)
            box = np.load(folder / "box.npy", allow_pickle=False)
            local_map = (folder.parent / "type_map.raw").read_text().split()
            types_file = folder / "real_atom_types.npy"
            if types_file.is_file():
                types = np.load(types_file, allow_pickle=False)
            else:
                types = np.loadtxt(folder.parent / "type.raw", dtype=int, ndmin=1)
                types = np.broadcast_to(types, (len(coord), len(types)))
            if len(coord) != len(box) or len(types) != len(coord):
                raise ValueError("Coordinate, cell and type frame counts differ")
            cache[folder] = (coord, box, remap_types(types, local_map, model.get_type_map()))
        coord, box, types = cache[folder]
        i = int(row["frame"])
        if not 0 <= i < len(coord) or coord.shape[1] != types.shape[1] * 3:
            raise ValueError("Invalid frame or atom count")
        logits = np.asarray(model.eval(coord[i : i + 1], box[i : i + 1], types[i])[0])[0]
        if logits.shape != (spec["classes"],) or not np.isfinite(logits).all():
            raise ValueError("Unexpected prediction shape or nonfinite logits")
        all_logits.append(logits)
        groups[row["recipe_id"]].append(logits)
        output.append(
            {
                "recipe_id": row["recipe_id"],
                "system": row["system"],
                "set": row.get("set", "set.000"),
                "frame": i,
                "n_atoms": len(types[i]),
                **fields(logits),
            }
        )
    destination = Path("predictions")
    destination.mkdir()
    write_csv(destination / "structure_predictions.csv", output)
    write_csv(
        destination / "recipe_predictions.csv",
        [
            {"recipe_id": recipe, "n_structures": len(logits), **fields(np.mean(logits, axis=0))}
            for recipe, logits in sorted(groups.items())
        ],
    )
    np.save(destination / "logits.npy", np.asarray(all_logits))
    write_json(
        destination / "summary.json",
        {
            "structures": len(output),
            "recipes": len(groups),
            "model_sha256": spec["model_sha256"],
            "class_order": spec["class_order"],
            "recipe_rule": "mean raw logits then argmax",
            "binary_rule": "argmax then merge classes > 0",
            "ground_truth_used": False,
            "accuracy_f1_computed": False,
        },
    )
    model.close()


if __name__ == "__main__":
    main()
