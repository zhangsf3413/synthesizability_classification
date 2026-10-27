#!/usr/bin/env python3
"""Verify an external 8/1/1 dpdata dataset without model execution or copying it."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from repro_common import ROOT, write_json
from stage_data import fingerprint


def check(source):
    with (ROOT / "metadata/three/dataset_index.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[row["dpdata_system"]].append(row)
    seen = set()
    recipes = defaultdict(set)
    counts = defaultdict(Counter)
    for relative, indexed in sorted(groups.items()):
        folder = source / relative / "set.000"
        arrays = [
            np.load(folder / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            for key in ("coord", "box", "real_atom_types", "synthesizability")
        ]
        coord, box, types, onehot = arrays
        if any(len(a) != len(indexed) for a in arrays):
            raise ValueError(f"Unexpected frame count: {relative}")
        if onehot.shape[1:] != (3,) or not np.array_equal(onehot, np.eye(3)[onehot.argmax(axis=1)]):
            raise ValueError(f"Expected one-hot (N,3): {relative}")
        for row in indexed:
            i = int(row["dpdata_frame"])
            identity = fingerprint(coord[i], box[i], types[i])
            if identity != row["structure_array_sha256"] or identity in seen:
                raise ValueError(f"Structure mismatch or duplicate: {relative}/{i}")
            if int(onehot[i].argmax()) != int(row["class_id"]):
                raise ValueError("Experimental label mismatch")
            seen.add(identity)
            recipes[row["split"]].add(row["recipe_id"])
            counts[row["split"]][row["synthesizability_label"]] += 1
    if any(
        recipes[a] & recipes[b]
        for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
    ):
        raise ValueError("Recipe leakage")
    return {
        "status": "passed",
        "structures": len(seen),
        "counts": {
            s: {"recipes": len(recipes[s]), "structures_by_class": dict(counts[s])}
            for s in ("train", "valid", "test")
        },
        "files_copied": 0,
        "local_model_execution": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.dataset_root.resolve())
    if args.output:
        write_json(args.output, result)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
