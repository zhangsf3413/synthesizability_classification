"""Read external dpdata on Bohrium and create the exact class-label layout."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from repro_common import load_json, sha256, write_json


def fingerprint(coord, box, types):
    digest = hashlib.sha256()
    for array, dtype in ((coord, "<f8"), (box, "<f8"), (types, "<i8")):
        digest.update(np.ascontiguousarray(array, dtype=dtype).reshape(-1).tobytes())
    return digest.hexdigest()


def stage(splits=("train", "valid")):
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise RuntimeError("Data staging for model runs is restricted to the Bohrium job")
    spec = load_json("reproduction_spec.json")
    manifest = load_json("experiment_manifest.json")
    source = Path(spec["external_dataset_root"]).resolve()
    selected_index = "holdout_index.csv" if tuple(splits) == ("test",) else "development_index.csv"
    if tuple(splits) not in (("train", "valid"), ("test",)):
        raise ValueError("Stage development or test separately")
    index = Path("metadata") / selected_index
    expected_hash = manifest[
        "holdout_index_sha256" if tuple(splits) == ("test",) else "development_index_sha256"
    ]
    if sha256(index) != expected_hash:
        raise ValueError("Split-index hash mismatch")
    with index.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    root = Path("heldout" if tuple(splits) == ("test",) else "data") / spec["data_name"]
    if root.exists():
        raise ValueError(f"Refusing to overwrite staged data: {root}")
    root.mkdir(parents=True)
    groups = defaultdict(list)
    for row in rows:
        if row["split"] not in splits:
            raise ValueError("Unexpected split in index")
        groups[row["dpdata_system"]].append(row)
    seen = set()
    for relative, indexed in sorted(groups.items()):
        src = source / relative
        target = root / relative
        arrays = {}
        for key in ("coord", "box", "real_atom_types", "synthesizability"):
            arrays[key] = np.load(src / "set.000" / f"{key}.npy", allow_pickle=False)
        n = len(indexed)
        if any(len(array) != n for array in arrays.values()):
            raise ValueError(f"Frame count mismatch: {relative}")
        onehot = arrays["synthesizability"]
        if onehot.shape != (n, 3) or not np.array_equal(onehot, np.eye(3)[onehot.argmax(axis=1)]):
            raise ValueError(f"Expected original three-column one-hot labels: {relative}")
        labels = onehot.argmax(axis=1).astype(np.int32)
        if spec["classes"] == 2:
            labels = (labels != 0).astype(np.int32)
        for row in indexed:
            i = int(row["dpdata_frame"])
            identity = fingerprint(
                arrays["coord"][i], arrays["box"][i], arrays["real_atom_types"][i]
            )
            if identity != row["structure_array_sha256"] or identity in seen:
                raise ValueError(f"Structure mismatch or duplicate: {relative}/{i}")
            seen.add(identity)
            label_key = "synthesizable_id" if spec["classes"] == 2 else "class_id"
            if int(row[label_key]) != int(labels[i]):
                raise ValueError("Experimental class label mismatch")
        (target / "set.000").mkdir(parents=True)
        for key in arrays:
            shutil.copyfile(src / "set.000" / f"{key}.npy", target / "set.000" / f"{key}.npy")
        for name in ("type.raw", "type_map.raw"):
            shutil.copyfile(src / name, target / name)
        np.save(target / "set.000" / f"{spec['label_name']}.npy", labels[:, None])
    shutil.copyfile(index, root / "dataset_index.csv")
    write_json(
        root / "staging_checks.json",
        {
            "structures": len(seen),
            "splits": splits,
            "source": str(source),
            "structure_hashes_verified": True,
            "labels": spec["label_name"],
        },
    )
    return root


def stage_background():
    spec = load_json("reproduction_spec.json")
    background = Path(spec["external_background_root"]).resolve()
    for split in ("train", "valid"):
        if not (background / split).is_dir():
            raise FileNotFoundError(f"Missing background split: {background / split}")
    target = Path("data/background")
    if target.exists():
        raise ValueError("Refusing to overwrite data/background")
    target.parent.mkdir(exist_ok=True)
    target.symlink_to(background, target_is_directory=True)
    pretrain = Path(spec["external_pretrained"]).resolve()
    manifest = load_json("experiment_manifest.json")
    if sha256(pretrain) != manifest["pretrained_sha256"]:
        raise ValueError("Original DPA4 pretrained checkpoint hash mismatch")
    Path("pretrained").mkdir()
    Path("pretrained/model.ckpt-200000.pt").symlink_to(pretrain)
