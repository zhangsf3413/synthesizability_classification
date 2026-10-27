#!/usr/bin/env python3
"""Convert a recipe_id,path CSV of POSCAR files to unlabeled prediction dpdata."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read
from repro_common import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="CSV columns: recipe_id,path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    with args.index.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        recipe = row["recipe_id"]
        if not recipe or Path(recipe).name != recipe or recipe in (".", ".."):
            raise ValueError("recipe_id must be a plain folder name")
        path = Path(row["path"])
        if not path.is_absolute():
            path = args.index.resolve().parent / path
        atoms = read(path, format="vasp")
        if abs(np.linalg.det(atoms.cell.array)) < 1e-8 or not np.isfinite(atoms.positions).all():
            raise ValueError(f"Expected finite periodic structure: {path}")
        grouped[(recipe, len(atoms))].append((path, atoms))
    if not grouped:
        raise ValueError("Input index has no structures")
    index = []
    for number, ((recipe, natoms), group) in enumerate(sorted(grouped.items()), 1):
        system = f"system_{number:04d}"
        folder = args.output / system / "set.000"
        folder.mkdir(parents=True)
        elements = sorted({symbol for _, atoms in group for symbol in atoms.get_chemical_symbols()})
        coords, cells, types = [], [], []
        for i, (path, atoms) in enumerate(group):
            coords.append(atoms.positions.reshape(-1))
            cells.append(atoms.cell.array.reshape(-1))
            types.append([elements.index(s) for s in atoms.get_chemical_symbols()])
            index.append(
                {
                    "recipe_id": recipe,
                    "system": system,
                    "set": "set.000",
                    "frame": i,
                    "source_path": str(path.resolve()),
                    "n_atoms": natoms,
                }
            )
        np.save(folder / "coord.npy", np.asarray(coords, dtype=np.float64))
        np.save(folder / "box.npy", np.asarray(cells, dtype=np.float64))
        np.save(folder / "real_atom_types.npy", np.asarray(types, dtype=np.int32))
        (folder.parent / "type_map.raw").write_text("\n".join(elements) + "\n")
        np.savetxt(folder.parent / "type.raw", np.zeros(natoms, dtype=int), fmt="%d")
    write_csv(args.output / "prediction_index.csv", index)
    print(f"Prepared {len(index)} unlabeled structures in {len(grouped)} systems: {args.output}")


if __name__ == "__main__":
    main()
