#!/usr/bin/env python3
"""Check a pinned full upstream checkout against the actual historical runtime files."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from repro_common import ROOT, load_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkout.resolve()
    lock = load_json(ROOT / "upstream/lock.json")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != lock["commit"]:
        raise ValueError(f"Expected commit {lock['commit']}, got {commit}")
    for name, expected in load_json(ROOT / "upstream/runtime_source_sha256.json").items():
        if sha256(source / name) != expected:
            raise ValueError(f"Patched source differs from the historical runtime: {name}")
    print("Full checkout matches all 869 historical runtime source files.")


if __name__ == "__main__":
    main()
