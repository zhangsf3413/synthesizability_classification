"""Native DeepMD argument lists; defaults are left to the pinned upstream parser."""

from __future__ import annotations

import shlex
import subprocess


def train_command():
    return [
        "dp",
        "--pt",
        "train",
        "input.json",
        "--finetune",
        "pretrained/model.ckpt-200000.pt",
        "--skip-neighbor-stat",
    ]


def test_command(checkpoint, system, detail):
    return [
        "dp",
        "--pt",
        "test",
        "-m",
        str(checkpoint),
        "-s",
        str(system),
        "-n",
        "0",
        "--head",
        "synthesizability",
        "-d",
        str(detail),
    ]


def run(command):
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)
