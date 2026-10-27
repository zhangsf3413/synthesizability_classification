"""Validate only native classification/CE multitask configurations and labels."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from repro_common import load_json, sha256


def expanded(config):
    result = deepcopy(config)
    model = result["model"]
    defaults = {
        key: value for key, value in model.items() if key not in ("shared_dict", "model_dict")
    }
    for branch in model["model_dict"].values():
        for key, value in defaults.items():
            branch.setdefault(key, deepcopy(value))
        for key in ("type_map", "descriptor", "fitting_net"):
            if isinstance(branch.get(key), str):
                branch[key] = deepcopy(model["shared_dict"][branch[key]])
    for key in defaults:
        model.pop(key)
    return result


def validate():
    from deepmd.utils.argcheck import normalize
    from deepmd.utils.data_system import process_systems

    config = load_json("input.json")
    spec = load_json("reproduction_spec.json")
    manifest = load_json("experiment_manifest.json")
    normalized = normalize(expanded(config), multi_task=True)
    branch = normalized["model"]["model_dict"]["synthesizability"]
    fitting = branch["fitting_net"]
    if fitting["type"] != "classification" or fitting["num_classes"] != spec["classes"]:
        raise ValueError("Only native classification heads are supported")
    if fitting["label_name"] != spec["label_name"] or fitting["neuron"] != [64, 64]:
        raise ValueError("Unexpected classification label or hidden layers")
    if normalized["loss_dict"]["synthesizability"]["type"] != "classification":
        raise ValueError("Expected the native classification cross-entropy loss")
    branches = config["model"]["model_dict"]
    if branches["opencsp"]["descriptor"] != branches["synthesizability"]["descriptor"]:
        raise ValueError("Descriptor configuration must be shared")
    if not fitting["trainable"] or not all(
        b["descriptor"]["trainable"] for b in normalized["model"]["model_dict"].values()
    ):
        raise ValueError("Classification head and shared descriptor must be trainable")
    root = Path("data") / spec["data_name"]
    if (root / "test").exists() or Path("heldout").exists():
        raise ValueError("Test must not be staged before validation selection")
    if sha256(root / "dataset_index.csv") != manifest["development_index_sha256"]:
        raise ValueError("Development index mismatch")
    counts, train_systems, train_labels, train_sizes = {}, [], [], []
    for split, key in (("train", "training_data"), ("valid", "validation_data")):
        systems = sorted((root / split).glob("*/set.000"))
        class_counts = Counter()
        classes = []
        for folder in systems:
            labels = np.load(folder / f"{spec['label_name']}.npy", allow_pickle=False)
            if labels.dtype != np.int32 or labels.shape[1:] != (1,) or not len(labels):
                raise ValueError("Expected nonempty int32 class labels with shape (N,1)")
            if not np.isin(labels, range(spec["classes"])).all() or not np.all(labels == labels[0]):
                raise ValueError("Invalid class range or conflicting labels within a recipe")
            class_counts[int(labels[0, 0])] += len(labels)
            classes.append(int(labels[0, 0]))
            if split == "train":
                train_sizes.append(len(labels))
        discovered = [str(Path(p).resolve()) for p in process_systems(str(root / split))]
        if discovered != [str(p.parent.resolve()) for p in systems]:
            raise ValueError("System order must match sorted sys_probs positions")
        data = config["training"]["data_dict"]["synthesizability"][key]
        if data["systems"] != f"data/{spec['data_name']}/{split}" or data["batch_size"] != 1:
            raise ValueError("Unexpected data path or batch size")
        expected = manifest["dataset_counts"][split]
        if (
            len(systems) != expected["recipes"]
            or sum(class_counts.values()) != expected["structures"]
        ):
            raise ValueError("Dataset counts mismatch")
        if {str(c): class_counts[c] for c in range(spec["classes"])} != expected[
            "structures_by_class"
        ]:
            raise ValueError("Class counts mismatch")
        counts[split] = {"recipes": len(systems), "structures_by_class": dict(class_counts)}
        if split == "train":
            train_systems, train_labels = systems, classes
    data = config["training"]["data_dict"]["synthesizability"]["training_data"]
    if spec["sampling"] == "balanced":
        per_class = Counter(train_labels)
        expected = [1 / (spec["classes"] * per_class[c]) for c in train_labels]
        if len(data["sys_probs"]) != len(train_systems) or not np.allclose(
            data["sys_probs"], expected, rtol=0, atol=1e-15
        ):
            raise ValueError("Balanced sampling probabilities mismatch")
        probabilities = data["sys_probs"]
    else:
        if "sys_probs" in data or data["auto_prob"] != "prob_sys_size":
            raise ValueError("Expected natural system-size sampling")
        probabilities = np.asarray(train_sizes) / sum(train_sizes)
    mass = {
        str(c): float(
            sum(p for p, label in zip(probabilities, train_labels, strict=True) if label == c)
        )
        for c in range(spec["classes"])
    }
    training = config["training"]
    if (
        training["model_prob"] != {"opencsp": 0.5, "synthesizability": 0.5}
        or training["numb_steps"] != 100000
        or training["save_freq"] != 5000
    ):
        raise ValueError("Unexpected multitask schedule")
    return {
        "status": "passed",
        "classification_loss": "cross_entropy",
        "counts": counts,
        "class_sampling_mass": mass,
        "test_arrays_opened": False,
    }
