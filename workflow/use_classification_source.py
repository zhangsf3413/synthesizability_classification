#!/usr/bin/env python3
"""Activate patched Python sources with the image's existing compiled DeepMD libs.

Only changes the current process's import search path. Never edits or installs
over the system environment, and never creates or evaluates a model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def activate(source: str | Path = "deepmd-kit-classification") -> dict:
    source = Path(source).resolve()
    if not (source / "deepmd/pt/model/model/sezm_classification_model.py").is_file():
        raise RuntimeError(f"Patched classification source is missing: {source}")
    import deepmd

    unexpected = [
        key
        for key in sys.modules
        if key.startswith("deepmd.")
        and key
        not in {
            "deepmd._version",
            "deepmd.__about__",
            "deepmd.lib",
        }
    ]
    if unexpected:
        raise RuntimeError(f"Activate before importing DeepMD implementation modules: {unexpected}")
    spec = importlib.util.find_spec("deepmd.lib")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("The image must contain a built DeepMD PyTorch installation")
    binary_root = Path(next(iter(spec.submodule_search_locations))).parent
    # A scikit-build editable finder can override sys.path and __path__. Disable
    # only a finder that redirects DeepMD; keep its generated binary resources.
    redirects = [
        finder for finder in sys.meta_path if "deepmd" in getattr(finder, "known_source_files", {})
    ]
    sys.meta_path[:] = [finder for finder in sys.meta_path if finder not in redirects]
    deepmd.__path__[:] = [str(source / "deepmd"), str(binary_root)]
    deepmd.__spec__.submodule_search_locations = deepmd.__path__
    sys.path.insert(0, str(source))
    from deepmd.utils import argcheck

    if not Path(argcheck.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Schema did not load from the patched repository")
    return {
        "source": str(source),
        "compiled_resources": str(binary_root),
        "schema_source": argcheck.__file__,
        "editable_redirectors_disabled": len(redirects),
    }
