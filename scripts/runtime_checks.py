"""Environment and GPU checks; import and call only inside a Bohrium job."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import unittest
from pathlib import Path

from repro_common import write_json


def initialize(run_tests=True):
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise RuntimeError("Model execution is restricted to Bohrium")
    from use_classification_source import activate

    loader = activate()
    import torch

    if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This reproduction targets a Bohrium V100 GPU")
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "source": loader,
        "deepmd": importlib.metadata.version("deepmd-kit"),
        "execution_location": "Bohrium",
        "training_data_bundled": False,
    }
    write_json("environment.json", report)
    if run_tests:
        suite = unittest.TestSuite()
        for relative in (
            "source/tests/test_classification_metrics.py",
            "source/tests/pt/test_classification.py",
            "source/tests/pt/test_sezm_classification.py",
        ):
            path = Path(loader["source"]) / relative
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        with Path("classification_unit_tests.log").open("w") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        print(Path("classification_unit_tests.log").read_text(), flush=True)
        if not result.wasSuccessful():
            raise RuntimeError("Classification GPU tests failed")
        report["unit_tests_passed"] = result.testsRun
        write_json("environment.json", report)
    return loader
