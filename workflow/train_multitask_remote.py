#!/usr/bin/env python3
"""Run native multitask training on Bohrium; hooks only observe gradients."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from collections import Counter
from pathlib import Path


def inspect_classification_head(model) -> dict:
    """Verify the requested native classification head before the first training step."""
    spec = json.loads(Path("reproduction_spec.json").read_text())
    classes, label_name = spec["classes"], spec["label_name"]
    fitting_type = model.get_fitting_net().serialize()["type"]
    if (
        fitting_type != "classification"
        or model.get_task_dim() != classes
        or model.get_var_name() != label_name
        or not model.get_intensive()
        or not model.atomic_model.is_classification
    ):
        raise RuntimeError("Unexpected classification head or label name.")
    return {
        "model_class": type(model).__name__,
        "num_classes": classes,
        "fitting_type": fitting_type,
        "label_name": model.get_var_name(),
        "method": "ce",
        "output_statistics_enabled": False,
    }


def main() -> None:
    if os.environ.get("BOHRIUM_MULTITASK_REMOTE") != "1":
        raise SystemExit("This entrypoint is restricted to the Bohrium job.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input_multitask_100k.json")
    parser.add_argument("--gpu", default="V100")
    args = parser.parse_args()
    config = json.loads(Path(args.input).read_text())
    expected_steps = config["training"]["numb_steps"]
    import torch
    from deepmd.main import main as dp_main
    from deepmd.pt.train.wrapper import ModelWrapper

    if not torch.cuda.is_available() or args.gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"This job requires a Bohrium {args.gpu} GPU.")
    report = {
        "status": "initializing",
        "gpu": torch.cuda.get_device_name(0),
        "expected_total_steps": expected_steps,
        "training_backward_counts": {},
        "descriptor_gradient_probes": {},
        "diagnostics_change_loss_or_sampling": False,
    }
    context = {"active_task": None, "wrapper": None, "snapshots": {}}
    counts = Counter()
    original_share = ModelWrapper.share_params
    original_forward = ModelWrapper.forward

    def write_report() -> None:
        report["training_backward_counts"] = dict(counts)
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        path = Path("multitask_diagnostics.json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    atexit.register(write_report)

    def share(self, *share_args, **kwargs):
        result = original_share(self, *share_args, **kwargs)
        if set(self.model) != {"opencsp", "synthesizability"}:
            raise RuntimeError("Unexpected model branches")
        classifier = self.model["synthesizability"]
        report["classification_head"] = inspect_classification_head(classifier)
        first = dict(self.model["opencsp"].get_descriptor().named_parameters())
        second = dict(self.model["synthesizability"].get_descriptor().named_parameters())
        shared = first.keys() == second.keys() and all(first[n] is second[n] for n in first)
        trainable = bool(first) and all(p.requires_grad for p in first.values())
        report.update(
            descriptor_parameters_shared_by_identity=shared,
            descriptor_all_parameters_trainable=trainable,
            descriptor_parameter_tensors=len(first),
            descriptor_parameter_elements=sum(p.numel() for p in first.values()),
            branch_head_parameter_elements={
                key: sum(p.numel() for p in model.get_fitting_net().parameters() if p.requires_grad)
                for key, model in self.model.items()
            },
        )
        if not shared or not trainable:
            raise RuntimeError("The descriptor must be shared and trainable.")
        context["wrapper"] = self
        for name, parameter in first.items():
            context["snapshots"][name] = parameter.detach().reshape(-1)[:256].cpu().clone()

            def gradient_hook(gradient, name=name):
                task = context["active_task"]
                if task is None or task in report["descriptor_gradient_probes"]:
                    return
                finite = bool(torch.isfinite(gradient).all().item())
                norm = float(gradient.detach().float().norm().item())
                if not finite:
                    raise RuntimeError(f"Nonfinite shared descriptor gradient in {task}: {name}")
                if norm > 0:
                    report["descriptor_gradient_probes"][task] = {
                        "parameter": name,
                        "gradient_l2_norm": norm,
                        "finite": finite,
                    }
                    print("MULTITASK_GRADIENT_PROBE", task, name, norm, flush=True)
                    write_report()

            parameter.register_hook(gradient_hook)
        write_report()
        print(
            "MULTITASK_SHARED_DESCRIPTOR",
            json.dumps(
                {key: value for key, value in report.items() if key.startswith("descriptor_")}
            ),
            flush=True,
        )
        return result

    def forward(self, *forward_args, **kwargs):
        # Internal force differentiation must not count as a training loss backward.
        context["active_task"] = None
        result = original_forward(self, *forward_args, **kwargs)
        if self.training and result[1] is not None and result[1].requires_grad:
            task = kwargs.get("task_key")

            def loss_hook(gradient):
                context["active_task"] = task
                counts[task] += 1
                return gradient

            result[1].register_hook(loss_hook)
        return result

    ModelWrapper.share_params = share
    ModelWrapper.forward = forward
    try:
        dp_main(
            [
                "--pt",
                "train",
                args.input,
                "--finetune",
                "pretrained/model.ckpt-200000.pt",
                "--skip-neighbor-stat",
            ]
        )
        wrapper = context["wrapper"]
        parameters = dict(wrapper.model["opencsp"].get_descriptor().named_parameters())
        differences = {
            name: float((parameters[name].detach().reshape(-1)[:256].cpu() - before).abs().max())
            for name, before in context["snapshots"].items()
            if before.numel()
        }
        report["descriptor_probe_max_abs_change"] = max(differences.values(), default=0.0)
        report["descriptor_changed"] = any(value > 0 for value in differences.values())
        if set(report["descriptor_gradient_probes"]) != {"opencsp", "synthesizability"}:
            raise RuntimeError("Both tasks must produce a finite nonzero descriptor gradient.")
        if sum(counts.values()) != expected_steps or not all(
            counts[key] > 0 for key in ("opencsp", "synthesizability")
        ):
            raise RuntimeError(f"Unexpected training backward counts: {counts}")
        if not report["descriptor_changed"]:
            raise RuntimeError("The shared descriptor probes did not change.")
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report()
        print("MULTITASK_DIAGNOSTICS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
