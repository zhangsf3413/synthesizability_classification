# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class ClassificationLoss(TaskLoss):
    """Cross-entropy loss for system-level single-label classification."""

    def __init__(
        self,
        num_classes: int,
        var_name: str,
        metric: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.var_name = var_name
        self.metric = (
            ["ce", "accuracy", "mae", "mse", "rmse"] if metric is None else metric
        )
        unknown = set(self.metric) - {"ce", "accuracy", "mae", "mse", "rmse"}
        if unknown:
            raise ValueError(f"Unknown classification metrics: {sorted(unknown)}")

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred = self._inject_atom_mask(model(**input_dict), input_dict)
        logits = model_pred[self.var_name]
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits with shape (batch, {self.num_classes}), "
                f"but got {tuple(logits.shape)}"
            )
        if not bool(torch.all(torch.isfinite(logits))):
            raise ValueError("Classification logits must be finite")

        target = label[self.var_name]
        if target.ndim == 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected labels with shape ({logits.shape[0]},) or "
                f"({logits.shape[0]}, 1), but got {tuple(target.shape)}"
            )
        if target.is_floating_point():
            if not bool(torch.all(torch.isfinite(target))):
                raise ValueError("Class labels must be finite integers")
            if not bool(torch.all(target == torch.round(target))):
                raise ValueError("Class labels must be integers")

        target = target.to(device=logits.device, dtype=torch.long)
        if bool(torch.any(target < 0)) or bool(torch.any(target >= self.num_classes)):
            raise ValueError(f"Class labels must be in [0, {self.num_classes})")

        loss = F.cross_entropy(logits, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Classification cross-entropy must be finite")
        # Softmax is used only for reporting probability errors. Cross-entropy
        # above takes raw logits and performs log-softmax internally.
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=-1)
            truth = F.one_hot(target, self.num_classes).to(logits.dtype)
            error = probabilities - truth
            mse = error.square().mean()
            metrics = {
                "ce": loss.detach(),
                "accuracy": (logits.argmax(dim=-1) == target).to(logits.dtype).mean(),
                "mae": error.abs().mean(),
                "mse": mse,
                "rmse": mse.sqrt(),
            }
        return model_pred, loss, {name: metrics[name] for name in self.metric}

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        return [
            DataRequirementItem(
                self.var_name,
                ndof=1,
                atomic=False,
                must=True,
            )
        ]

    def serialize(self) -> dict:
        return {
            "@class": "ClassificationLoss",
            "@version": 1,
            "num_classes": self.num_classes,
            "var_name": self.var_name,
            "metric": self.metric,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ClassificationLoss":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
