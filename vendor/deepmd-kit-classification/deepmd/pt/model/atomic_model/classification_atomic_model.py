# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.classification import (
    ClassificationFittingNet,
)
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPClassificationAtomicModel(DPAtomicModel):
    """Atomic model producing unnormalized classification logits."""

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, ClassificationFittingNet):
            raise TypeError("fitting must be a ClassificationFittingNet")
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        return True

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        # Integer class IDs are categorical and have no regression statistics.
        return None

    def change_out_bias(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
    ) -> None:
        return None

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Logits must not be shifted or scaled by label statistics.
        return ret
