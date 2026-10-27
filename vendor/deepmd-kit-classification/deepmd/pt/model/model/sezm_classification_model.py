# SPDX-License-Identifier: LGPL-3.0-or-later
"""System-level classification through the SeZM sparse-edge descriptor."""

import torch

from deepmd.pt.model.model.model import (
    BaseModel,
)

from .sezm_property_model import (
    SeZMPropertyModel,
)


@BaseModel.register("SeZMClassification")
@BaseModel.register("sezm_classification")
@BaseModel.register("DPA4Classification")
@BaseModel.register("dpa4_classification")
class SeZMClassificationModel(SeZMPropertyModel):
    """Average per-atom logits without energy derivatives or label scaling.

    The non-conservative sparse-edge path is shared with invariant properties.
    Categorical output statistics are disabled in ``SeZMAtomicModel`` so that
    training, checkpoint reloads, and inference use the same raw logits.
    """

    model_type = "SeZMClassification"

    @torch.jit.export
    def get_task_type(self) -> str:
        """Identify classification independently of the user-defined label name."""
        return "classification"
