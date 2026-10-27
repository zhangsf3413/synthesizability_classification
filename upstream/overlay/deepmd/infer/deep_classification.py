# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference interface for system-level single-label classification."""

from typing import (
    Any,
)

import numpy as np

from deepmd.utils.classification import (
    classification_probabilities,
)

from .deep_property import (
    DeepProperty,
)


class DeepClassification(DeepProperty):
    """Evaluate raw logits with ``eval`` or probabilities with ``eval_probabilities``.

    The inherited ``eval`` contract returns a tuple whose first array has shape
    ``(n_structures, num_classes)``. The final category is its row-wise argmax.
    """

    def eval_probabilities(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return system probabilities; per-atom logits are not atom labels."""
        return classification_probabilities(self.eval(*args, **kwargs)[0])
