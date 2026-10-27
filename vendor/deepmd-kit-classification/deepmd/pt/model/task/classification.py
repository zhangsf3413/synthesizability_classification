# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@Fitting.register("classification")
class ClassificationFittingNet(InvarFitting):
    """Fit per-atom logits for system-level single-label classification."""

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        num_classes: int,
        label_name: str = "class",
        neuron: list[int] = [128, 128, 128],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        trainable: bool | list[bool] = True,
        seed: int | None = None,
        default_fparam: list | None = None,
        **kwargs: Any,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        super().__init__(
            var_name=label_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=num_classes,
            neuron=neuron,
            bias_atom_e=None,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            default_fparam=default_fparam,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.num_classes],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=True,
                )
            ]
        )

    def get_intensive(self) -> bool:
        """Class logits are averaged over atoms, independent of cell size."""
        return True

    @classmethod
    def deserialize(cls, data: dict) -> "ClassificationFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("dim_out")
        data["label_name"] = data.pop("var_name")
        return super().deserialize(data)

    def serialize(self) -> dict:
        data = {
            **InvarFitting.serialize(self),
            "type": "classification",
            "num_classes": self.num_classes,
        }
        data["@version"] = 1
        return data

    exclude_types: list[int]
