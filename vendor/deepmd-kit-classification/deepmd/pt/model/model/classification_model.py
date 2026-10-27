# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any

import torch

from deepmd.dpmodel.output_def import OutputVariableDef
from deepmd.pt.model.atomic_model import DPClassificationAtomicModel
from deepmd.pt.model.model.model import BaseModel

from .dp_model import DPModelCommon
from .make_model import make_model

DPClassificationModel_ = make_model(DPClassificationAtomicModel)


@BaseModel.register("classification")
class ClassificationModel(DPModelCommon, DPClassificationModel_):
    """System-level single-label classification model."""

    model_type = "classification"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DPModelCommon.__init__(self)
        DPClassificationModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        data = self.model_output_def().get_data()
        output = {
            f"atom_{self.get_var_name()}": data[self.get_var_name()],
            self.get_var_name(): data[f"{self.get_var_name()}_redu"],
        }
        if "mask" in data:
            output["mask"] = data["mask"]
        return output

    def _translate(self, result: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = {
            f"atom_{self.get_var_name()}": result[self.get_var_name()],
            self.get_var_name(): result[f"{self.get_var_name()}_redu"],
        }
        if "mask" in result:
            output["mask"] = result["mask"]
        return output

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._translate(
            self.forward_common(
                coord,
                atype,
                box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                charge_spin=charge_spin,
            )
        )

    @torch.jit.export
    def get_task_type(self) -> str:
        return "classification"

    @torch.jit.export
    def get_task_dim(self) -> int:
        return self.get_fitting_net().dim_out

    @torch.jit.export
    def get_intensive(self) -> bool:
        return True

    @torch.jit.export
    def get_var_name(self) -> str:
        return self.get_fitting_net().var_name

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self._translate(
            self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                comm_dict=comm_dict,
                extra_nlist_sort=self.need_sorted_nlist_for_lower(),
                charge_spin=charge_spin,
            )
        )
