# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4 classification integration checks, executed on Bohrium for this project."""

import unittest
from unittest.mock import (
    Mock,
    patch,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.classification import (
    ClassificationLoss,
)
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model import (
    SeZMClassificationModel,
    get_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.utils import (
    env,
)


def tiny_model_params() -> dict:
    """Use the existing SeZM property test's small descriptor geometry."""
    return {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, True],
            "mlp_bias": False,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "type": "classification",
            "label_name": "synthesizable",
            "num_classes": 2,
            "neuron": [8],
            "activation_function": "silu",
            "resnet_dt": False,
            "precision": "float32",
            "seed": 7,
        },
        "use_compile": False,
        "enable_tf32": False,
    }


class TestSeZMClassification(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2024)
        self.model = get_model(tiny_model_params()).to(env.DEVICE)

    @staticmethod
    def frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            [
                [
                    [0.5, 0.5, 0.5],
                    [1.5, 1.5, 0.5],
                    [0.5, 1.5, 1.5],
                    [1.5, 0.5, 1.5],
                    [2.2, 2.1, 2.3],
                ]
            ],
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 0, 1, 1, 0]], device=env.DEVICE)
        box = 5.0 * torch.eye(3, dtype=coord.dtype, device=env.DEVICE).reshape(1, 9)
        return coord, atype, box

    def test_logits_intensive_no_derivatives_and_serialization(self) -> None:
        self.assertIsInstance(self.model, SeZMClassificationModel)
        self.assertTrue(self.model.get_intensive())
        self.assertEqual(self.model.get_task_type(), "classification")
        coord, atype, box = self.frame()
        result = self.model(coord, atype, box=box)
        self.assertEqual(result["synthesizable"].shape, (1, 2))
        self.assertEqual(result["atom_synthesizable"].shape, (1, 5, 2))
        self.assertNotIn("force", result)
        self.assertNotIn("virial", result)
        torch.testing.assert_close(
            result["synthesizable"], result["atom_synthesizable"].mean(dim=1)
        )
        restored = BaseModel.deserialize(self.model.serialize()).to(env.DEVICE)
        self.assertIsInstance(restored, SeZMClassificationModel)
        self.assertTrue(restored.atomic_model.is_classification)
        torch.testing.assert_close(
            result["synthesizable"], restored(coord, atype, box=box)["synthesizable"]
        )

    def test_classification_skips_output_stats_even_after_reload(self) -> None:
        atom = self.model.atomic_model
        with torch.no_grad():
            atom.out_bias.fill_(13.0)
            atom.out_std.fill_(7.0)
        restored = BaseModel.deserialize(self.model.serialize()).to(env.DEVICE)
        for atom in (atom, restored.atomic_model):
            sampler = Mock(
                side_effect=AssertionError("must not sample labels for regression")
            )
            bias, std = atom.out_bias.clone(), atom.out_std.clone()
            atom.compute_or_load_out_stat(sampler)
            for mode in ("change-by-statistic", "set-by-statistic"):
                atom.change_out_bias(sampler, bias_adjust_mode=mode)
            sampler.assert_not_called()
            torch.testing.assert_close(atom.out_bias, bias)
            torch.testing.assert_close(atom.out_std, std)
            logits = torch.tensor([[[2.0, -1.0]]], device=env.DEVICE)
            result = atom.apply_out_stat(
                {"synthesizable": logits.clone()},
                torch.tensor([[0]], device=env.DEVICE),
            )
            torch.testing.assert_close(result["synthesizable"], logits)

    def test_energy_keeps_regression_stats_and_bias(self) -> None:
        params = tiny_model_params()
        params["fitting_net"] = {
            "type": "dpa4_ener",
            "neuron": [8],
            "precision": "float32",
        }
        atom = get_model(params).to(env.DEVICE).atomic_model
        self.assertFalse(atom.is_classification)
        with patch.object(BaseAtomicModel, "change_out_bias") as change:
            sampler = Mock()
            atom.change_out_bias(sampler, bias_adjust_mode="set-by-statistic")
            change.assert_called_once_with(sampler, None, "set-by-statistic")
        with torch.no_grad():
            atom.out_bias.fill_(3.0)
            atom.out_std.fill_(10.0)
        energy = torch.ones((1, 1, 1), device=env.DEVICE)
        result = atom.apply_out_stat(
            {"energy": energy}, torch.tensor([[0]], device=env.DEVICE)
        )
        torch.testing.assert_close(
            result["energy"], torch.full_like(result["energy"], 4.0)
        )

    def test_cross_entropy_gradient_metrics_and_loss_roundtrip(self) -> None:
        coord, atype, box = self.frame()
        loss = ClassificationLoss(2, "synthesizable")
        result, value, metrics = loss(
            {"coord": coord, "atype": atype, "box": box},
            self.model,
            {"synthesizable": torch.tensor([[0]], device=env.DEVICE)},
            natoms=5,
        )
        expected = F.cross_entropy(
            result["synthesizable"], torch.tensor([0], device=env.DEVICE)
        )
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(metrics["ce"], value.detach())
        torch.testing.assert_close(metrics["rmse"].square(), metrics["mse"])
        value.backward()
        self.assertTrue(
            any(
                p.grad is not None and bool(torch.count_nonzero(p.grad))
                for p in self.model.get_descriptor().parameters()
            )
        )
        self.assertEqual(
            ClassificationLoss.deserialize(loss.serialize()).metric, loss.metric
        )
