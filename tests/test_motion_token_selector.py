"""Regression checks for motion-guided patch-token selection."""

import ast
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_selector():
    tree = ast.parse((ROOT / "models/base/adapter.py").read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", "") == "D2STMotionTokenSelector"
    ]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<motion-selector>", "exec"),
        namespace,
    )
    return namespace["D2STMotionTokenSelector"]


Selector = load_selector()


class MotionTokenSelectorTests(unittest.TestCase):
    def test_shape_and_finite_diagnostics(self):
        selector = Selector(8)
        cls_features = torch.randn(2, 4, 8)
        patch_features = torch.randn(2, 4, 16, 8)
        output = selector(cls_features, patch_features)
        self.assertEqual(tuple(output.shape), tuple(cls_features.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(torch.tensor(list(selector.get_diagnostics().values()))).all())

    def test_zero_initialized_projection_preserves_baseline(self):
        selector = Selector(8)
        cls_features = torch.randn(1, 4, 8)
        patch_features = torch.randn(1, 4, 16, 8)
        output = selector(cls_features, patch_features)
        self.assertTrue(torch.allclose(output, cls_features, atol=1e-6))

    def test_selector_parameters_receive_gradients(self):
        selector = Selector(8)
        cls_features = torch.randn(2, 4, 8, requires_grad=True)
        patch_features = torch.randn(2, 4, 16, 8, requires_grad=True)
        selector(cls_features, patch_features).square().mean().backward()
        self.assertIsNotNone(selector.alpha.grad)
        self.assertIsNotNone(selector.action_projection[-1].weight.grad)

    def test_config_pair_differs_only_in_selector_and_runtime_fields(self):
        import yaml

        experiment = yaml.safe_load(
            (ROOT / "config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_MOTION_TOKEN.yaml").read_text()
        )
        control = yaml.safe_load(
            (ROOT / "config/ssv2_full/ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_MOTION_TOKEN_CONTROL.yaml").read_text()
        )
        self.assertTrue(experiment["MOTION_TOKEN_SELECTOR"]["ENABLE"])
        self.assertFalse(control["MOTION_TOKEN_SELECTOR"]["ENABLE"])
        self.assertTrue(experiment["TASK_MATCH"]["ENABLE"])
        self.assertTrue(experiment["MULTI_VELOCITY"]["ENABLE"])
        for key in experiment:
            if key in {"MOTION_TOKEN_SELECTOR", "INIT_METHOD", "CUDA_VISIBLE_DEVICES", "OUTPUT_DIR"}:
                continue
            self.assertEqual(experiment[key], control[key], key)


if __name__ == "__main__":
    unittest.main()
