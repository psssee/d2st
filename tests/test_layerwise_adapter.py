"""CPU checks for the depth-aware spatial/temporal adapter router."""

import ast
import math
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_router():
    tree = ast.parse((ROOT / "models/base/adapter.py").read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if getattr(node, "name", "") == "D2STLayerwiseSTRouter"
    ]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<layerwise-router>", "exec"),
        namespace,
    )
    return namespace["D2STLayerwiseSTRouter"]


Router = load_router()


class LayerwiseAdapterTests(unittest.TestCase):
    def test_shallow_and_deep_initialization_have_expected_bias(self):
        shallow = Router(0, 12, transition_layer=6)
        deep = Router(11, 12, transition_layer=6)
        self.assertGreater(shallow.get_weights()[0], shallow.get_weights()[1])
        self.assertGreater(deep.get_weights()[1], deep.get_weights()[0])

    def test_shape_and_gradient(self):
        router = Router(6, 12, transition_layer=6)
        spatial = torch.randn(5, 4, 7, requires_grad=True)
        temporal = torch.randn(5, 4, 7, requires_grad=True)
        output = router(spatial, temporal)
        self.assertEqual(tuple(output.shape), tuple(spatial.shape))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(router.routing_logits.grad)
        self.assertTrue(torch.isfinite(router.routing_logits.grad).all())

    def test_config_pair_differs_only_in_router_and_runtime_fields(self):
        import yaml

        experiment = yaml.safe_load(
            (ROOT / "config/ssv2_full/ViT_SSv2_full_LAYERWISE_ST_ADAPTER.yaml").read_text()
        )
        control = yaml.safe_load(
            (ROOT / "config/ssv2_full/ViT_SSv2_full_LAYERWISE_ST_ADAPTER_CONTROL.yaml").read_text()
        )
        self.assertTrue(experiment["LAYERWISE_ST_ADAPTER"]["ENABLE"])
        self.assertFalse(control["LAYERWISE_ST_ADAPTER"]["ENABLE"])
        for key in experiment:
            if key in {"LAYERWISE_ST_ADAPTER", "INIT_METHOD", "CUDA_VISIBLE_DEVICES", "OUTPUT_DIR"}:
                continue
            self.assertEqual(experiment[key], control[key], key)


if __name__ == "__main__":
    unittest.main()
