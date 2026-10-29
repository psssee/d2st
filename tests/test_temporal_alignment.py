"""Regression tests for D2ST's order-aware core matching path."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def rearrange_frames(tensor, pattern):
    if pattern != "q s t c -> q (s t) c":
        raise AssertionError(f"Unexpected feature-only rearrange: {pattern}")
    return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    for node in nodes:
        node.decorator_list = []
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


adapter_namespace = load_definitions(
    ROOT / "models/base/adapter.py",
    {"OTAM_dist", "ViT_CLIP"},
    {"torch": torch, "nn": nn, "F": F, "rearrange": rearrange_frames},
)
ViT = adapter_namespace["ViT_CLIP"]


class TemporalAlignmentTests(unittest.TestCase):
    def make_model(self, num_frames=3):
        model = ViT.__new__(ViT)
        nn.Module.__init__(model)
        model.num_frames = num_frames
        model.temporal_alignment_lambda = 0.5
        model.extract_class_indices = MethodType(
            lambda self, labels, label: torch.nonzero(
                labels == label, as_tuple=False
            ).reshape(-1),
            model,
        )
        model.args = SimpleNamespace()
        return model

    def test_otam_is_finite_for_large_distances(self):
        distances = torch.full((1, 1, 4, 4), 1000.0, requires_grad=True)
        result = adapter_namespace["OTAM_dist"](distances)
        self.assertTrue(torch.isfinite(result).all())
        result.sum().backward()
        self.assertTrue(torch.isfinite(distances.grad).all())

    def test_bimhm_is_order_invariant_but_otam_distinguishes_reversal(self):
        model = self.make_model()
        support = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]
        )
        labels = torch.tensor([0])
        ordered_bimhm = model._bimhm_class_dist(support, support, labels)
        reversed_bimhm = model._bimhm_class_dist(
            support, support.flip(1), labels
        )
        torch.testing.assert_close(ordered_bimhm, reversed_bimhm)

        ordered_otam = model._temporal_alignment_class_dist(
            support, support, labels
        )
        reversed_otam = model._temporal_alignment_class_dist(
            support, support.flip(1), labels
        )
        self.assertGreater(
            (ordered_otam - reversed_otam).abs().item(), 1e-4
        )

    def test_temporal_alignment_backpropagates(self):
        model = self.make_model()
        support = torch.randn(4, 3, 6, requires_grad=True)
        query = torch.randn(2, 3, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        distances = model._temporal_alignment_class_dist(
            support, query, labels
        )
        self.assertEqual(tuple(distances.shape), (2, 2))
        distances.sum().backward()
        self.assertTrue(torch.isfinite(support.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == "__main__":
    unittest.main()
