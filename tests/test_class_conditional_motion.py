"""Regression checks for the joint class-conditional semantic-motion matcher."""

import ast
import math
from pathlib import Path
import unittest

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class DummyTaskMatcher:
    def __init__(self, weights):
        self.weights = weights

    def get_match_weights(self, support, query, labels):
        return self.weights.expand(query.shape[0], -1, -1)


class DummyVelocityMatcher:
    def __init__(self):
        self.velocities = (1, 2)
        self.velocity_logits = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _motion_features(features, velocity):
        return features[:, velocity:] - features[:, :-velocity]


def load_matcher():
    tree = ast.parse((ROOT / "models/base/adapter.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", "") ==
             "D2STClassConditionalSemanticMotionMatcher"]
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<joint-matcher>", "exec"), namespace)
    return namespace["D2STClassConditionalSemanticMotionMatcher"]


Matcher = load_matcher()


class ClassConditionalMotionTests(unittest.TestCase):
    def test_shape_finite_and_motion_mix_gradient(self):
        torch.manual_seed(7)
        matcher = Matcher(dim=6, num_frames=4)
        support = torch.randn(4, 4, 6, requires_grad=True)
        query = torch.randn(3, 4, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        task = DummyTaskMatcher(torch.full((1, 4, 4), 1.0 / 16.0))
        velocity = DummyVelocityMatcher()

        logits = matcher(support, query, labels, task, velocity)
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(matcher.motion_mix_logit.grad)
        self.assertTrue(torch.isfinite(matcher.motion_mix_logit.grad))
        self.assertIsNotNone(matcher.velocity_router[-1].weight.grad)
        self.assertTrue(torch.isfinite(matcher.velocity_router[-1].weight.grad).all())

    def test_task_weights_and_class_order_are_respected(self):
        matcher = Matcher(dim=5, num_frames=3, velocities=(1, 2), motion_mix_init=0.1)
        support = torch.randn(4, 3, 5)
        query = torch.randn(2, 3, 5)
        labels = torch.tensor([8, 8, 3, 3])
        weights = torch.zeros(1, 3, 3)
        weights[:, torch.arange(3), torch.arange(3)] = 1.0 / 3.0
        logits = matcher(
            support, query, labels, DummyTaskMatcher(weights), DummyVelocityMatcher()
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
