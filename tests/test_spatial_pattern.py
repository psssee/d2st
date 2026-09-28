"""CPU regression tests; no CLIP download or video decoding dependencies.

Load the production matcher, fusion forward, and training checks via AST.
The ViT encoder is replaced with supplied features, so this is not an
end-to-end backbone test. Run: python -m unittest discover -s tests -v
"""

import ast
import logging
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import torch
import yaml
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.spatial_pattern_diagnostics import SpatialPatternMeter


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if getattr(node, 'name', '') in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def rearrange_frames(tensor, pattern):
    if pattern != 'q s t c -> q (s t) c':
        raise AssertionError(f'Unexpected feature-only rearrange: {pattern}')
    return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])


adapter_namespace = load_definitions(
    ROOT / 'models/base/adapter.py',
    {'D2STSpatialPatternMatcher', 'ViT_CLIP', 'D2STTaskAwareMatcher',
     'D2STMultiVelocityMatcher', 'OTAM_dist'},
    {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'rearrange': rearrange_frames},
)
Matcher = adapter_namespace['D2STSpatialPatternMatcher']
ViT = adapter_namespace['ViT_CLIP']
training_namespace = load_definitions(
    ROOT / 'runs/train_net_few_shot.py',
    {'_validate_spatial_pattern_training'},
    {'torch': torch, 'logger': logging.getLogger(__name__)},
)
validate_training = training_namespace['_validate_spatial_pattern_training']


def apply_production_freeze_policy(model):
    tree = ast.parse((ROOT / 'runs/train_net_few_shot.py').read_text(encoding='utf-8'))
    train = next(node for node in tree.body if getattr(node, 'name', '') == 'train_few_shot')
    index = next(i for i, node in enumerate(train.body)
                 if isinstance(node, ast.Assign)
                 and any(getattr(target, 'id', '') == 'trainable_markers'
                         for target in node.targets))
    nodes = train.body[index:index + 2]
    assert isinstance(nodes[1], ast.For)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<freeze-policy>', 'exec'),
         {'model': model})


def make_feature_model(detach=True, enabled=True):
    model = ViT.__new__(ViT)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(ADAPTER=SimpleNamespace(WIDTH=6), TRAIN=SimpleNamespace())
    model.num_frames = 3
    model.focus_enable = False
    model.task_match_enable = False
    model.multi_velocity_enable = False
    model.proto_calib_enable = False
    model.spatial_pattern_enable = enabled
    model.spatial_pattern_detach_input = detach
    model.spatial_pattern_matcher = Matcher(3, grid_size=2)
    model.spatial_pattern_alpha = nn.Parameter(torch.tensor(-4.0))
    model.register_buffer('spatial_pattern_logit_delta', torch.tensor(0.0), persistent=False)
    model.conv1 = nn.Linear(6, 6)

    def get_feat(self, supplied_features, return_patch_tokens=False):
        return supplied_features if return_patch_tokens else supplied_features[0]

    model.get_feat = MethodType(get_feat, model)
    return model


def episode(shots=1):
    support = (torch.randn(2 * shots * 3, 6, requires_grad=True),
               torch.randn(2 * shots * 3, 16, 6, requires_grad=True))
    query = (torch.randn(3 * 3, 6, requires_grad=True),
             torch.randn(3 * 3, 16, 6, requires_grad=True))
    return {'support_set': support, 'target_set': query,
            'support_labels': torch.arange(2).repeat_interleave(shots)}


class SpatialPatternTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)

    def test_region_pooling_matches_quadrants(self):
        patches = torch.arange(16.0).reshape(1, 1, 16, 1)
        pooled = Matcher(1)._pool_spatial_patterns(patches).flatten()
        torch.testing.assert_close(pooled, torch.tensor([2.5, 4.5, 10.5, 12.5]))

    def test_matching_axes_and_scale_against_reference(self):
        matcher = Matcher(3)
        support, query = torch.randn(4, 3, 16, 6), torch.randn(2, 3, 16, 6)
        labels = torch.tensor([7, 2, 7, 2])
        scores = matcher(support, query, labels)
        support_patterns = matcher._pool_spatial_patterns(support)
        query_patterns = F.normalize(matcher._pool_spatial_patterns(query), dim=-1)
        prototypes = F.normalize(torch.stack([
            support_patterns[labels == label].mean(0) for label in [2, 7]
        ]), dim=-1)
        expected = torch.zeros(2, 2)
        for q in range(2):
            for c in range(2):
                distances = torch.zeros(3, 3)
                for t in range(3):
                    for u in range(3):
                        region = 1.0 - query_patterns[q, t] @ prototypes[c, u].T
                        distances[t, u] = (region.min(0).values.mean()
                                           + region.min(1).values.mean()) / 2
                expected[q, c] = -(distances.min(0).values.sum()
                                    + distances.min(1).values.sum())
        torch.testing.assert_close(scores, expected)
        self.assertTrue(torch.isfinite(scores).all())

    def test_identical_tokens_and_repeated_shots(self):
        matcher = Matcher(3)
        support, query = torch.randn(2, 3, 16, 6), torch.randn(3, 3, 16, 6)
        labels = torch.tensor([2, 7])
        one_shot = matcher(support, query, labels)
        five_shot = matcher(support.repeat_interleave(5, 0), query,
                            labels.repeat_interleave(5))
        torch.testing.assert_close(one_shot, five_shot)
        scores = matcher(support[:1], support[:1], labels[:1])
        torch.testing.assert_close(scores, torch.zeros_like(scores), atol=2e-6, rtol=0)

    def test_rejects_bad_patch_grid(self):
        with self.assertRaisesRegex(ValueError, 'square patch grid'):
            Matcher(3)._pool_spatial_patterns(torch.randn(2, 3, 15, 6))

    def test_frozen_gate_regression_and_optimizer_step(self):
        model = make_feature_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        apply_production_freeze_policy(model)
        self.assertTrue(model.spatial_pattern_alpha.requires_grad)
        self.assertFalse(model.conv1.weight.requires_grad)
        validate_training(model, optimizer)
        inputs = episode(shots=5)
        before = model.spatial_pattern_alpha.detach().clone()
        outputs = model(inputs)
        F.cross_entropy(outputs['logits'], torch.tensor([0, 1, 0])).backward()
        validate_training(model, optimizer, check_gradient=True)
        self.assertGreater(model.spatial_pattern_alpha.grad.abs().item(), 0.0)
        self.assertIsNone(inputs['support_set'][1].grad)
        self.assertIsNone(inputs['target_set'][1].grad)
        self.assertIsNotNone(inputs['support_set'][0].grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.spatial_pattern_alpha.detach()))

    def test_startup_check_rejects_frozen_or_excluded_gate(self):
        model = make_feature_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.spatial_pattern_alpha.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, 'must be trainable'):
            validate_training(model, optimizer)
        model.spatial_pattern_alpha.requires_grad_(True)
        optimizer = torch.optim.SGD(model.conv1.parameters(), lr=0.1)
        with self.assertRaisesRegex(RuntimeError, 'included in the optimizer'):
            validate_training(model, optimizer)

    def test_missing_gradient_fails(self):
        model = make_feature_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(RuntimeError, 'no finite gradient'):
            validate_training(model, optimizer, check_gradient=True)

    def test_optional_patch_gradient(self):
        model = make_feature_model(detach=False)
        inputs = episode()
        F.cross_entropy(model(inputs)['logits'], torch.tensor([0, 1, 0])).backward()
        self.assertTrue(torch.isfinite(inputs['target_set'][1].grad).all())
        self.assertGreater(inputs['target_set'][1].grad.abs().sum().item(), 0)

    def test_disabled_branch_and_paired_outputs(self):
        model = make_feature_model().eval()
        inputs = episode()
        enabled = model(inputs)
        model.spatial_pattern_enable = False
        disabled = model(inputs)
        torch.testing.assert_close(enabled['logits_without_spatial'], disabled['logits'])
        self.assertNotIn('logits_without_spatial', disabled)
        alpha = torch.sigmoid(model.spatial_pattern_alpha)
        expected = disabled['logits'] + alpha * (enabled['spatial_pattern_logits']
                                                - disabled['logits'].detach())
        torch.testing.assert_close(enabled['logits'], expected)

    def test_combined_task_and_velocity_fusion(self):
        for shots in [1, 5]:
            with self.subTest(shots=shots):
                model = make_feature_model()
                model.task_match_enable = True
                model.task_match_use_focus = False
                model.task_matcher = adapter_namespace['D2STTaskAwareMatcher'](6, 3)
                model.task_match_alpha = nn.Parameter(torch.tensor(-4.0))
                model.multi_velocity_enable = True
                model.multi_velocity_matcher = adapter_namespace['D2STMultiVelocityMatcher'](6, 3)
                model.multi_velocity_alpha = nn.Parameter(torch.tensor(-4.0))
                model.register_buffer('multi_velocity_logit_delta', torch.tensor(0.0), persistent=False)
                inputs = episode(shots)
                outputs = model(inputs)
                self.assertEqual(outputs['logits'].shape, (3, 2))
                F.cross_entropy(outputs['logits'], torch.tensor([0, 1, 0])).backward()
                for name in ['spatial_pattern_alpha', 'task_match_alpha', 'multi_velocity_alpha']:
                    grad = getattr(model, name).grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                model.eval()
                enabled = model(inputs)
                model.spatial_pattern_enable = False
                without = model(inputs)
                torch.testing.assert_close(enabled['logits_without_spatial'], without['logits'])

    def test_nonfinite_gate_gradient_fails(self):
        model = make_feature_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.spatial_pattern_alpha.grad = torch.tensor(float('nan'))
        with self.assertRaisesRegex(RuntimeError, 'no finite gradient'):
            validate_training(model, optimizer, check_gradient=True)

    def test_run_configs_preserve_hyperparameters(self):
        config_dir = ROOT / 'config/ssv2_full'
        for name in ['ViT_SSv2_full_SPATIAL_PATTERN',
                     'ViT_SSv2_full_TASK_MATCH_MULTI_VELOCITY_SPATIAL_PATTERN']:
            original = yaml.safe_load((config_dir / (name + '.yaml')).read_text())
            for suffix in ['GATE_FIX', 'AUDIT']:
                changed = yaml.safe_load((config_dir / (name + '_' + suffix + '.yaml')).read_text())
                self.assertEqual(changed.pop('OUTPUT_DIR'), original['OUTPUT_DIR'] + '_' + suffix)
                if suffix == 'AUDIT':
                    self.assertIs(changed['TRAIN']['ENABLE'], False)
                    self.assertEqual(changed['TEST'].pop('CHECKPOINT_FILE_PATH'),
                                     original['OUTPUT_DIR'] + '/checkpoints/checkpoint_best.pyth')
                    changed['TRAIN']['ENABLE'] = True
                expected = {key: value for key, value in original.items() if key != 'OUTPUT_DIR'}
                self.assertEqual(changed, expected)

    def test_paired_meter_counts_and_episode_uncertainty(self):
        meter = SpatialPatternMeter()
        logits = lambda predictions: F.one_hot(torch.tensor(predictions), 2).float()
        labels = torch.tensor([0, 1])
        for full in [[0, 1], [1, 0]]:
            meter.update({'logits': logits(full),
                          'logits_without_spatial': logits([0, 0]),
                          'spatial_pattern_logits': logits([0, 1])}, labels)
        result = meter.summary()
        self.assertEqual(result['full_acc'], 50.0)
        self.assertEqual(result['without_spatial_acc'], 50.0)
        self.assertEqual(result['spatial_only_acc'], 100.0)
        self.assertEqual(result['helped_predictions'], 1)
        self.assertEqual(result['hurt_predictions'], 1)
        self.assertEqual(result['paired_delta_pp'], 0.0)
        self.assertAlmostEqual(result['paired_ci95_halfwidth_pp'], 98.0)
        self.assertEqual(SpatialPatternMeter().summary(), {})


if __name__ == '__main__':
    unittest.main()
