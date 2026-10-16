"""Paired evaluation of the spatial branch using the same checkpoint/episodes."""

import math

import torch
import torch.distributed as dist


class SpatialPatternMeter:
    def __init__(self):
        self.totals = None

    @torch.no_grad()
    def update(self, outputs, labels):
        if 'logits_without_spatial' not in outputs:
            return
        labels = labels.reshape(-1).long()
        prediction = outputs['logits'].argmax(dim=-1)
        without_prediction = outputs['logits_without_spatial'].argmax(dim=-1)
        spatial_prediction = outputs['spatial_pattern_logits'].argmax(dim=-1)
        correct = prediction.eq(labels)
        without_correct = without_prediction.eq(labels)
        delta = correct.double().mean() - without_correct.double().mean()
        values = torch.stack([
            correct.sum(), without_correct.sum(), spatial_prediction.eq(labels).sum(),
            labels.new_tensor(labels.numel()), prediction.ne(without_prediction).sum(),
            (correct & ~without_correct).sum(), (~correct & without_correct).sum(),
            delta, delta.square(), delta.new_tensor(1.0),
        ]).to(dtype=torch.float64)
        if self.totals is None:
            self.totals = values
        else:
            self.totals += values

    def summary(self):
        if self.totals is None:
            return {}
        totals = self.totals.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(totals)
        full, without, spatial, count, changed, helped, hurt, delta, delta_sq, n = (
            totals.cpu().tolist()
        )
        mean_delta = delta / n
        # Cluster by episode: queries in the same few-shot task are dependent.
        ci_halfwidth = None
        if n > 1:
            variance = max((delta_sq - delta * delta / n) / (n - 1), 0.0)
            ci_halfwidth = 1.96 * math.sqrt(variance / n) * 100.0
        return {
            'episodes': int(n),
            'queries': int(count),
            'full_acc': 100.0 * full / count,
            'without_spatial_acc': 100.0 * without / count,
            'spatial_only_acc': 100.0 * spatial / count,
            'paired_delta_pp': 100.0 * mean_delta,
            'paired_ci95_halfwidth_pp': ci_halfwidth,
            'changed_predictions': int(changed),
            'helped_predictions': int(helped),
            'hurt_predictions': int(hurt),
        }

    def log(self, logger):
        summary = self.summary()
        if summary:
            logger.info('Spatial pattern paired evaluation: %s', summary)


class RelationalTemporalMeter:
    """Paired evaluation for the relational temporal residual branch."""

    def __init__(self):
        self.totals = None

    @torch.no_grad()
    def update(self, outputs, labels):
        if 'logits_without_relation_match' not in outputs:
            return
        labels = labels.reshape(-1).long()
        full = outputs['logits'].argmax(dim=-1).eq(labels)
        without = outputs['logits_without_relation_match'].argmax(dim=-1).eq(labels)
        relation = outputs['relation_match_logits'].argmax(dim=-1).eq(labels)
        values = torch.stack([
            full.sum(), without.sum(), relation.sum(),
            labels.new_tensor(labels.numel()), full.ne(without).sum(),
            (full & ~without).sum(), (~full & without).sum(),
        ]).to(dtype=torch.float64)
        self.totals = values if self.totals is None else self.totals + values

    def summary(self):
        if self.totals is None:
            return {}
        full, without, relation, count, changed, helped, hurt = self.totals.cpu().tolist()
        return {
            'queries': int(count),
            'full_acc': 100.0 * full / count,
            'without_relation_acc': 100.0 * without / count,
            'relation_only_acc': 100.0 * relation / count,
            'paired_delta_pp': 100.0 * (full - without) / count,
            'changed_predictions': int(changed),
            'helped_predictions': int(helped),
            'hurt_predictions': int(hurt),
        }

    def log(self, logger):
        summary = self.summary()
        if summary:
            logger.info('Relational temporal paired evaluation: %s', summary)
