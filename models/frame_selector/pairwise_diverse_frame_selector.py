#!/usr/bin/env python3

"""Pairwise-diverse Bi-MHM frame selector for D2ST.

D2ST classifies a query by comparing every query frame with every support
frame and then summing bidirectional minimum matching distances.  For that
head, selecting only the locally highest-score frames is often brittle: two
selected frames can explain the same action phase, leaving the min-matching
head with weak coverage for the remaining phases.

This module keeps the old segment-wise selector/checkpoint contract, but uses
an MMR-style inference rule inside each segment:

    utility(frame) = learned_score(frame)
                   + coverage_weight * distance_to_selected_frames
                   + boundary_weight * segment_boundary_bonus

The training loss remains Bi-MHM-first and adds small set-level terms that
encourage the selected frame set to be non-redundant and still cover the 32-frame
clip.  It is intentionally conservative so it can be compared against the
existing ``bimhm`` selector through config only.
"""

import torch
import torch.nn.functional as F

from models.frame_selector.bimhm_frame_selector import (
    BiMHMFrameSelector,
    bimhm_loss,
)


PAIRWISE_DIVERSE_TYPES = {
    "pairwise_diverse",
    "pairwise_diverse_bimhm",
    "diverse_bimhm",
    "d2st_pairwise_diverse",
}


class PairwiseDiverseFrameSelector(BiMHMFrameSelector):
    """Segment-wise selector with score + feature-diversity selection.

    The parameterized part is still only ``score_net`` from the base selector,
    so checkpoints remain lightweight.  ``coverage_weight`` and
    ``boundary_weight`` are config-side inference hyperparameters.
    """

    def __init__(
        self,
        feat_dim=512,
        total_frames=32,
        select_frames=8,
        segments=4,
        coverage_weight=0.35,
        boundary_weight=0.05,
    ):
        super().__init__(
            feat_dim=feat_dim,
            total_frames=total_frames,
            select_frames=select_frames,
            segments=segments,
        )
        self.coverage_weight = float(coverage_weight)
        self.boundary_weight = float(boundary_weight)

    def _segment_boundary_bonus(self, seg_len, device, dtype):
        if seg_len <= 1 or self.boundary_weight <= 0:
            return torch.zeros(seg_len, device=device, dtype=dtype)
        pos = torch.linspace(0, 1, seg_len, device=device, dtype=dtype)
        # SSv2 is phase-sensitive; a tiny endpoint prior helps avoid selecting
        # only the most static middle frames while segment partitioning still
        # preserves global order.
        bonus = torch.maximum(pos, 1.0 - pos)
        return self.boundary_weight * bonus

    def _select_segment(self, seg_feats, seg_scores, start):
        """Greedy MMR-like selection within one temporal segment."""
        B, seg_len, _ = seg_feats.shape
        device = seg_feats.device
        dtype = seg_scores.dtype
        norm_feats = F.normalize(seg_feats.float(), dim=-1)
        sim = torch.bmm(norm_feats, norm_feats.transpose(1, 2))
        dist = (1.0 - sim).to(dtype)

        boundary_bonus = self._segment_boundary_bonus(seg_len, device, dtype)
        selected = []
        selected_mask = torch.zeros(B, seg_len, device=device, dtype=torch.bool)

        for _ in range(self.per_seg):
            if selected:
                selected_idx = torch.stack(selected, dim=1)
                gather_idx = selected_idx.unsqueeze(1).expand(-1, seg_len, -1)
                min_dist = dist.gather(2, gather_idx).min(dim=2).values
            else:
                min_dist = torch.zeros_like(seg_scores)

            utility = seg_scores + self.coverage_weight * min_dist + boundary_bonus
            utility = utility.masked_fill(selected_mask, torch.finfo(dtype).min)
            next_idx = utility.argmax(dim=-1)
            selected.append(next_idx)
            selected_mask[torch.arange(B, device=device), next_idx] = True

        seg_idx = torch.stack(selected, dim=1)
        seg_idx, _ = seg_idx.sort(dim=-1)
        return seg_idx + int(start)

    def forward(self, video_feats, return_scores=False):
        B, T, _ = video_feats.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} frames, got {T}")

        frame_scores = self.score_net(video_feats).squeeze(-1)
        selected_indices = []
        for i in range(self.segments):
            start = i * self.segment_size
            end = (i + 1) * self.segment_size
            selected_indices.append(
                self._select_segment(
                    video_feats[:, start:end],
                    frame_scores[:, start:end],
                    start,
                )
            )

        selected_indices = torch.cat(selected_indices, dim=-1)
        selected_indices, _ = selected_indices.sort(dim=-1)

        if return_scores:
            return selected_indices, frame_scores
        return selected_indices

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", coverage_weight={self.coverage_weight}, "
              f"boundary_weight={self.boundary_weight}"
        )


def selected_diversity_loss(selected_feats, threshold=0.55):
    """Penalize redundant selected frames within each video."""
    B, K, _ = selected_feats.shape
    if K <= 1:
        return selected_feats.sum() * 0.0
    feats = F.normalize(selected_feats, dim=-1)
    sim = torch.bmm(feats, feats.transpose(1, 2))
    eye = torch.eye(K, device=selected_feats.device, dtype=torch.bool)
    off_diag = sim[:, ~eye].view(B, K, K - 1)
    return F.relu(off_diag - float(threshold)).mean()


def coverage_reconstruction_loss(selected_feats, all_feats):
    """Make selected frames cover the original T-frame clip."""
    if all_feats is None:
        return selected_feats.sum() * 0.0
    selected = F.normalize(selected_feats, dim=-1)
    all_norm = F.normalize(all_feats, dim=-1)
    sim = torch.bmm(all_norm, selected.transpose(1, 2))
    nearest_sim = sim.max(dim=-1).values
    return (1.0 - nearest_sim).mean()


def pairwise_diverse_bimhm_loss(
    selected_feats,
    labels,
    frame_scores=None,
    all_feats=None,
    margin=0.3,
    temperature=1.0,
    mode="match_aware",
    triplet_weight=1.0,
    class_ce_weight=1.0,
    match_weight=0.5,
    normalize_by_frames=False,
    match_normalize_targets=True,
    diversity_weight=0.15,
    coverage_recon_weight=0.10,
    diversity_threshold=0.55,
):
    """Bi-MHM loss plus conservative set-level diversity/coverage terms."""
    loss_dict = bimhm_loss(
        selected_feats=selected_feats,
        labels=labels,
        frame_scores=frame_scores,
        all_feats=all_feats,
        margin=margin,
        temperature=temperature,
        mode=mode,
        triplet_weight=triplet_weight,
        class_ce_weight=class_ce_weight,
        match_weight=match_weight,
        normalize_by_frames=normalize_by_frames,
        match_normalize_targets=match_normalize_targets,
    )

    div = selected_diversity_loss(
        selected_feats,
        threshold=diversity_threshold,
    )
    cov = coverage_reconstruction_loss(selected_feats, all_feats)
    total = (
        loss_dict["loss_bimhm_total"]
        + float(diversity_weight) * div
        + float(coverage_recon_weight) * cov
    )

    loss_dict["loss_pairwise_diversity"] = div
    loss_dict["loss_coverage_recon"] = cov
    loss_dict["loss_pairwise_bimhm_total"] = total
    return loss_dict


def ste_select_features_pairwise_diverse(selector, frame_scores, feats, tau=0.5):
    """STE selection whose forward path matches pairwise-diverse inference."""
    B, T, D = feats.shape
    K = selector.K
    sel_list, idx_list = [], []

    for s in range(selector.segments):
        start = s * selector.segment_size
        end = (s + 1) * selector.segment_size
        seg_sc = frame_scores[:, start:end]
        seg_fe = feats[:, start:end, :]

        hard_idx_global = selector._select_segment(seg_fe, seg_sc, start)
        hard_idx = hard_idx_global - start
        bidx = torch.arange(B, device=feats.device).unsqueeze(1).expand(
            -1, selector.per_seg
        )
        hard = seg_fe[bidx, hard_idx]

        sm = F.softmax(seg_sc / tau, dim=-1)
        sw, si = sm.sort(dim=-1, descending=True)
        topk_w = sw[:, :selector.per_seg]
        topk_f = seg_fe.gather(
            1,
            si[:, :selector.per_seg].unsqueeze(-1).expand(-1, -1, D),
        )
        soft = topk_w.unsqueeze(-1) * topk_f

        sel_list.append(hard + (soft - soft.detach()))
        idx_list.append(hard_idx_global)

    all_idx = torch.cat(idx_list, dim=-1)
    sorted_idx = all_idx.argsort(dim=-1)
    batch_perm = torch.arange(B, device=feats.device).unsqueeze(1).expand(-1, K)
    all_feats = torch.cat(sel_list, dim=1)

    return all_feats[batch_perm, sorted_idx], all_idx[batch_perm, sorted_idx]
