#!/usr/bin/env python3

"""Phase-offset Bi-MHM frame selector for D2ST.

This selector is intentionally lightweight: uniform sampling provides a stable
8-phase temporal skeleton, and the learnable scorer only chooses a small local
offset around each phase anchor.  It is designed for the dilemma where a fully
learned selector becomes another 32-frame recognizer, while a pure MLP top-k
selector is too weak to preserve action phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.frame_selector.otam_frame_selector import OTAMFrameSelector
from models.frame_selector.bimhm_frame_selector import bimhm_loss
from models.frame_selector.pairwise_diverse_frame_selector import (
    coverage_reconstruction_loss,
)


PHASE_OFFSET_TYPES = {
    "phase_offset",
    "phase_offset_bimhm",
    "anchor_offset_bimhm",
    "d2st_phase_offset",
}


class PhaseOffsetFrameSelector(OTAMFrameSelector):
    """Uniform phase anchors with learnable local offsets.

    ``SELECT_FRAMES`` defines the number of phase slots.  For T=32,K=8, the
    default windows are 8 local regions of 4 frames.  Each slot selects exactly
    one frame from its local region, keeping stable temporal coverage while
    allowing data-driven adjustment.
    """

    def __init__(
        self,
        feat_dim=512,
        total_frames=32,
        select_frames=8,
        segments=8,
        window_size=4,
        offset_range=2,
        center_bias=0.05,
    ):
        # Use one segment per selected phase slot so the inherited attributes
        # K/per_seg/checkpoint interface remain compatible.
        super().__init__(
            feat_dim=feat_dim,
            total_frames=total_frames,
            select_frames=select_frames,
            segments=select_frames,
        )
        self.window_size = max(1, int(window_size))
        self.offset_range = max(0, int(offset_range))
        self.center_bias = float(center_bias)
        self.register_buffer("anchors", self._build_anchors(), persistent=False)

    def _build_anchors(self):
        anchors = []
        for i in range(self.K):
            start = int(round(i * self.T / self.K))
            end = int(round((i + 1) * self.T / self.K))
            end = max(end, start + 1)
            anchors.append(min(self.T - 1, (start + end - 1) // 2))
        return torch.tensor(anchors, dtype=torch.long)

    def _slot_indices(self, slot, device):
        anchor = int(self.anchors[slot].item())
        half = max(self.window_size // 2, self.offset_range)
        start = max(0, anchor - half)
        end = min(self.T, anchor + half + 1)
        if end - start < self.window_size:
            deficit = self.window_size - (end - start)
            start = max(0, start - deficit)
            end = min(self.T, end + deficit)
        return torch.arange(start, end, device=device, dtype=torch.long), anchor

    def _center_prior(self, candidates, anchor, dtype):
        if self.center_bias <= 0:
            return torch.zeros_like(candidates, dtype=dtype)
        dist = (candidates.float() - float(anchor)).abs()
        scale = max(float(self.offset_range), 1.0)
        # Mildly prefer the uniform anchor unless the learned score says a
        # nearby frame is better.  This keeps the selector close to uniform.
        return (-self.center_bias * dist / scale).to(dtype)

    def forward(self, video_feats, return_scores=False):
        B, T, _ = video_feats.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} frames, got {T}")

        score_dtype = next(self.score_net.parameters()).dtype
        video_feats = torch.nan_to_num(
            video_feats.to(dtype=score_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        frame_scores = self.score_net(video_feats).squeeze(-1)
        frame_scores = torch.nan_to_num(frame_scores, nan=0.0, posinf=0.0, neginf=0.0)

        selected = []
        for slot in range(self.K):
            candidates, anchor = self._slot_indices(slot, video_feats.device)
            slot_scores = frame_scores[:, candidates]
            slot_scores = slot_scores + self._center_prior(candidates, anchor, slot_scores.dtype)
            best = slot_scores.argmax(dim=-1)
            selected.append(candidates[best])

        selected_indices = torch.stack(selected, dim=1)
        selected_indices, _ = selected_indices.sort(dim=-1)
        if return_scores:
            return selected_indices, frame_scores
        return selected_indices

    def extra_repr(self):
        return (
            f"T={self.T}, K={self.K}, phase_slots={self.K}, "
            f"window_size={self.window_size}, offset_range={self.offset_range}, "
            f"center_bias={self.center_bias}"
        )


def phase_offset_bimhm_loss(
    selected_feats,
    labels,
    frame_scores=None,
    all_feats=None,
    margin=0.3,
    temperature=1.0,
    mode="triplet",
    triplet_weight=1.0,
    class_ce_weight=1.0,
    match_weight=0.0,
    normalize_by_frames=False,
    match_normalize_targets=True,
    coverage_recon_weight=0.05,
):
    """Bi-MHM-first loss for phase-offset selection.

    The default is deliberately simpler than pairwise-diverse: triplet dominates
    and coverage is a small regularizer.  Uniform phase slots already enforce
    temporal diversity, so no explicit diversity penalty is needed.
    """
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
    cov = coverage_reconstruction_loss(selected_feats, all_feats)
    total = loss_dict["loss_bimhm_total"] + float(coverage_recon_weight) * cov
    loss_dict["loss_coverage_recon"] = cov
    loss_dict["loss_phase_offset_total"] = total
    return loss_dict


def ste_select_features_phase_offset(selector, frame_scores, feats, tau=0.5):
    """STE selection whose hard path matches phase-offset inference."""
    B, T, D = feats.shape
    feats = torch.nan_to_num(feats.float(), nan=0.0, posinf=0.0, neginf=0.0)
    frame_scores = torch.nan_to_num(frame_scores, nan=0.0, posinf=0.0, neginf=0.0)

    sel_list = []
    idx_list = []
    for slot in range(selector.K):
        candidates, anchor = selector._slot_indices(slot, feats.device)
        slot_scores = frame_scores[:, candidates]
        slot_scores = slot_scores + selector._center_prior(candidates, anchor, slot_scores.dtype)
        hard_pos = slot_scores.argmax(dim=-1)
        hard_idx = candidates[hard_pos]
        hard = feats[torch.arange(B, device=feats.device), hard_idx]

        weights = F.softmax(slot_scores / max(float(tau), 1e-6), dim=-1)
        soft = torch.bmm(weights.unsqueeze(1), feats[:, candidates, :]).squeeze(1)
        sel_list.append(hard + (soft - soft.detach()))
        idx_list.append(hard_idx)

    selected_feats = torch.stack(sel_list, dim=1)
    selected_indices = torch.stack(idx_list, dim=1)
    sorted_order = selected_indices.argsort(dim=-1)
    batch = torch.arange(B, device=feats.device).unsqueeze(1).expand(-1, selector.K)
    return selected_feats[batch, sorted_order], selected_indices[batch, sorted_order]
