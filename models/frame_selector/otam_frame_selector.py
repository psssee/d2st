#!/usr/bin/env python3

import torch
import torch.nn as nn


class OTAMFrameSelector(nn.Module):
    """Lightweight frame selector used by BDTS cached decoding.

    This is the inference-only architecture that matches the selector checkpoint
    trained in the CLIP-FSAR project.  It scores T cached CLIP frame features and
    selects K temporally sorted frame indices by segment-wise top-k.
    """

    def __init__(self, feat_dim=512, total_frames=32, select_frames=8, segments=4):
        super().__init__()
        self.T = int(total_frames)
        self.K = int(select_frames)
        self.segments = int(segments)

        assert self.T % self.segments == 0
        assert self.K % self.segments == 0
        self.per_seg = self.K // self.segments
        self.segment_size = self.T // self.segments

        self.score_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.score_net:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, video_feats, return_scores=False):
        if video_feats.dim() != 3:
            raise ValueError(f"Expected (B,T,D) features, got {tuple(video_feats.shape)}")
        batch_size, num_frames, _ = video_feats.shape
        if num_frames != self.T:
            raise ValueError(f"Expected {self.T} frames, got {num_frames}")

        frame_scores = self.score_net(video_feats).squeeze(-1)
        selected_indices = []
        for seg_id in range(self.segments):
            start = seg_id * self.segment_size
            end = (seg_id + 1) * self.segment_size
            seg_scores = frame_scores[:, start:end]
            _, seg_idx = torch.topk(seg_scores, self.per_seg, dim=-1)
            selected_indices.append(seg_idx + start)

        selected_indices = torch.cat(selected_indices, dim=-1)
        selected_indices, _ = selected_indices.sort(dim=-1)
        if return_scores:
            return selected_indices, frame_scores
        return selected_indices

    def select_features(self, video_feats, return_scores=True):
        indices, scores = self.forward(video_feats, return_scores=True)
        batch_idx = torch.arange(video_feats.shape[0], device=video_feats.device).unsqueeze(1)
        selected_feats = video_feats[batch_idx, indices]
        if return_scores:
            return selected_feats, indices, scores
        return selected_feats, indices
