#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""
Frame Selector training model.

Wraps a frozen CLIP backbone and a trainable OTAMFrameSelector.
Used by runs/train_frame_selector.py for standalone pretraining.

馃敡 鐗瑰緛缂撳瓨: 鏂板 forward_from_feats() 鎺ュ彛锛岀洿鎺ヨ緭鍏ラ鎻愬彇鐗瑰緛锛?
   璺宠繃 CLIP 涓诲共鎺ㄧ悊锛岃繑鍥炰笌 forward() 瀹屽叏涓€鑷寸殑杈撳嚭鏍煎紡銆?

Architecture:
    videos (B,T,3,H,W)
      鈫?CLIP backbone (frozen, FP16) 鈫?(B*T, D)
      鈫?reshape 鈫?(B, T, D)
      鈫?OTAMFrameSelector (trainable, FP32) 鈫?(B, K, D) + indices

    鎴栫洿鎺ヨ緭鍏?(B, T, D) 棰勬彁鍙栫壒寰侊紙缂撳瓨妯″紡锛?
      鈫?OTAMFrameSelector (trainable, FP32) 鈫?(B, K, D) + indices
"""

import torch
import torch.nn as nn

from models.frame_selector.otam_frame_selector import (
    OTAMFrameSelector,
    get_feat_dim_from_backbone,
)
from models.frame_selector.bimhm_frame_selector import BiMHMFrameSelector
import clip


class FrameSelectorModel(nn.Module):
    """
    Standalone model for frame selector pretraining.

    Args:
        cfg: Config object with FRAME_SELECTOR settings.
    """
    def __init__(self, cfg):
        super().__init__()
        fs_cfg = cfg.FRAME_SELECTOR

        # 鈹€鈹€ Resolve feature dimension 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        backbone_name = fs_cfg.BACKBONE_NAME
        feat_dim = fs_cfg.FEAT_DIM if fs_cfg.FEAT_DIM > 0 else get_feat_dim_from_backbone(backbone_name)
        self.total_frames = fs_cfg.TOTAL_FRAMES
        self.select_frames = fs_cfg.SELECT_FRAMES

        # 鈹€鈹€ Load CLIP backbone (frozen) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        clip_model, self.preprocess = clip.load(backbone_name, device="cpu")
        self.backbone = clip_model.visual
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        print(f"[FrameSelectorModel] Backbone: {backbone_name}, feat_dim={feat_dim} (frozen, FP16)")

        # 鈹€鈹€ Frame Selector (trainable, FP32) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        selector_type = str(getattr(fs_cfg, "TYPE", "otam")).lower()
        selector_cls = (
            BiMHMFrameSelector
            if selector_type in {"bimhm", "bi_mhm", "d2st_bimhm"}
            else OTAMFrameSelector
        )
        self.selector = selector_cls(
            feat_dim=feat_dim,
            total_frames=self.total_frames,
            select_frames=self.select_frames,
            segments=fs_cfg.SEGMENTS,
        )
        print(f"[FrameSelectorModel] Selector type={selector_type}: "
              f"T={self.total_frames}鈫扠={self.select_frames}, "
              f"segments={fs_cfg.SEGMENTS}, params={sum(p.numel() for p in self.selector.parameters())}")

        # FP16 autocast context (鍏煎鏂版棫 PyTorch API)
        self._autocast = None
        if torch.cuda.is_available():
            _v = torch.__version__.split('+')[0].split('.')[:2]
            _ver = tuple(int(x) for x in _v)
            if _ver >= (2, 0):
                self._autocast = torch.amp.autocast(device_type='cuda', enabled=True)
            else:
                self._autocast = torch.cuda.amp.autocast(enabled=True)

    def forward(self, videos):
        """
        鍘熷鍓嶅悜锛氳緭鍏ュ師濮嬭棰?鈫?CLIP 鈫?鎶藉抚鍣ㄣ€?

        Args:
            videos: (B, T, 3, H, W) raw video frames
        Returns:
            selected_feats:  (B, K, D)
            selected_indices: (B, K)
            frame_scores:    (B, T)
            all_feats:       (B, T, D) 鈥?all frame features (no grad)
        """
        B, T, C, H, W = videos.shape
        assert T == self.total_frames, \
            f"Expected {self.total_frames} frames, got {T}"

        # Flatten batch x temporal
        videos_flat = videos.view(B * T, C, H, W)

        # CLIP 涓诲共 FP16 鎺ㄧ悊锛坒rozen锛屼笉褰卞搷璁粌绮惧害锛?
        with torch.no_grad():
            if self._autocast is not None:
                with self._autocast:
                    feats = self.backbone(videos_flat)
            else:
                feats = self.backbone(videos_flat)
        feats = feats.float().view(B, T, -1)  # (B, T, D)

        # 鎶藉抚鍣?
        selected_feats, indices, scores = self.selector.select_features(
            feats, return_scores=True
        )

        return selected_feats, indices, scores, feats

    # 馃敡 鐗瑰緛缂撳瓨锛氭柊澧炴帴鍙ｏ紝鐩存帴杈撳叆棰勬彁鍙栫壒寰侊紝璺宠繃 CLIP
    def forward_from_feats(self, all_feats):
        """
        缂撳瓨妯″紡鍓嶅悜锛氳緭鍏ラ鎻愬彇鐗瑰緛 鈫?鎶藉抚鍣紝璺宠繃 CLIP 涓诲共銆?

        涓?forward() 杩斿洖瀹屽叏涓€鑷寸殑杈撳嚭鏍煎紡锛岀‘淇濊缁冭剼鏈彲鏃犵紳鍒囨崲銆?

        Args:
            all_feats: (B, T, D) pre-extracted CLIP features
        Returns:
            selected_feats:  (B, K, D)
            selected_indices: (B, K)
            frame_scores:    (B, T)
            all_feats:       (B, T, D) 鈥?same as input
        """
        selected_feats, indices, scores = self.selector.select_features(
            all_feats, return_scores=True
        )
        return selected_feats, indices, scores, all_feats

