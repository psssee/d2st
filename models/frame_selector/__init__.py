#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""Frame selector package for learnable frame selection."""

from .otam_frame_selector import (
    OTAMFrameSelector,
    otam_triplet_loss,
    ste_select_features,
    get_feat_dim_from_backbone,
)
from .bimhm_frame_selector import (
    BiMHMFrameSelector,
    bimhm_distance,
    bimhm_loss,
)
from .pairwise_diverse_frame_selector import (
    PAIRWISE_DIVERSE_TYPES,
    PairwiseDiverseFrameSelector,
    pairwise_diverse_bimhm_loss,
    ste_select_features_pairwise_diverse,
)

__all__ = [
    "OTAMFrameSelector",
    "otam_triplet_loss",
    "ste_select_features",
    "get_feat_dim_from_backbone",
    "BiMHMFrameSelector",
    "bimhm_distance",
    "bimhm_loss",
    "PAIRWISE_DIVERSE_TYPES",
    "PairwiseDiverseFrameSelector",
    "pairwise_diverse_bimhm_loss",
    "ste_select_features_pairwise_diverse",
]


