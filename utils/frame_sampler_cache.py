#!/usr/bin/env python3

import sys
import logging as py_logging
import torch
import utils.logging as logging

logger = logging.get_logger(__name__)

_frame_selector_singleton = None
_selection_log_emitted = False


def _emit_info(message):
    if logger.isEnabledFor(py_logging.INFO):
        logger.info(message)
    else:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()


def _use_cuda_selector(fs_cfg):
    return bool(getattr(fs_cfg, "USE_CUDA", False)) and torch.cuda.is_available()


def _selector_spec(fs_cfg):
    from models.frame_selector.otam_frame_selector import OTAMFrameSelector
    from models.frame_selector.bimhm_frame_selector import BiMHMFrameSelector
    from models.frame_selector.pairwise_diverse_frame_selector import (
        PAIRWISE_DIVERSE_TYPES,
        PairwiseDiverseFrameSelector,
    )

    selector_type = str(getattr(fs_cfg, "TYPE", "otam")).lower()
    if selector_type in PAIRWISE_DIVERSE_TYPES:
        return selector_type, PairwiseDiverseFrameSelector, {
            "coverage_weight": float(getattr(fs_cfg, "COVERAGE_WEIGHT", 0.35)),
            "boundary_weight": float(getattr(fs_cfg, "BOUNDARY_WEIGHT", 0.05)),
        }
    if selector_type in {"bimhm", "bi_mhm", "d2st_bimhm"}:
        return selector_type, BiMHMFrameSelector, {}
    return selector_type, OTAMFrameSelector, {}


def get_frame_selector(cfg):
    global _frame_selector_singleton
    if _frame_selector_singleton is not None:
        return _frame_selector_singleton

    fs_cfg = cfg.FRAME_SELECTOR
    feat_dim = fs_cfg.FEAT_DIM if getattr(fs_cfg, "FEAT_DIM", 0) > 0 else 512
    selector_type, selector_cls, selector_kwargs = _selector_spec(fs_cfg)
    _emit_info(
        f"[FrameSamplerCache] Building {selector_cls.__name__}: "
        f"type={selector_type}, T={fs_cfg.TOTAL_FRAMES} -> K={fs_cfg.SELECT_FRAMES}, "
        f"segments={fs_cfg.SEGMENTS}, feat_dim={feat_dim}"
    )

    selector = selector_cls(
        feat_dim=feat_dim,
        total_frames=fs_cfg.TOTAL_FRAMES,
        select_frames=fs_cfg.SELECT_FRAMES,
        segments=fs_cfg.SEGMENTS,
        **selector_kwargs,
    )

    ckpt_path = getattr(fs_cfg, "PRETRAINED_CKPT", "")
    if ckpt_path:
        _emit_info(f"[FrameSamplerCache] Loading pretrained weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        ckpt_selector_type = None
        if isinstance(state, dict):
            ckpt_selector_type = state.get("selector_type")
            if "selector_state_dict" in state:
                state = state["selector_state_dict"]
        if ckpt_selector_type and str(ckpt_selector_type).lower() != selector_type:
            logger.warning(
                "[FrameSamplerCache] checkpoint selector_type=%s but config TYPE=%s; "
                "using config TYPE for inference.",
                ckpt_selector_type,
                selector_type,
            )
        selector.load_state_dict(state)
        _emit_info("[FrameSamplerCache] Weights loaded successfully")
    else:
        logger.warning("[FrameSamplerCache] No PRETRAINED_CKPT specified; using random selector.")

    selector.eval()
    for param in selector.parameters():
        param.requires_grad = False

    if _use_cuda_selector(fs_cfg):
        selector = selector.cuda().half()
        _emit_info("[FrameSamplerCache] Moved to CUDA, FP16")
    else:
        _emit_info("[FrameSamplerCache] Using CPU FP32")

    _frame_selector_singleton = selector
    _emit_info(
        f"[{selector_cls.__name__}] Initialized (cache decode): "
        f"T={fs_cfg.TOTAL_FRAMES} -> K={fs_cfg.SELECT_FRAMES}"
    )
    return _frame_selector_singleton


def get_selected_indices(feat_32, cfg):
    global _selection_log_emitted
    selector = get_frame_selector(cfg)
    if feat_32.dim() == 2:
        feat_32 = feat_32.unsqueeze(0)

    device = next(selector.parameters()).device
    dtype = next(selector.parameters()).dtype
    feat_input = feat_32.to(device=device, dtype=dtype)

    with torch.no_grad():
        _, sel_indices = selector.select_features(feat_input, return_scores=False)

    indices = sel_indices[0].detach().cpu().tolist()
    if not _selection_log_emitted:
        _emit_info(
            f"[{selector.__class__.__name__}] First selection completed: "
            f"input_shape={tuple(feat_32.shape)}, selected_indices={indices}"
        )
        _selection_log_emitted = True
    return indices
