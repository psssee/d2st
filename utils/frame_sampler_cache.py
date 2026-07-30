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


def get_frame_selector(cfg):
    global _frame_selector_singleton
    if _frame_selector_singleton is not None:
        return _frame_selector_singleton

    from models.frame_selector.otam_frame_selector import OTAMFrameSelector

    fs_cfg = cfg.FRAME_SELECTOR
    feat_dim = fs_cfg.FEAT_DIM if getattr(fs_cfg, "FEAT_DIM", 0) > 0 else 512
    _emit_info(
        "[FrameSamplerCache] Building OTAMFrameSelector: "
        f"T={fs_cfg.TOTAL_FRAMES} -> K={fs_cfg.SELECT_FRAMES}, "
        f"segments={fs_cfg.SEGMENTS}, feat_dim={feat_dim}"
    )

    selector = OTAMFrameSelector(
        feat_dim=feat_dim,
        total_frames=fs_cfg.TOTAL_FRAMES,
        select_frames=fs_cfg.SELECT_FRAMES,
        segments=fs_cfg.SEGMENTS,
    )

    ckpt_path = getattr(fs_cfg, "PRETRAINED_CKPT", "")
    if ckpt_path:
        _emit_info(f"[FrameSamplerCache] Loading pretrained weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "selector_state_dict" in state:
            state = state["selector_state_dict"]
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
        "[OTAMSelector] Initialized (cache decode): "
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
            "[OTAMSelector] First selection completed: "
            f"input_shape={tuple(feat_32.shape)}, selected_indices={indices}"
        )
        _selection_log_emitted = True
    return indices
