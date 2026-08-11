#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""
OTAM-driven Frame Selector pretraining script.

Trains OTAMFrameSelector using OTAM optimal transport distance as the
core training signal with Straight-Through Estimator (STE).

🔧 特征缓存: 支持 FEAT_CACHE.ENABLE 配置开关。
   开启后数据集直接加载预提取的 CLIP 特征张量，
   模型调用 forward_from_feats() 跳过 CLIP 主干推理，
   训练速度提升 10× 以上。

Architecture:
    Video (B,T,3,H,W)
      -> CLIP (frozen, eval, FP16) -> (B*T, D)
      -> score_net -> scores -> per-segment STE -> selected (B, K, D)
      -> OTAM triplet loss (可微分，唯一核心损失) -> train score_net

    🔧 缓存模式:
      Pre-extracted feats (B,T,D)
      -> score_net -> scores -> per-segment STE -> selected (B, K, D)
      -> OTAM triplet loss (可微分，唯一核心损失) -> train score_net

Usage:
    # 在线模式（默认）
    CUDA_VISIBLE_DEVICES=0 python runs/train_frame_selector.py \
        --cfg configs/projects/FRAMESELECTOR/ssv2_fs_train.yaml

    # 缓存模式（先提取特征）
    python tools/extract_fs_features.py \
        --cfg configs/projects/FRAMESELECTOR/ssv2_fs_train.yaml
    # 然后设置 FEAT_CACHE.ENABLE: true 再运行训练
"""

import os
import sys
import argparse
import time
import datetime
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

sys.path.append(os.path.abspath(os.curdir))

from models.frame_selector.fs_dataset import build_fs_dataloader, _feat_cache_cfg, _cache_tag
from models.frame_selector.fs_model import FrameSelectorModel
from models.frame_selector.otam_frame_selector import otam_triplet_loss, ste_select_features
from models.frame_selector.bimhm_frame_selector import bimhm_loss
from utils.config import Config
import utils.logging as logging
import utils.misc as misc

logger = logging.get_logger(__name__)


def _compute_selector_loss(selected_feats, labels, frame_scores, hard_indices,
                           all_feats, cfg, margin, index_weight, score_margin_weight):
    """Dispatch the configured selector loss without changing old OTAM mode."""
    loss_name = str(getattr(cfg.LOSS, "NAME", "otam_triplet")).lower()
    if loss_name in {"bimhm", "bimhm_triplet", "bimhm_class_ce",
                     "bimhm_match_aware", "bimhm_combined"}:
        if loss_name == "bimhm_class_ce":
            mode = "class_ce"
        elif loss_name == "bimhm_match_aware":
            mode = "match_aware"
        elif loss_name == "bimhm_combined":
            mode = "combined"
        else:
            mode = "triplet"

        return bimhm_loss(
            selected_feats=selected_feats,
            labels=labels,
            frame_scores=frame_scores,
            all_feats=all_feats,
            margin=margin,
            temperature=float(getattr(cfg.LOSS, "TEMPERATURE", 1.0)),
            mode=mode,
            triplet_weight=float(getattr(cfg.LOSS, "TRIPLET_WEIGHT", 1.0)),
            class_ce_weight=float(getattr(cfg.LOSS, "CLASS_CE_WEIGHT", 1.0)),
            match_weight=float(getattr(cfg.LOSS, "MATCH_WEIGHT", 1.0)),
            match_normalize_targets=bool(
                getattr(cfg.LOSS, "MATCH_NORMALIZE_TARGETS", True)
            ),
            normalize_by_frames=bool(
                getattr(cfg.LOSS, "NORMALIZE_BY_FRAMES", False)
            ),
        )

    return otam_triplet_loss(
        selected_feats=selected_feats,
        labels=labels,
        frame_scores=frame_scores,
        margin=margin,
        index_compact_weight=index_weight,
        score_margin_weight=score_margin_weight,
        indices=hard_indices,
        segments=cfg.FRAME_SELECTOR.SEGMENTS,
        total_frames=cfg.FRAME_SELECTOR.TOTAL_FRAMES,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="OTAM Frame Selector Training")
    parser.add_argument("--cfg", dest="cfg_file", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


class _NoOpSummaryWriter:
    """Fallback writer used when tensorboard is not installed."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(cfg, model, optimizer, epoch, loss, is_best=False):
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "selector_state_dict": model.selector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "selector_type": getattr(cfg.FRAME_SELECTOR, "TYPE", "otam"),
        "loss_name": getattr(cfg.LOSS, "NAME", "otam_triplet"),
        "config": cfg.cfg_dict if hasattr(cfg, "cfg_dict") else {},
    }
    torch.save(ckpt, os.path.join(output_dir, "frame_selector_latest.pth"))
    save_period = getattr(cfg, "SAVE_PERIOD", 5)
    if epoch % save_period == 0:
        p = os.path.join(output_dir, f"frame_selector_epoch{epoch:04d}.pth")
        torch.save(ckpt, p); logger.info(f"Checkpoint: {p}")
    if is_best:
        p = os.path.join(output_dir, "frame_selector_best.pth")
        torch.save(ckpt, p); logger.info(f"Best checkpoint: {p} (loss={loss:.6f})")


def log_iter_stats(epoch, max_epoch, it, max_it, loss, triplet, pos, neg, lr, td, eta):
    logging.log_json_stats({
        "_type": "train_iter",
        "epoch": "{}/{}".format(epoch + 1, max_epoch),
        "iter": "{}/{}".format(it + 1, max_it),
        "time_diff": round(td, 2),
        "eta": str(datetime.timedelta(seconds=int(eta))),
        "loss": round(loss, 6), "triplet": round(triplet, 6),
        "pos_dist": round(pos, 6), "neg_dist": round(neg, 6), "lr": round(lr, 8),
    })


def log_epoch_stats(epoch, max_epoch, td, eta, loss, triplet, pos, neg, lr):
    logging.log_json_stats({
        "_type": "train_epoch",
        "epoch": "{}/{}".format(epoch + 1, max_epoch),
        "time_diff": round(td, 2),
        "eta": str(datetime.timedelta(seconds=int(eta))),
        "lr": round(lr, 8),
        "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        "loss": round(loss, 6), "triplet": round(triplet, 6),
        "pos_dist": round(pos, 6), "neg_dist": round(neg, 6),
    })


def log_val_epoch_stats(epoch, max_epoch, loss):
    logging.log_json_stats({
        "_type": "val_epoch",
        "epoch": "{}/{}".format(epoch + 1, max_epoch),
        "loss": round(loss, 6),
    })


def train():
    args = parse_args()
    cfg = Config(load=True)
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logging.setup_logging(cfg, "train_fs.log")

    logger.info("OTAM Frame Selector Pretraining")
    logger.info(f"Config: {args.cfg_file}")
    logger.info(f"Output: {output_dir}")
    logger.info("Config:\n{}".format(cfg.dump() if hasattr(cfg, "dump") else cfg))

    device = torch.device("cuda" if torch.cuda.is_available() and getattr(cfg, "NUM_GPUS", 1) > 0 else "cpu")
    logger.info(f"Device: {device}")

    # ── 缓存状态 ────────────────────────────────────────────────────
    # 🔧 特征缓存：读取配置开关，打印状态
    cache_info = _feat_cache_cfg(cfg)
    use_cache = cache_info["ENABLE"]
    if use_cache:
        cache_tag = _cache_tag(cfg)
        logger.info(f"🔧 特征缓存已开启: tag={cache_tag}, dir={cache_info['CACHE_DIR']}")
    else:
        logger.info("🔧 特征缓存已关闭（默认），使用在线视频解码 + CLIP")

    # ── Data ────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_loader = build_fs_dataloader(cfg, "train")
    has_val = hasattr(cfg.DATA, "VAL_LIST") and cfg.DATA.VAL_LIST
    val_loader = build_fs_dataloader(cfg, "val") if has_val else None
    logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} iters/epoch")

    # ── Model ───────────────────────────────────────────────────────
    model = FrameSelectorModel(cfg).to(device)
    # CRITICAL: CLIP backbone must stay in eval mode
    model.eval()
    model.selector.train()
    total_params = sum(p.numel() for p in model.selector.parameters())
    logger.info(f"Trainable (selector) params: {total_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────
    optim_cfg = cfg.SOLVER
    optimizer = torch.optim.AdamW(model.selector.parameters(), lr=optim_cfg.BASE_LR,
                                  weight_decay=optim_cfg.WEIGHT_DECAY)
    max_epoch = optim_cfg.MAX_EPOCH
    warmup_epochs = getattr(optim_cfg, "WARMUP_EPOCHS", 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs) \
        if getattr(optim_cfg, "LR_POLICY", "cosine") == "cosine" else None

    # ── Loss config ─────────────────────────────────────────────────
    loss_name = str(getattr(cfg.LOSS, "NAME", "otam_triplet")).lower()
    margin = getattr(cfg.LOSS, "MARGIN", 0.3)
    index_weight = getattr(cfg.LOSS, "INDEX_COMPACT_WEIGHT", 0.05)
    score_margin_weight = getattr(cfg.LOSS, "SCORE_MARGIN_WEIGHT", 0.0)
    ste_tau = getattr(cfg.FRAME_SELECTOR, "STE_TAU", 0.5)
    logger.info(
        f"Selector loss={loss_name}, margin={margin}, "
        f"score_margin_weight={score_margin_weight} (0=disabled), "
        f"index_compact_weight={index_weight}, STE_tau={ste_tau}"
    )

    # ── TensorBoard ────────────────────────────────────────────────
    if SummaryWriter is None:
        logger.warning("tensorboard is not installed; TensorBoard logging is disabled.")
        writer = _NoOpSummaryWriter()
    else:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    log_period = getattr(cfg, "LOG_PERIOD", 50)
    best_val_loss = float("inf")
    global_step = 0
    train_timer = time.time()

    logger.info("=" * 80)
    logger.info(
        f"{loss_name} training | Epochs: {max_epoch} | Log: {log_period} | "
        f"Cache={'ON' if use_cache else 'OFF'}"
    )
    logger.info("=" * 80)

    for epoch in range(max_epoch):
        model.eval()
        model.selector.train()

        epoch_loss = AverageMeter()
        epoch_trip = AverageMeter()
        epoch_pos = AverageMeter()
        epoch_neg = AverageMeter()
        epoch_start = time.time()

        if epoch < warmup_epochs:
            lr = optim_cfg.BASE_LR * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups: pg["lr"] = lr
        elif scheduler: scheduler.step()

        for batch_idx, (data, labels) in enumerate(train_loader):
            labels = labels.to(device, non_blocking=True)
            B = labels.shape[0]

            # ── Forward ────────────────────────────────────────────
            # 🔧 特征缓存：data.dim()==5 → 原始视频，用 model.forward()
            #            data.dim()==3 → 缓存特征，用 model.forward_from_feats()
            if data.dim() == 5:
                data = data.to(device, non_blocking=True)
                _, _, frame_scores, all_feats = model(data)
            else:
                all_feats = data.to(device, non_blocking=True)  # (B,T,D)
                # 🔧 特征缓存：直接调用 forward_from_feats 跳过 CLIP
                _, _, frame_scores, all_feats = model.forward_from_feats(all_feats)

            # ── STE selection + configured selector loss ───────────
            ste_feats, hard_indices = ste_select_features(
                model.selector, frame_scores, all_feats, tau=ste_tau)

            loss_dict = _compute_selector_loss(
                selected_feats=ste_feats, labels=labels,
                frame_scores=frame_scores,
                hard_indices=hard_indices,
                all_feats=all_feats,
                cfg=cfg,
                margin=margin,
                index_weight=index_weight,
                score_margin_weight=score_margin_weight,
            )
            total_key = (
                "loss_bimhm_total"
                if "loss_bimhm_total" in loss_dict
                else "loss_otam_total"
            )
            loss = loss_dict[total_key]

            # ── Backward ──────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()

            # ✅ 梯度自检：首个 iteration
            if epoch == 0 and batch_idx == 0:
                _first_layer = model.selector.score_net[0]
                _grad = _first_layer.weight.grad
                if _grad is None:
                    logger.warning(
                        "⚠️ OTAM 梯度自检: score_net[0].weight.grad = None\n"
                        "   可能 batch 内无同类样本，属正常跳过。"
                    )
                else:
                    _gn = _grad.norm().item()
                    if _gn < 1e-6:
                        logger.warning(f"⚠️ OTAM 梯度异常小: grad.norm()={_gn:.8f}")
                    else:
                        logger.info(f"✅ OTAM 梯度自检通过: grad.norm()={_gn:.6f}")

            nn.utils.clip_grad_norm_(model.selector.parameters(), max_norm=5.0)
            optimizer.step()

            # ── Update meters ─────────────────────────────────────
            lr_now = optimizer.param_groups[0]["lr"]
            epoch_loss.update(loss.item(), B)
            triplet_value = loss_dict.get(
                "loss_bimhm_triplet",
                loss_dict.get("loss_otam_triplet", loss.detach() * 0.0),
            )
            pos_value = loss_dict.get(
                "loss_bimhm_pos_dist",
                loss_dict.get("loss_otam_pos_dist", loss.detach() * 0.0),
            )
            neg_value = loss_dict.get(
                "loss_bimhm_neg_dist",
                loss_dict.get("loss_otam_neg_dist", loss.detach() * 0.0),
            )
            epoch_trip.update(triplet_value.item(), B)
            epoch_pos.update(pos_value.item(), B)
            epoch_neg.update(neg_value.item(), B)

            # ── Log ───────────────────────────────────────────────
            if batch_idx % log_period == 0:
                elapsed = time.time() - train_timer
                done = epoch * len(train_loader) + batch_idx + 1
                total_iters = max_epoch * len(train_loader)
                eta_sec = (elapsed / done) * (total_iters - done) if done else 0

                log_iter_stats(epoch, max_epoch, batch_idx, len(train_loader),
                               loss.item(), epoch_trip.val, epoch_pos.val, epoch_neg.val,
                               lr_now, time.time() - epoch_start, eta_sec)

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/triplet", epoch_trip.val, global_step)
                writer.add_scalar("train/pos_dist", epoch_pos.val, global_step)
                writer.add_scalar("train/neg_dist", epoch_neg.val, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)

            global_step += 1

        # ── Save epoch checkpoint ──────────────────────────────────
        save_checkpoint(cfg, model, optimizer, epoch, epoch_loss.avg, is_best=False)

        log_epoch_stats(epoch, max_epoch,
                        time.time() - epoch_start,
                        (time.time() - train_timer) / (epoch + 1) * (max_epoch - epoch - 1),
                        epoch_loss.avg, epoch_trip.avg, epoch_pos.avg, epoch_neg.avg, lr_now)
        writer.add_scalar("train/epoch_loss", epoch_loss.avg, epoch)
        writer.add_scalar("train/epoch_triplet", epoch_trip.avg, epoch)

        # ── Validation ────────────────────────────────────────────
        if val_loader:
            model.eval()
            val_loss = AverageMeter()
            with torch.no_grad():
                for data, labels in val_loader:
                    labels = labels.to(device, non_blocking=True)
                    # 🔧 特征缓存：兼容两种输入模式
                    if data.dim() == 5:
                        data = data.to(device, non_blocking=True)
                        sf, _, val_frame_scores, val_all_feats = model(data)
                    else:
                        feats = data.to(device, non_blocking=True)
                        sf, _, val_frame_scores, val_all_feats = model.forward_from_feats(feats)
                    vd = _compute_selector_loss(
                        selected_feats=sf,
                        labels=labels,
                        frame_scores=val_frame_scores,
                        hard_indices=None,
                        all_feats=val_all_feats,
                        cfg=cfg,
                        margin=margin,
                        index_weight=0.0,
                        score_margin_weight=0.0,
                    )
                    val_key = (
                        "loss_bimhm_total"
                        if "loss_bimhm_total" in vd
                        else "loss_otam_total"
                    )
                    val_loss.update(vd[val_key].item(), data.shape[0])

            log_val_epoch_stats(epoch, max_epoch, val_loss.avg)
            writer.add_scalar("val/loss", val_loss.avg, epoch)

            if val_loss.avg < best_val_loss:
                best_val_loss = val_loss.avg
                save_checkpoint(cfg, model, optimizer, epoch, val_loss.avg, is_best=True)
                logger.info(f"New best val loss: {best_val_loss:.6f} at epoch {epoch}")
        else:
            if epoch_loss.avg < best_val_loss:
                best_val_loss = epoch_loss.avg
                save_checkpoint(cfg, model, optimizer, epoch, epoch_loss.avg, is_best=True)

        writer.add_scalar("train/best_val_loss", best_val_loss, epoch)

    # ── Done ──────────────────────────────────────────────────────
    total_time = time.time() - train_timer
    writer.close()
    logger.info("=" * 80)
    logger.info(f"Done! Time: {str(datetime.timedelta(seconds=int(total_time)))}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info(f"Final: {os.path.join(output_dir, 'frame_selector_best.pth')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    train()


