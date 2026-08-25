#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from utils.meters import TrainMeter, ValMeter
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

logger = logging.get_logger(__name__)


def _unwrap_meta_episode(task_dict):
    """Remove the DataLoader dimension for the single-episode meta learner."""
    episode_batch = int(task_dict["target_labels"].shape[0])
    if episode_batch != 1:
        raise ValueError(
            "Meta-batch training currently processes one episode per forward; "
            f"set TRAIN/TEST.BATCH_SIZE=1, got {episode_batch}."
        )
    return {key: value[0] for key, value in task_dict.items()}


def _log_fusion_state(model):
    model_without_ddp = model.module if hasattr(model, "module") else model
    fusion_module = next(
        (
            module
            for module in model_without_ddp.modules()
            if hasattr(module, "get_fusion_weights")
        ),
        None,
    )
    if fusion_module is None:
        return

    fusion_weights = fusion_module.get_fusion_weights()
    if fusion_weights:
        logger.info(
            "Fusion weights: %s",
            ", ".join(
                "{}={:.6f}".format(name, value)
                for name, value in fusion_weights.items()
            ),
        )
    if hasattr(fusion_module, "get_calibration_diagnostics"):
        diagnostics = fusion_module.get_calibration_diagnostics()
        if diagnostics:
            logger.info(
                "Prototype calibration: %s",
                ", ".join(
                    "{}={:.6f}".format(name, value)
                    for name, value in diagnostics.items()
                ),
            )


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader):
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    train_meter.iter_tic()

    best_val_acc = float("-inf")
    bad_eval_count = 0
    early_stop_cfg = getattr(cfg.TRAIN, "EARLY_STOP", None)
    early_stop_enabled = bool(
        early_stop_cfg is not None
        and getattr(early_stop_cfg, "ENABLE", False)
        and val_loader is not None
    )
    early_stop_patience = int(getattr(early_stop_cfg, "PATIENCE", 0))
    early_stop_min_delta = float(getattr(early_stop_cfg, "MIN_DELTA", 0.0))
    accumulation_steps = max(int(cfg.TRAIN.BATCH_SIZE_PER_TASK), 1)
    normalize_accumulated_gradients = bool(
        getattr(cfg.TRAIN, "NORMALIZE_ACCUMULATED_GRADIENTS", False)
    )
    gradient_scale = accumulation_steps if normalize_accumulated_gradients else 1
    logger.info(
        "Gradient accumulation: steps=%d, normalize=%s",
        accumulation_steps,
        normalize_accumulated_gradients,
    )

    for cur_iter, task_dict in enumerate(train_loader):
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', 'real_support_labels']'''
        if cur_iter >= cfg.TRAIN.NUM_TRAIN_TASKS:
            break
        # Evaluate before the next episode. The checkpoint is saved only when
        # this evaluation improves, so the output directory keeps the best
        # generalizing model instead of the final overfit model.
        cur_epoch = cur_iter // cfg.SOLVER.STEPS_ITER
        if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0:
            cur_epoch_save = cur_iter // cfg.TRAIN.VAL_FRE_ITER
            val_meter.set_model_ema_enabled(False)
            val_acc = eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                cfg,
            )
            _log_fusion_state(model)
            if val_acc > best_val_acc + early_stop_min_delta:
                best_val_acc = val_acc
                bad_eval_count = 0
                model_bucket = None
                checkpoint_path = cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                    cfg,
                    model_bucket,
                    checkpoint_name="checkpoint_best.pyth",
                )
                logger.info(
                    "New best validation accuracy: %.4f; checkpoint=%s",
                    best_val_acc,
                    checkpoint_path,
                )
            else:
                bad_eval_count += 1
                logger.info(
                    "Validation accuracy %.4f did not improve best %.4f "
                    "(%d/%d evaluations without improvement).",
                    val_acc,
                    best_val_acc,
                    bad_eval_count,
                    early_stop_patience,
                )
                if early_stop_enabled and bad_eval_count > early_stop_patience:
                    logger.info(
                        "Early stopping at iteration %d; best validation "
                        "accuracy was %.4f.",
                        cur_iter + 1,
                        best_val_acc,
                    )
                    break
            model.train()

        task_dict = _unwrap_meta_episode(task_dict)
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k].cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(float(cur_iter) / cfg.SOLVER.STEPS_ITER, cfg)
        optim.set_lr(optimizer, lr)

        model_dict = model(task_dict)
        target_logits = model_dict['logits']

        if hasattr(cfg.TRAIN, "USE_CLASSIFICATION_VALUE"):
            loss = F.cross_entropy(
                model_dict["logits"], task_dict["target_labels"].long()
            ) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(
                model_dict["class_logits"],
                torch.cat(
                    [
                        task_dict["real_support_labels"],
                        task_dict["real_target_labels"],
                    ],
                    0,
                ).long(),
            )
        else:
            loss = F.cross_entropy(
                model_dict["logits"], task_dict["target_labels"].long()
            )

        # check Nan Loss.
        if math.isnan(loss):
            loss.backward(retain_graph=False)
            optimizer.zero_grad()
            continue
        (loss / gradient_scale).backward(retain_graph=False)

        # optimize
        if ((cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0):
            optimizer.step()
            optimizer.zero_grad()

        # Compute the errors.
        preds = target_logits
        num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
        top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())

        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(top1_err, top5_err, loss, lr, train_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch + cfg.TRAIN.NUM_FOLDS - 1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    model.eval()
    val_meter.iter_tic()

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        task_dict = _unwrap_meta_episode(task_dict)
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k].cuda(non_blocking=True)

        # preds, logits = model(inputs)
        model_dict = model(task_dict)

        target_logits = model_dict['logits']
        loss = F.cross_entropy(
            model_dict["logits"], task_dict["target_labels"].long()
        )

        # Compute the errors.
        labels = task_dict['target_labels']
        preds = target_logits
        num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
        top1_err, top5_err = [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (loss.item(), top1_err.item(), top5_err.item())
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(top1_err, top5_err, val_loader.batch_size * max(misc.get_num_gpus(cfg), 1))
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_acc = 100.0 - val_meter.num_top1_mis / val_meter.num_samples
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return val_acc


def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH - start_epoch) % cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, cur_epoch)

    # Keep the pretrained CLIP backbone frozen while training D2ST extensions.
    trainable_markers = (
        'class_embedding',
        'temporal_embedding',
        'Adapter',
        'ln_post',
        'classification_layer',
        'focus_branch',
        'focus_alpha',
        'task_matcher',
        'task_match_alpha',
        'proto_calibrator',
        'proto_calib_alpha',
    )
    for name, param in model.named_parameters():
        if not any(marker in name for marker in trainable_markers):
            param.requires_grad = False

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        for name, param in model.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))

    train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, val_meter, val_loader)
