#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" Builder for the dataloader."""

import torch
import utils.misc as misc
import utils.logging as logging
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from datasets.base.few_shot_dataset import Few_shot

logger = logging.get_logger(__name__)


def get_sampler(cfg, dataset, split, shuffle):
    if misc.get_num_gpus(cfg) > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def _keyframe_selection_enabled(cfg, split):
    if split == "train":
        value = getattr(cfg.TRAIN, "USE_KEYFRAME_SELECTION", False)
    elif split == "val":
        value = getattr(
            getattr(cfg, "VAL", cfg.TRAIN),
            "USE_KEYFRAME_SELECTION",
            getattr(cfg.TRAIN, "USE_KEYFRAME_SELECTION", False),
        )
    elif split in ("test", "submission"):
        value = getattr(
            cfg.TEST,
            "USE_KEYFRAME_SELECTION",
            getattr(cfg.TRAIN, "USE_KEYFRAME_SELECTION", False),
        )
    else:
        value = False
    return value is True


def _uses_cuda_frame_selector_in_loader(cfg, split):
    if not _keyframe_selection_enabled(cfg, split):
        return False
    if not hasattr(cfg, "FRAME_SELECTOR"):
        return False

    fs_cfg = cfg.FRAME_SELECTOR
    if not getattr(fs_cfg, "ENABLE", False):
        return False
    if not getattr(fs_cfg, "ENABLE_CACHE_DECODE", False):
        return False
    if not getattr(fs_cfg, "FEAT_CACHE_DIR", ""):
        return False

    sampling_method = getattr(cfg.DATA, "SAMPLING_METHOD", "default")
    if sampling_method not in (
        "otam_learned", "otam_anchor_keyframe",
        "bimhm_learned", "bimhm_anchor_keyframe",
        "pairwise_diverse_bimhm",
    ):
        return False
    return torch.cuda.is_available()


def build_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object.
    """
    assert split in ["train", "val", "test", "submission"]
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    else:
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(cfg, split)

    # Create a sampler for multi-process training
    sampler = get_sampler(cfg, dataset, split, shuffle)
    num_workers = int(cfg.DATA_LOADER.NUM_WORKERS)
    loader_kwargs = {}
    if num_workers > 0 and _uses_cuda_frame_selector_in_loader(cfg, split):
        loader_kwargs["multiprocessing_context"] = "spawn"
        logger.info(
            "Using spawn DataLoader workers for CUDA frame selector "
            "during cache-based keyframe decoding."
        )

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
        **loader_kwargs,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    sampler = loader.sampler
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(cur_epoch)


def build_dataset(cfg, split):
    return Few_shot(cfg, split)


