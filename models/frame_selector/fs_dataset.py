#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""
Standalone video dataset for Frame Selector pretraining.

Reads video paths from a list file, decodes TOTAL_FRAMES per video,
and applies CLIP-standard transforms.

馃敡 鐗瑰緛缂撳瓨: 鏀寔閫氳繃 cfg.DATA.FEAT_CACHE.ENABLE 寮€鍏筹紝
   寮€鍚悗鐩存帴浠庣紦瀛樼洰褰曞姞杞介鎻愬彇鐨?CLIP 鐗瑰緛寮犻噺锛?
   璺宠繃瑙嗛瑙ｇ爜涓?CLIP 鍓嶅悜鎺ㄧ悊锛岃缁冮€熷害鎻愬崌 10脳 浠ヤ笂銆?
   鍏抽棴鏃跺畬鍏ㄥ鐜板師鏈夊湪绾垮姞杞介€昏緫銆?
"""

import os
import math
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import numpy as np
from decord import VideoReader, cpu

BICUBIC = InterpolationMode.BICUBIC


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Sample multiple videos per class for metric-learning selector losses.

    A normal shuffled batch on SSv2 often contains no positive pair.  This
    sampler keeps the old DataLoader behavior by default and is enabled only
    through LOSS.CLASS_BALANCED_BATCH.  It works with both decoded videos and
    feature-cache datasets because both expose ``labels``.
    """

    def __init__(self, labels, classes_per_batch=4, samples_per_class=2,
                 drop_last=True):
        self.labels = [int(x) for x in labels]
        self.classes_per_batch = max(2, int(classes_per_batch))
        self.samples_per_class = max(2, int(samples_per_class))
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.drop_last = bool(drop_last)

        self.class_to_indices = {}
        for index, label in enumerate(self.labels):
            self.class_to_indices.setdefault(label, []).append(index)
        self.classes = [
            label for label, indices in self.class_to_indices.items()
            if len(indices) >= self.samples_per_class
        ]
        if len(self.classes) < self.classes_per_batch:
            raise ValueError(
                "ClassBalancedBatchSampler needs at least "
                f"{self.classes_per_batch} classes with "
                f"{self.samples_per_class} samples each; got "
                f"{len(self.classes)} valid classes."
            )

    def __iter__(self):
        num_batches = len(self)
        for _ in range(num_batches):
            permutation = torch.randperm(len(self.classes)).tolist()
            chosen_classes = [
                self.classes[i] for i in permutation[:self.classes_per_batch]
            ]
            batch = []
            for label in chosen_classes:
                indices = self.class_to_indices[label]
                order = torch.randperm(len(indices)).tolist()
                batch.extend(indices[i] for i in order[:self.samples_per_class])
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)


# ============================================================================
# 馃敡 鐗瑰緛缂撳瓨锛氱敓鎴愬敮涓€缂撳瓨鏍囩锛岄槻姝㈤厤缃彉鏇村悗璇敤鏃х紦瀛?
# ============================================================================
def _cache_tag(cfg):
    """Generate unique cache tag from backbone + total_frames."""
    bn = cfg.FRAME_SELECTOR.BACKBONE_NAME.replace("/", "-")
    return f"{bn}_T{cfg.FRAME_SELECTOR.TOTAL_FRAMES}"


def _feat_cache_cfg(cfg):
    """Safely read FEAT_CACHE config as a dict with defaults."""
    raw = getattr(cfg.DATA, "FEAT_CACHE", None)
    if raw is None:
        return {"ENABLE": False, "CACHE_DIR": "", "AUTO_REBUILD": False}
    # YAML 瑙ｆ瀽鍚庡彲鑳芥槸瀵硅薄鎴?dict
    if hasattr(raw, "ENABLE"):
        return {
            "ENABLE": getattr(raw, "ENABLE", False),
            "CACHE_DIR": getattr(raw, "CACHE_DIR", ""),
            "AUTO_REBUILD": getattr(raw, "AUTO_REBUILD", False),
        }
    return {
        "ENABLE": raw.get("ENABLE", False),
        "CACHE_DIR": raw.get("CACHE_DIR", ""),
        "AUTO_REBUILD": raw.get("AUTO_REBUILD", False),
    }


# ============================================================================
# 鍘熷瑙嗛鏁版嵁闆嗭紙瑙ｇ爜 + CLIP 棰勫鐞嗭級
# ============================================================================
class FrameSelectorVideoDataset(Dataset):
    """
    Simple video dataset for frame selector training.

    Each line in the list file should be:
        relative_video_path  label_index

    The video_path is joined with DATA_ROOT_DIR.
    Decodes FRAME_SELECTOR.TOTAL_FRAMES uniformly sampled frames per video.
    """
    def __init__(self, cfg, split="train"):
        self.root = cfg.DATA.DATA_ROOT_DIR
        self.total_frames = cfg.FRAME_SELECTOR.TOTAL_FRAMES

        # Determine list file
        if split == "train":
            list_name = cfg.DATA.TRAIN_LIST
        elif split == "val":
            list_name = getattr(cfg.DATA, "VAL_LIST", None)
        elif split == "test":
            list_name = getattr(cfg.DATA, "TEST_LIST", None)
        else:
            raise ValueError(f"Unknown split: {split}")

        list_path = os.path.join(cfg.DATA.ANNO_DIR, list_name)
        with open(list_path, "r") as f:
            lines = f.readlines()

        self.videos = []
        self.labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            import re

            # 鈹€鈹€ Format 1: HMDB51/UCF101 鈫?"train0//videos/class/file.avi" 鈹€鈹€
            if "//" in line:
                parts = line.split("//")
                class_prefix = parts[0]
                video_path = "//".join(parts[1:])
                match = re.search(r"(\d+)", class_prefix)
                if match:
                    vid_ext = getattr(cfg.DATA, "FILE_EXTENSION", ".mp4")
                    if not any(video_path.endswith(ext)
                               for ext in [".mp4", ".avi", ".webm", ".mkv"]):
                        video_path = video_path + vid_ext
                    self.videos.append(video_path)
                    self.labels.append(int(match.group(1)))
                else:
                    print(f"  [SKIP] cannot parse label from: {line}")
                continue

            # 鈹€鈹€ Format 2: SSv2 鈫?"train{cls_id}/{video_id}" or "test{cls_id}/{video_id}" 鈹€鈹€
            if "/" in line and (line.startswith("train") or line.startswith("test")):
                parts = line.split("/")
                if len(parts) == 2 and parts[1]:
                    match = re.search(r"(\d+)", parts[0])
                    if match:
                        video_id = parts[1]
                        # SSv2 video IDs have no extension 鈫?read ext from config
                        vid_ext = getattr(cfg.DATA, "FILE_EXTENSION", ".mp4")
                        if not any(video_id.endswith(ext)
                                   for ext in [".mp4", ".avi", ".webm", ".mkv"]):
                            video_id = video_id + vid_ext
                        self.videos.append(video_id)
                        self.labels.append(int(match.group(1)))
                        continue

            # 鈹€鈹€ Format 3: Simple 鈫?"relative/path  label_index" 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            parts = line.split()
            if len(parts) >= 2:
                self.videos.append(parts[0])
                self.labels.append(int(parts[1]))

        # 鈹€鈹€ CLIP-standard transforms 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        crop_size = getattr(cfg.DATA, "TRAIN_CROP_SIZE", 224)
        scale_size = getattr(cfg.DATA, "TRAIN_JITTER_SCALES", [256, 256])
        mean = getattr(cfg.DATA, "MEAN", [0.48145466, 0.4578275, 0.40821073])
        std = getattr(cfg.DATA, "STD", [0.26862954, 0.26130258, 0.27577711])

        if split == "train" and getattr(cfg.DATA, "USE_AUG", True):
            from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
            self.transform = Compose([
                RandomResizedCrop(crop_size, scale=(0.8, 1.0), interpolation=BICUBIC),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = Compose([
                Resize(scale_size[0], interpolation=BICUBIC),
                CenterCrop(crop_size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

        print(f"[FrameSelectorVideoDataset] split={split}, {len(self.videos)} videos, "
              f"TOTAL_FRAMES={self.total_frames}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root, self.videos[idx])
        label = self.labels[idx]

        for retry in range(5):
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                total = len(vr)
                indices = torch.linspace(0, total - 1, self.total_frames).long().tolist()
                frames_np = vr.get_batch(indices).asnumpy()
                del vr

                frames_list = []
                for i in range(frames_np.shape[0]):
                    img = Image.fromarray(frames_np[i])
                    frames_list.append(self.transform(img))
                frames = torch.stack(frames_list)  # (T, 3, H, W)

                return frames, label
            except Exception as e:
                print(f"  [SKIP] decode error: {video_path} (retry {retry+1}/5): {e}")
                idx = (idx + 1) % len(self.videos)
                video_path = os.path.join(self.root, self.videos[idx])
                label = self.labels[idx]
                continue

        print(f"  [FALLBACK] zero tensor for: {video_path}")
        return torch.zeros(self.total_frames, 3, 224, 224), label


# ============================================================================
# 馃敡 鐗瑰緛缂撳瓨锛氱紦瀛樼壒寰佹暟鎹泦
# ============================================================================
class FrameSelectorFeatureDataset(Dataset):
    """
    Load pre-extracted CLIP features from cache.
    褰?FEAT_CACHE.ENABLE=True 鏃舵浛浠?FrameSelectorVideoDataset 浣跨敤銆?

    缂撳瓨鐩綍缁撴瀯锛?
        {CACHE_DIR}/
            {tag}/
                train/{index:08d}.pt   鈫?FP16 (T, D)
                val/{index:08d}.pt
                train_metadata.pt
                val_metadata.pt

    寮傚父澶勭悊锛?
        - 缂撳瓨鏂囦欢缂哄け 鈫?鎵撳嵃璀﹀憡鍚庤繑鍥炲叏闆剁壒寰佸厹搴?
        - 缂撳瓨鐩綍涓嶅瓨鍦?鈫?鎶涘嚭 FileNotFoundError锛堟彁绀鸿繍琛屾彁鍙栬剼鏈級
    """
    def __init__(self, cfg, split="train"):
        cache_info = _feat_cache_cfg(cfg)
        root_dir = cache_info["CACHE_DIR"]
        tag = _cache_tag(cfg)
        self.cache_dir = os.path.join(root_dir, tag, split)

        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(
                f"馃敡 鐗瑰緛缂撳瓨: 鐩綍涓嶅瓨鍦?{self.cache_dir}\n"
                f"   璇峰厛杩愯: python tools/extract_fs_features.py --cfg ..."
            )

        # 鍔犺浇鍏冩暟鎹?
        meta_path = os.path.join(root_dir, tag, f"{split}_metadata.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            self.labels = meta.get("labels", [])
            self.total = meta.get("total", 0)
            cached_backbone = meta.get("backbone", "")
            cached_frames = meta.get("total_frames", 0)
            # 馃敡 鐗瑰緛缂撳瓨锛氭牎楠岄厤缃竴鑷存€?
            if cached_backbone != cfg.FRAME_SELECTOR.BACKBONE_NAME:
                print(f"  [WARN] Cache backbone mismatch: cached={cached_backbone}, "
                      f"config={cfg.FRAME_SELECTOR.BACKBONE_NAME}")
            if cached_frames != cfg.FRAME_SELECTOR.TOTAL_FRAMES:
                print(f"  [WARN] Cache frames mismatch: cached={cached_frames}, "
                      f"config={cfg.FRAME_SELECTOR.TOTAL_FRAMES}")
        else:
            # 鏃犲厓鏁版嵁鏃舵壂鎻忔枃浠?
            files = sorted(
                [f for f in os.listdir(self.cache_dir) if f.endswith(".pt")],
                key=lambda x: int(x.replace(".pt", "")),
            )
            self.total = len(files)
            self.labels = [0] * self.total
            print(f"  [WARN] No metadata found at {meta_path}, using filenames only")

        self.auto_rebuild = cache_info.get("AUTO_REBUILD", False)

        print(f"[FrameSelectorFeatureDataset] split={split}, {self.total} videos "
              f"鈫?{self.cache_dir}")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, f"{idx:08d}.pt")
        try:
            feats = torch.load(path)
            # 馃敡 鐗瑰緛缂撳瓨锛欶P16 鈫?FP32锛坰elector 闇€瑕?FP32 杈撳叆锛?
            feats = feats.float()
        except Exception as e:
            print(f"  [WARN] Cache load failed: {path} ({e}), returning zeros")
            feats = torch.zeros(32, 512)
        label = self.labels[idx] if idx < len(self.labels) else 0
        return feats, label


# ============================================================================
# DataLoader 宸ュ巶锛堣嚜鍔ㄦ娴嬬紦瀛樺紑鍏筹級
# ============================================================================
def build_fs_dataloader(cfg, split="train"):
    """Build DataLoader for frame selector training.

    馃敡 鐗瑰緛缂撳瓨锛氭牴鎹?FEAT_CACHE.ENABLE 鑷姩閫夋嫨鏁版嵁闆嗭細
        True  鈫?FrameSelectorFeatureDataset锛堣烦杩囪В鐮?+ CLIP锛?
        False 鈫?FrameSelectorVideoDataset锛堝師濮嬭棰戣В鐮侊紝榛樿锛?
    """
    cache_info = _feat_cache_cfg(cfg)
    use_cache = cache_info["ENABLE"]

    if use_cache:
        try:
            dataset = FrameSelectorFeatureDataset(cfg, split)
            print(f"  >> Feature cache enabled: loading {split} from cache")
        except FileNotFoundError as e:
            print(f"  >> {e}")
            if cache_info.get("AUTO_REBUILD", False):
                print("  >> AUTO_REBUILD=True, falling back to online video loading")
                dataset = FrameSelectorVideoDataset(cfg, split)
            else:
                raise
    else:
        dataset = FrameSelectorVideoDataset(cfg, split)

    batch_size = getattr(cfg.DATA, "BATCH_SIZE", 4) if split == "train" \
                 else getattr(cfg.DATA, "VAL_BATCH_SIZE", 4)
    num_workers = getattr(cfg.DATA, "NUM_WORKERS", 4)
    loss_name = str(getattr(cfg.LOSS, "NAME", "otam_triplet")).lower()
    use_balanced = bool(
        getattr(cfg.LOSS, "CLASS_BALANCED_BATCH", False)
    ) and loss_name.startswith("bimhm")

    if use_balanced:
        sampler = ClassBalancedBatchSampler(
            dataset.labels,
            classes_per_batch=getattr(cfg.LOSS, "CLASSES_PER_BATCH", 4),
            samples_per_class=getattr(cfg.LOSS, "SAMPLES_PER_CLASS", 2),
            drop_last=(split == "train"),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(
            f"  >> Bi-MHM class-balanced batches: "
            f"{sampler.classes_per_batch} classes 脳 "
            f"{sampler.samples_per_class} samples"
        )
    else:
        shuffle = split == "train"
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )
    return loader



