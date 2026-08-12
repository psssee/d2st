#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 
# -----------------------------------------------
# Modified by Qizhong Tan
# -----------------------------------------------

""" BaseVideoDataset object to be extended for specific datasets. """

import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.utils.dlpack as dlpack
import utils.logging as logging
import re
import abc
import time
import random
import decord
import traceback
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu

decord.bridge.set_bridge('native')
from torchvision.transforms import Compose

import utils.bucket as bu

logger = logging.get_logger(__name__)


class BaseVideoDataset(torch.utils.data.Dataset):
    """
    The BaseVideoDataset object provides a base object for all the video/image/video-text datasets.
    Abstract methods are provided for completion in the specific datasets.
    Necessary methods for all datasets such as "_decode_video", "_decode_image", 
    "__getitem__" (with standard procedure for loading the data) as well as sampling methods 
    such as "_interval_based_sampling" and "_segment_based_sampling" are implemented. 
    The specific video datasets can be extended from this dataset according to different needs.
    """

    def __init__(self, cfg, split):
        """
        For initialization of the dataset, the global cfg and the split need to provided.
        Args:
            cfg     (Config): The global config object.
            split   (str): The split, e.g., "train", "val", "test"
        """
        self.cfg = cfg
        self.split = split
        self.data_root_dir = cfg.DATA.DATA_ROOT_DIR
        self.anno_dir = cfg.DATA.ANNO_DIR
        self._corrupted_videos = set()

        if self.split in ["train", "val"]:
            self.dataset_name = cfg.TRAIN.DATASET
            self._num_clips = 1
        elif self.split in ["test", "submission"]:
            self.dataset_name = cfg.TEST.DATASET
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        else:
            raise NotImplementedError("Split not supported")

        self._num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self._sampling_rate = cfg.DATA.SAMPLING_RATE

        self.gpu_transform = cfg.AUGMENTATION.USE_GPU  # whether or not to perform the transform on GPU

        self.decode = self._decode_video  # decode function, decode videos by default

        self.buckets = {}

        # if set to true, _pre_transformation_config will be called before every transformations
        # this is used in the testset, where cropping positions are set for the controlled crop
        self._pre_transformation_config_required = False
        self._construct_dataset(cfg)
        self._config_transform()

    @abc.abstractmethod
    def _get_dataset_list_name(self):
        """
        Returns the list for the dataset. 
        Returns:
            name (str): name of the list to be read
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_sample_info(self, index):
        """
        Returns the sample info corresponding to the index.
        Args: 
            index (int): target index
        Returns:
            sample_info (dict): contains different informations to be used later
                Things that must be included are:
                "path" indicating the target's path w.r.t. index
                "supervised_label" indicating the class of the target 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_ssl_label(self, frames):
        """
        Uses cfg to obtain ssl label.
        Returns:
            ssl_label (dict): self-supervised labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _config_transform(self):
        """
        Uses cfg to config transforms and assign the transforms to self.transform
        Note: This is only used in the supervised setting.
            For self-supervised training, the augmentations are performed in the 
            corresponding generator.
        """
        self.transform = Compose([])
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _custom_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        raise NotImplementedError

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        """
        Returns the list of frame indexes in the video for decoding.
        Args:
            vid_length (int): video length
            clip_idx (int): clip index, -1 if random sampling (interval based sampling)
            num_clips (int): overall number of clips for clip_idx != -1 (interval based sampling)
            num_frames (int): number of frames to sample
            interval (int): the step size for interval based sampling (interval based sampling)
            random_sample (int): whether to randomly sample one frame from each segment (segment based sampling)
        Returns:
            frame_id_list (list): indicates which frames to sample from the video
        """
        if self.cfg.PRETRAIN.ENABLE and self.split == "train":
            return self._custom_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate, random_sample)
        else:
            if self.cfg.DATA.SAMPLING_MODE == "interval_based":
                # return self._interval_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
                return self._interval_based_sampling(vid_length, vid_fps, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, self._sampling_rate)
            elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
                return self._segment_based_sampling(vid_length, clip_idx, self.cfg.TEST.NUM_ENSEMBLE_VIEWS, self._num_frames, random_sample)
            else:
                raise NotImplementedError

    def _construct_dataset(self, cfg):
        """
        Constructs the dataset according to the global config object.
        Currently supports reading from csv, json and txt.
        Args:
            cfg (Config): The global config object.
        """
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        try:
            logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
            local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
            local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
            if local_file[-4:] == ".csv":
                import pandas
                lines = pandas.read_csv(local_file)
                for line in lines.values.tolist():
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            elif local_file[-4:] == "json":
                import json
                with open(local_file, "r") as f:
                    lines = json.load(f)
                for line in lines:
                    for idx in range(self._num_clips):
                        self._samples.append(line)
                        self._spatial_temporal_index.append(idx)
            else:
                with open(local_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line.strip())
                            self._spatial_temporal_index.append(idx)
            logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
        except:
            raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

        # validity check
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))

    def _read_video(self, video_path, index):
        """
        Wrapper for downloading the video and generating the VideoReader object for reading the video.
        Args:
            video_path (str): video path to read the video from. Can in OSS form or in local hard drives.
            index      (int):  for debug.
        Returns:
            vr              (VideoReader):  VideoReader object wrapping the video.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + video_path.split('/')[-1]
        try:
            vr = None
            tmp_file = self._get_object_to_file(video_path, tmp_file, read_from_buffer=True, num_retries=1 if self.split == "train" else 20)
            vr = VideoReader(tmp_file, num_threads=1)
            success = True
        except:
            success = False
        file_to_remove = [tmp_file] if video_path[:3] == "oss" else [None]  # if not downloaded from oss, then no files need to be removed
        return vr, file_to_remove, success

    def _keyframe_selection_enabled(self):
        """Enable BDTS/keyframe selection only for explicit YAML boolean true."""
        if self.split == "train":
            value = getattr(self.cfg.TRAIN, "USE_KEYFRAME_SELECTION", False)
        elif self.split == "val":
            value = getattr(
                getattr(self.cfg, "VAL", self.cfg.TRAIN),
                "USE_KEYFRAME_SELECTION",
                getattr(self.cfg.TRAIN, "USE_KEYFRAME_SELECTION", False),
            )
        elif self.split in ("test", "submission"):
            value = getattr(
                self.cfg.TEST,
                "USE_KEYFRAME_SELECTION",
                getattr(self.cfg.TRAIN, "USE_KEYFRAME_SELECTION", False),
            )
        else:
            value = False
        return value is True

    def _should_use_cache_decode(self, use_keyframe=None):
        if use_keyframe is None:
            use_keyframe = self._keyframe_selection_enabled()
        if not use_keyframe:
            return False
        if not hasattr(self.cfg, "FRAME_SELECTOR"):
            return False
        fs_cfg = self.cfg.FRAME_SELECTOR
        if not getattr(fs_cfg, "ENABLE", False):
            return False
        if not getattr(fs_cfg, "ENABLE_CACHE_DECODE", False):
            return False
        if not getattr(fs_cfg, "FEAT_CACHE_DIR", ""):
            return False
        sampling_method = getattr(self.cfg.DATA, "SAMPLING_METHOD", "default")
        return sampling_method in (
            "otam_learned", "otam_anchor_keyframe",
            "bimhm_learned", "bimhm_anchor_keyframe",
            "pairwise_diverse_bimhm",
        )

    def _get_feat_cache_path(self, sample_info, index=None):
        cache_dir = self.cfg.FRAME_SELECTOR.FEAT_CACHE_DIR
        video_path = sample_info["path"].replace("\\", "/")
        root = self.cfg.DATA.DATA_ROOT_DIR.replace("\\", "/").rstrip("/") + "/"

        rel_path = video_path
        if rel_path.startswith(root):
            rel_path = rel_path[len(root):]
        rel_path = rel_path.replace("//", "/")
        rel_path = os.path.splitext(rel_path)[0]

        candidates = []
        rel_parts = [p for p in rel_path.split("/") if p]
        splits_to_try = []
        if hasattr(self, "split") and self.split:
            splits_to_try.append(self.split)
            if self.split == "test":
                splits_to_try.append("val")
            elif self.split == "val":
                splits_to_try.append("test")
        splits_to_try.append(None)

        for split_name in splits_to_try:
            if split_name is None:
                candidates.append(os.path.join(cache_dir, *rel_parts) + ".pt")
            else:
                candidates.append(os.path.join(cache_dir, split_name, *rel_parts) + ".pt")

        if index is not None:
            for split_name in splits_to_try:
                if split_name is None:
                    candidates.append(os.path.join(cache_dir, f"{index:08d}.pt"))
                else:
                    candidates.append(os.path.join(cache_dir, split_name, f"{index:08d}.pt"))

        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    def _get_uniform_anchor_indices(self, total_frames, num_anchors):
        total_frames = int(total_frames)
        num_anchors = max(0, min(int(num_anchors), total_frames))
        if num_anchors == 0:
            return []

        anchors = []
        for i in range(num_anchors):
            start = int(round(i * total_frames / num_anchors))
            end = int(round((i + 1) * total_frames / num_anchors))
            end = max(end, start + 1)
            anchors.append(min(total_frames - 1, (start + end - 1) // 2))
        return sorted(set(anchors))

    def _select_otam_anchor_keyframe_indices(self, key_indices, total_frames, target_frames):
        fs_cfg = self.cfg.FRAME_SELECTOR
        total_frames = int(total_frames)
        target_frames = int(target_frames)
        num_anchors = int(getattr(fs_cfg, "NUM_ANCHOR_FRAMES", target_frames // 2))
        num_keys = int(getattr(fs_cfg, "NUM_KEY_FRAMES", target_frames - num_anchors))
        min_gap = int(getattr(fs_cfg, "MIN_FRAME_GAP", 2))

        num_anchors = max(0, min(num_anchors, target_frames))
        num_keys = max(0, min(num_keys, target_frames - num_anchors))
        anchors = self._get_uniform_anchor_indices(total_frames, num_anchors)
        selected = list(anchors)

        clean_keys = []
        for idx in key_indices:
            idx = int(idx)
            if 0 <= idx < total_frames and idx not in clean_keys:
                clean_keys.append(idx)

        for idx in clean_keys:
            if len(selected) >= num_anchors + num_keys:
                break
            if idx in selected:
                continue
            if all(abs(idx - anchor) > min_gap for anchor in anchors):
                selected.append(idx)

        for idx in clean_keys:
            if len(selected) >= target_frames:
                break
            if idx not in selected:
                selected.append(idx)

        if len(selected) < target_frames:
            fallback = torch.linspace(0, max(total_frames - 1, 0), target_frames).long().tolist()
            for idx in fallback:
                idx = int(idx)
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_frames:
                    break

        if len(selected) < target_frames:
            for idx in range(total_frames):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= target_frames:
                    break

        selected = sorted(selected[:target_frames])
        if len(selected) != target_frames:
            raise RuntimeError(
                f"[AnchorKeyframe] expected {target_frames} frames, got {len(selected)}"
            )
        return selected

    def _decode_with_cache(self, vr, sample_info, total_video_frames, target_frames, index=None):
        fs_cfg = self.cfg.FRAME_SELECTOR
        total_frames_cfg = int(fs_cfg.TOTAL_FRAMES)
        cache_path = self._get_feat_cache_path(sample_info, index=index)

        if not os.path.isfile(cache_path):
            policy = getattr(fs_cfg, "CACHE_MISMATCH_POLICY", "error")
            if policy == "skip":
                logger.warning(f"[CacheDecode] cache file not found, skip sample: {cache_path}")
                return None
            raise FileNotFoundError(
                f"[CacheDecode] cache file not found: {cache_path}\n"
                f"  video path: {sample_info['path']}\n"
                f"  FEAT_CACHE_DIR: {fs_cfg.FEAT_CACHE_DIR}"
            )

        feat_32 = torch.load(cache_path, map_location="cpu").float()
        if feat_32.dim() == 3 and feat_32.shape[0] == 1:
            feat_32 = feat_32.squeeze(0)
        if feat_32.shape[0] != total_frames_cfg:
            raise RuntimeError(
                f"[CacheDecode] expected {total_frames_cfg} cached frames, "
                f"got shape={tuple(feat_32.shape)} from {cache_path}"
            )

        from utils.frame_sampler_cache import get_selected_indices
        rel_indices = get_selected_indices(feat_32, self.cfg)

        sampling_method = getattr(self.cfg.DATA, "SAMPLING_METHOD", "default")
        if sampling_method in ("otam_anchor_keyframe", "bimhm_anchor_keyframe"):
            rel_indices = self._select_otam_anchor_keyframe_indices(
                key_indices=rel_indices,
                total_frames=total_frames_cfg,
                target_frames=target_frames,
            )
        sample_info["rel_indices"] = rel_indices
        sample_info["rel_total"] = total_frames_cfg

        frame_indices_32 = torch.linspace(
            0, total_video_frames - 1, total_frames_cfg
        ).long().tolist()
        real_frame_numbers = [frame_indices_32[i] for i in rel_indices]
        real_frame_numbers = [
            max(0, min(int(i), total_video_frames - 1)) for i in real_frame_numbers
        ]

        frames = dlpack.from_dlpack(vr.get_batch(real_frame_numbers).to_dlpack()).clone()

        count = getattr(self, "_bdts_log_count", 0)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        if worker_id == 0 and count < 2:
            logger.info(
                f"[BDTS] {sampling_method}: {sample_info.get('path', '?')} "
                f"rel={rel_indices}, real={real_frame_numbers}"
            )
            self._bdts_log_count = count + 1
        return frames

    def _decode_video(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the video given the sample info.
        Args:
            sample_info         (dict): containing the "path" key specifying the location of the video.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
        Returns:
            data            (dict): key "video" for the video data.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        vr, file_to_remove, success = self._read_video(path, index)

        if not success:
            return vr, file_to_remove, success

        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test" or self.split == "submission":
            clip_idx = self._spatial_temporal_index[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                self.spatial_idx = 0
            else:
                self.spatial_idx = self._spatial_temporal_index[index] % self.cfg.TEST.NUM_SPATIAL_CROPS

        target_frames = self.cfg.DATA.NUM_INPUT_FRAMES
        use_keyframe = self._keyframe_selection_enabled()
        if self._should_use_cache_decode(use_keyframe):
            cache_frames = self._decode_with_cache(
                vr, sample_info, len(vr), target_frames, index=index
            )
            if cache_frames is None:
                del vr
                return None, file_to_remove, False
            del vr
            data = {"video": cache_frames}
            if "rel_indices" in sample_info:
                total = sample_info.get("rel_total", self.cfg.FRAME_SELECTOR.TOTAL_FRAMES)
                if total > 1:
                    ts = [i / (total - 1) for i in sample_info["rel_indices"]]
                else:
                    ts = [0.0 for _ in sample_info["rel_indices"]]
                data["timestamps"] = torch.tensor(ts, dtype=torch.float32)
            return data, file_to_remove, True

        frame_list = []
        for idx in range(num_clips_per_video):
            # for each clip in the video,
            # a list is generated before decoding the specified frames from the video
            list_ = self._get_video_frames_list(
                len(vr),
                vr.get_avg_fps(),
                clip_idx,
                random_sample=True if self.split == "train" else False
            )
            frames = None
            frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
            frame_list.append(frames)
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        del vr
        return {"video": frames}, file_to_remove, True

    def _read_image(self, path, index):
        """
        Wrapper for downloading the image and generating the PIL.Image object for reading the image.
        Args:
            path    (str): image path to read the image from. Can in OSS form or in local hard drives.
            index   (int):  for debug.
        Returns:
            img             (PIL.Image):    image object for further processing.
            file_to_remove  (list):         list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool):         flag for the indication of success or not.
        """
        tmp_file = str(round(time.time() * 1000)) + path.split('/')[-1]
        for tmp in range(10):
            try:
                img = None
                tmp_file = self._get_object_to_file(path, tmp_file, read_from_buffer=True)
                if isinstance(tmp_file, str):
                    with open(tmp_file, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                else:
                    img = Image.open(tmp_file).convert('RGB')
                success = True
                break
            except:
                success = False
        file_to_remove = [tmp_file] if path[:3] == "oss" else [None]
        return img, file_to_remove, success

    def _decode_image(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the image given the sample info.
        Args:
            sample_info         (dict): containing the "path" key specifying the location of the image.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
                                        specifically in this function, num_clips_per_video does not matter since all things to be decoded is one image.
        Returns:
            data            (dict): key "video" for the image data.
                                    because this is a video database, the images will be in the shape of 1,H,W,C before further processing.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        img, tmp_file, success = self._read_image(path, index)

        if not success:
            return None, tmp_file, success

        frame = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], len(img.getbands()))
        frame = frame.unsqueeze(0)  # 1, H, W, C
        return {"video": frame}, tmp_file, True

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
        Returns:
            frames (dict): {
                "video": (tensor),
                "text_embedding" (optional): (tensor)
            }
            labels (dict): {
                "supervised": (tensor),
                "self-supervised" (optional): (...)
            }
        """
        sample_info = self._get_sample_info(index)

        # skip known-corrupted videos silently
        if sample_info["path"] in self._corrupted_videos:
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)

        # decode the data
        # local files: retrying won't help a corrupted file; OSS files: retry for transient network errors
        is_oss = sample_info["path"][:3] == "oss"
        retries = (1 if self.split == "train" else 10) if is_oss else 1
        for retry in range(retries):
            try:
                data, file_to_remove, success = self.decode(
                    sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                )
                break
            except Exception as e:
                success = False
                if retry == 0:
                    traceback.print_exc()
                logger.debug("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                    retry + 1, retries, index, sample_info["path"]
                ))

        if not success:
            self._corrupted_videos.add(sample_info["path"])
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)

        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()

        labels = {}
        labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
        if self.cfg.PRETRAIN.ENABLE:
            # generates the different augmented samples for pre-training
            try:
                data, labels["self-supervised"] = self.ssl_generator(data, index)
            except Exception as e:
                traceback.print_exc()
                print("Error at Vid index: {}, Vid path: {}, Vid shape: {}".format(
                    index, sample_info["path"], data["video"].shape
                ))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        else:
            # augment the samples for supervised training
            labels["self-supervised"] = {}
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():
                data["video"] = self.transform(data["video"])  # C, T, H, W = 3, 16, 240, 320, RGB

        # if the model is SlowFast, generate two sets of inputs with different framerates.
        if self.cfg.VIDEO.BACKBONE.META_ARCH == "Slowfast":
            slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1] // self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA + 1).long()[:-1]
            fast_frames = data["video"].clone()
            slow_frames = data["video"][:, slow_idx, :, :].clone()
            data["video"] = [slow_frames, fast_frames]
        bu.clear_tmp_file(file_to_remove)

        return data, labels, index, {}

    def _get_object_to_file(self, obj_file: str, local_file, read_from_buffer=False, num_retries=10):
        """
        Wrapper for downloading the file object.
        Args:
            obj_file         (str):  the target file to be downloaded (if it starts by "oss").
            local_file       (str):  the local file to store the downloaded file.
            read_from_butter (bool): whether or not to directly download to the memory
            num_retries      (int):  number of retries.
        Returns:
            str or BytesIO depending on the read_from_buffer flag
            if read_from_buffer==True:
                returns BytesIO
            else:
                returns str (indicating the location of the specified file)
        """
        if obj_file[:3] == "oss":
            bucket_name = obj_file.split('/')[2]
            if bucket_name not in self.buckets.keys():
                self.buckets[bucket_name] = self._initialize_oss(bucket_name)
            if read_from_buffer:
                local_file = bu.read_from_buffer(
                    self.buckets[bucket_name],
                    obj_file,
                    bucket_name,
                    num_retries
                )
            else:
                bu.read_from_bucket(
                    self.buckets[bucket_name],
                    obj_file,
                    local_file,
                    bucket_name,
                    num_retries
                )
            return local_file
        else:
            return obj_file

    def _initialize_oss(self, bucket_name):
        """
        Initializes the oss bucket.
        Currently supporting two OSS accounts.
        """
        if hasattr(self.cfg.OSS, "SECONDARY_DATA_OSS") and \
                self.cfg.OSS.SECONDARY_DATA_OSS.ENABLE and \
                bucket_name in self.cfg.OSS.SECONDARY_DATA_OSS.BUCKETS:
            return bu.initialize_bucket(
                self.cfg.OSS.SECONDARY_DATA_OSS.KEY,
                self.cfg.OSS.SECONDARY_DATA_OSS.SECRET,
                self.cfg.OSS.SECONDARY_DATA_OSS.ENDPOINT,
                bucket_name
            )
        else:
            return bu.initialize_bucket(
                self.cfg.OSS.KEY,
                self.cfg.OSS.SECRET,
                self.cfg.OSS.ENDPOINT,
                bucket_name
            )

    def __len__(self):
        """
        Returns the number of samples.
        """
        if hasattr(self.cfg.TRAIN, "NUM_SAMPLES") and self.split == 'train':
            return self.cfg.TRAIN.NUM_SAMPLES
        else:
            return len(self._samples)


    # def _interval_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval):
    def _interval_based_sampling(self, vid_length, vid_fps, clip_idx, num_clips, num_frames, interval):
        if num_frames == 1:
            index = [random.randint(0, vid_length - 1)]
        else:
            if self.split == "train" and hasattr(self.cfg.DATA, "SAMPLING_RATE_TRAIN"):
                interval = self.cfg.DATA.SAMPLING_RATE_TRAIN
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS
            elif hasattr(self.cfg.DATA, "SAMPLING_RATE_TEST") and self.cfg.DATA.SAMPLING_RATE_TEST > 40:
                interval = vid_length // num_frames
                clip_length = vid_length // num_frames * num_frames
                index = [random.randint(ind * interval, ind * interval + interval - 1) for ind in range(num_frames)]
                return index
            elif self.cfg.DATA.SAMPLING_RATE > 40:  # SAMPLING_RATE_TEST
                interval = vid_length // num_frames
                clip_length = vid_length // num_frames * num_frames
                index = [random.randint(ind * interval, ind * interval + interval - 1) for ind in range(num_frames)]
                return index
            else:
                # transform FPS
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS

            if clip_length > vid_length:
                clip_length = vid_length // num_frames * num_frames

            max_idx = max(vid_length - clip_length + 1, 0)
            if clip_idx == -1:  # random sampling
                start_idx = random.uniform(0, max_idx)
            else:
                if num_clips == 1:
                    start_idx = max_idx / 2
                else:
                    start_idx = max_idx * clip_idx / num_clips
            end_idx = start_idx + clip_length - interval

            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, vid_length - 1).long()

        return index

    def _segment_based_sampling(self, vid_length, clip_idx, num_clips, num_frames, random_sample):
        """
        Generates the frame index list using segment based sampling.
        Args:
            vid_length    (int):  the length of the whole video (valid selection range).
            clip_idx      (int):  -1 for random temporal sampling, and positive values for sampling specific clip from the video
            num_clips     (int):  the total clips to be sampled from each video. 
                                    combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames    (int):  number of frames in each sampled clips.
            random_sample (bool): whether or not to randomly sample from each segment. True for train and False for test.
        Returns:
            index (tensor): the sampled frame indexes
        """
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_length, num_frames + 1)
        for idx in range(num_frames):
            if random_sample:
                index[idx] = random.uniform(index_range[idx], index_range[idx + 1])
            else:
                if num_clips == 1:
                    index[idx] = (index_range[idx] + index_range[idx + 1]) / 2
                else:
                    index[idx] = index_range[idx] + (index_range[idx + 1] - index_range[idx]) * (clip_idx + 1) / num_clips
        index = torch.round(torch.clamp(index, 0, vid_length - 1)).long()

        return index


