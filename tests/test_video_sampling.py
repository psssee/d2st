"""Exercise production sampling/retry methods with an in-memory video reader."""

import ast
import logging
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.utils.dlpack as dlpack


ROOT = Path(__file__).resolve().parents[1]


def load_methods(path, class_name, methods, namespace, base=object):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    source = next(node for node in tree.body if getattr(node, 'name', '') == class_name)
    nodes = [node for node in source.body if getattr(node, 'name', '') in methods]
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return type(class_name, (base,), {name: scope[name] for name in methods})


NAMESPACE = {
    'torch': torch, 'random': random, 'os': os, 'dlpack': dlpack,
    'logger': logging.getLogger(__name__),
    'bu': SimpleNamespace(clear_tmp_file=Mock()),
    'traceback': SimpleNamespace(print_exc=Mock()),
}
Base = load_methods(
    ROOT / 'datasets/base/base_dataset.py', 'BaseVideoDataset',
    {'_interval_based_sampling', '_segment_based_sampling',
     '_get_video_frames_list', '_decode_video'}, NAMESPACE,
)
FewShot = load_methods(
    ROOT / 'datasets/base/few_shot_dataset.py', 'Few_shot',
    {
        'get_seq', 'get_seq_query', '__getitem__', '_unpack_seq',
        '_pack_timestamps', '_get_episode_classes',
        '_get_episode_num_videos', '_get_episode_video',
    },
    NAMESPACE, Base,
)


class MemoryReader:
    def __init__(self, length):
        self.frames = torch.arange(1, length + 1, dtype=torch.uint8).view(-1, 1, 1, 1)
        self.frames = self.frames.expand(-1, 2, 2, 3).contiguous()
        self.requested = None

    def __len__(self):
        return len(self.frames)

    def get_avg_fps(self):
        return 12.0

    def get_batch(self, indices):
        self.requested = torch.as_tensor(indices, dtype=torch.long)
        frames = self.frames[self.requested]
        return SimpleNamespace(to_dlpack=lambda: dlpack.to_dlpack(frames))


def dataset(split='test', length=7):
    data = FewShot()
    data.cfg = SimpleNamespace(
        PRETRAIN=SimpleNamespace(ENABLE=False),
        DATA=SimpleNamespace(NUM_INPUT_FRAMES=8, SAMPLING_RATE=50, TARGET_FPS=12,
                             SAMPLING_MODE='interval_based', VIDEO_EXT='.webm',
                             TRAIN_CROP_SIZE=2, TEST_CROP_SIZE=2),
        TEST=SimpleNamespace(NUM_ENSEMBLE_VIEWS=1, NUM_SPATIAL_CROPS=1, EPISODE_SEED=2026),
        TRAIN=SimpleNamespace(META_BATCH=True, WAY=2, QUERY_PER_CLASS=1, QUERY_PER_CLASS_TEST=1),
        AUGMENTATION=SimpleNamespace(), VIDEO=SimpleNamespace(BACKBONE=SimpleNamespace(META_ARCH='ViT')),
        RANDOM_SEED=18, SUPPORT_SHOT=1,
    )
    data.split = data.split_dataset = split
    data.dataset_name = 'SSv2-Full'
    data.data_root_dir = '/videos'
    data._num_frames, data._sampling_rate = 8, 50
    data._spatial_temporal_index = [0] * 30
    data._corrupted_videos = set()
    data.gpu_transform = data._pre_transformation_config_required = False
    data.transform = data.transform_query = lambda x: x.permute(3, 0, 1, 2)
    data._keyframe_selection_enabled = lambda: False
    data._should_use_cache_decode = lambda _: False
    reader = MemoryReader(length)
    data._read_video = Mock(return_value=(reader, [None], True))
    data.decode = data._decode_video
    data.split_few_shot = SimpleNamespace(
        get_rand_vid=lambda label, idx: ('113474', 3),
        get_unique_classes=lambda: [0, 1], get_num_videos_for_class=lambda label: 10,
    )
    return data, reader


class VideoSamplingTests(unittest.TestCase):
    def test_short_clips_repeat_only_valid_frames(self):
        for length in range(1, 8):
            for split in ['train', 'val', 'test']:
                for rate in [2, 50]:
                    with self.subTest(length=length, split=split, rate=rate):
                        data, _ = dataset(split, length)
                        data.cfg.DATA.SAMPLING_RATE = rate
                        indices = data._interval_based_sampling(length, 12, -1, 1, 8, rate)
                        self.assertEqual(indices.dtype, torch.long)
                        self.assertEqual(indices.numel(), 8)
                        self.assertEqual(indices[0].item(), 0)
                        self.assertEqual(indices[-1].item(), length - 1)
                        self.assertTrue((indices[1:] >= indices[:-1]).all())
                        self.assertEqual(set(indices.tolist()), set(range(length)))

    def test_original_random_segments_unchanged_for_normal_clips(self):
        data, _ = dataset()
        for length in [8, 9, 16, 79, 200]:
            for use_test_rate in [False, True]:
                data.cfg.DATA.SAMPLING_RATE = 2 if use_test_rate else 50
                data.cfg.DATA.SAMPLING_RATE_TEST = 50 if use_test_rate else 0
                generator = random.Random(42)
                width = length // 8
                expected = [generator.randrange(i * width, (i + 1) * width) for i in range(8)]
                random.seed(42)
                actual = data._interval_based_sampling(length, 12, 0, 1, 8, 50)
                self.assertEqual(actual, expected)

    def test_invalid_lengths_and_frame_counts_are_explicit_errors(self):
        data, _ = dataset()
        for length, frames in [(0, 8), (-1, 8), (10, 0), (10, -1)]:
            with self.assertRaisesRegex(ValueError, 'positive video length and frame count'):
                data._interval_based_sampling(length, 12, 0, 1, frames, 50)
            with self.assertRaisesRegex(ValueError, 'positive video length and frame count'):
                data._segment_based_sampling(length, 0, 1, frames, False)

    def test_single_frame_and_segment_sampling_remain_valid(self):
        data, _ = dataset(length=1)
        self.assertEqual(data._interval_based_sampling(1, 12, 0, 1, 1, 50), [0])
        for random_sample in [True, False]:
            indices = data._segment_based_sampling(1, 0, 1, 8, random_sample)
            torch.testing.assert_close(indices, torch.zeros(8, dtype=torch.long))

    def test_short_video_decodes_real_frames_for_support_and_query(self):
        for method in ['get_seq', 'get_seq_query']:
            data, reader = dataset(length=7)
            video, video_id, timestamps = getattr(data, method)(0, 0)
            self.assertEqual(video.shape, (8, 3, 2, 2))
            self.assertEqual(video_id, 3)
            self.assertIsNone(timestamps)
            torch.testing.assert_close(video[:, 0, 0, 0].long(), reader.requested + 1)
            self.assertEqual(data._corrupted_videos, set())

    def test_empty_reader_never_reaches_batch_decode(self):
        data, reader = dataset(length=0)
        result, cleanup, success = data.decode({'path': '/videos/empty.webm'}, 0)
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIsNone(reader.requested)
        self.assertEqual(cleanup, [None])

    def test_failed_or_empty_decode_raises_instead_of_returning_dummy(self):
        for method in ['get_seq', 'get_seq_query']:
            for result in [(None, [None], False), ({'video': torch.empty(0)}, [None], True)]:
                with self.subTest(method=method, success=result[2]):
                    data, _ = dataset()
                    data.decode = Mock(return_value=result)
                    with self.assertRaisesRegex(ValueError, 'Video'):
                        getattr(data, method)(0, 0)
                    self.assertEqual(len(data._corrupted_videos), 1)
                    with self.assertRaisesRegex(ValueError, 'known-corrupted video'):
                        getattr(data, method)(0, 0)
                    data.decode.assert_called_once()

    def test_episode_retry_returns_complete_distinct_sets_reproducibly(self):
        def run():
            data, _ = dataset()
            calls = []

            def get_seq(label, idx):
                calls.append((label, idx))
                if len(calls) == 1:
                    raise ValueError('Unreadable video')
                vid = 1 + label * 10 + idx
                return torch.full((8, 3, 2, 2), vid), vid, None

            data.get_seq = get_seq
            return data[0], calls

        first, calls = run()
        second, _ = run()
        self.assertEqual(len(calls), 5)
        self.assertEqual(first['support_set'].shape, (16, 3, 2, 2))
        self.assertEqual(first['target_set'].shape, (16, 3, 2, 2))
        self.assertEqual(sorted(first['support_labels'].tolist()), [0, 1])
        support_ids = set(first['support_set'][:, 0, 0, 0].tolist())
        query_ids = set(first['target_set'][:, 0, 0, 0].tolist())
        self.assertFalse(support_ids & query_ids)
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_episode_uses_frozen_index_when_wrapper_is_replaced(self):
        data, _ = dataset()
        data._episode_videos = ('class0-video', 'class1-video')
        data._episode_class_to_indices = {0: (0,), 1: (1,)}
        data._episode_classes = (0, 1)
        data.split_few_shot = 17

        self.assertEqual(data._get_episode_classes(), [0, 1])
        self.assertEqual(data._get_episode_num_videos(1), 1)
        self.assertEqual(data._get_episode_video(1, 0), ('class1-video', 1))

    def test_episode_index_rejects_missing_or_out_of_range_videos(self):
        data, _ = dataset()
        data._episode_videos = ('class0-video',)
        data._episode_class_to_indices = {0: (0,)}
        with self.assertRaisesRegex(ValueError, 'No videos found'):
            data._get_episode_video(1, 0)
        with self.assertRaisesRegex(IndexError, 'out of range'):
            data._get_episode_video(0, 1)

    def test_exhausted_retries_never_return_a_partial_episode(self):
        data, _ = dataset()
        data.get_seq = Mock(side_effect=ValueError('No decodable video'))
        with self.assertRaisesRegex(RuntimeError, 'complete episode 0 after 5 attempts'):
            data[0]
        self.assertEqual(data.get_seq.call_count, 5)


if __name__ == '__main__':
    unittest.main()
