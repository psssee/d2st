#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""
绂荤嚎棰勬彁鍙?CLIP 鐗瑰緛锛屼緵 OTAM 鎶藉抚鍣ㄨ缁冧娇鐢ㄣ€?

鐢ㄦ硶锛?
    python tools/extract_fs_features.py --cfg configs/projects/FRAMESELECTOR/ssv2_fs_train.yaml

娴佺▼锛?
    閬嶅巻鏁版嵁闆嗘墍鏈夎棰?鈫?瑙ｇ爜 TOTAL_FRAMES 甯?鈫?鍥哄畾鍙樻崲 (Resize+CenterCrop)
    鈫?CLIP 涓诲共鎺ㄧ悊 鈫?(T, D) FP16 鐗瑰緛 鈫?淇濆瓨鍒扮紦瀛樼洰褰?

缂撳瓨鏂囦欢鍛藉悕瑙勫垯锛歿backbone_name_clean}_T{total_frames}/{split}/{index:08d}.pt

馃敡 鍏抽敭璁捐锛?
    1. 璁粌闆嗕篃浣跨敤楠岃瘉闆嗗悓娆惧浐瀹氬彉鎹紙Resize + CenterCrop锛夛紝涓嶄娇鐢ㄩ殢鏈哄寮?
       鍘熷洜锛氭娊甯у櫒瀵圭┖闂村寮轰笉鏁忔劅锛屽浐瀹氬彉鎹㈡崲鍙栫殑 10脳 鎻愰€熸敹鐩婅繙澶т簬澧炲己鏀剁泭
    2. FP16 瀛樺偍锛岃妭鐪?50% 纾佺洏绌洪棿
    3. 鏂偣缁紶锛氬凡瀛樺湪鐨勭紦瀛樻枃浠惰嚜鍔ㄨ烦杩?
    4. 鏂囦欢鍚嶅寘鍚骞茬綉鍚?+ 鎬诲抚鏁版爣璇嗭紝閰嶇疆鍙樻洿鍚庤嚜鍔ㄩ噸寤虹紦瀛?
"""

import os
import sys
import time
import argparse
import datetime
import traceback

import torch

sys.path.append(os.path.abspath(os.curdir))

from models.frame_selector.fs_dataset import FrameSelectorVideoDataset
import clip
from utils.config import Config
import utils.logging as logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract CLIP features for frame selector")
    parser.add_argument("--cfg", dest="cfg_file", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if cache exists")
    parser.add_argument("--splits", type=str, default=None,
                        help="Splits to extract, comma-separated (e.g. 'train,val,test'). "
                             "Default: auto-detect from config.")
    parser.add_argument("--num-splits", type=int, default=1,
                        help="Split dataset into N parts for multi-GPU extraction.")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of this process (0-based, < num-splits).")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Automatically spawn N GPU workers (overrides num-splits & rank).")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _cache_tag(cfg):
    """Generate unique cache tag from backbone + total_frames."""
    bn = cfg.FRAME_SELECTOR.BACKBONE_NAME.replace("/", "-")
    return f"{bn}_T{cfg.FRAME_SELECTOR.TOTAL_FRAMES}"


@torch.no_grad()
def extract_split(cfg, split, backbone, cache_split_dir, device, overwrite=False,
                  num_splits=1, rank=0):
    """
    Extract features for one split.

    馃敡 鐗瑰緛缂撳瓨锛氫娇鐢ㄥ浐瀹氬彉鎹紙闈為殢鏈哄寮猴級锛岀‘淇濈紦瀛樼壒寰佷笌鎺ㄧ悊涓€鑷淬€?
    鏁版嵁闆嗗湪鏋勯€犳椂 split="val" 浼氳烦杩囬殢鏈哄寮猴紝涓庢彁鍙栬剼鏈竴鑷淬€?
    """
    os.makedirs(cache_split_dir, exist_ok=True)

    # 馃敡 鐗瑰緛缂撳瓨锛氬浐瀹氬彉鎹紙val transform = Resize + CenterCrop锛?
    full_dataset = FrameSelectorVideoDataset(cfg, split)
    total_videos = len(full_dataset)

    # 澶氬崱鍒囧垎锛氭瘡寮犲崱澶勭悊鑷繁鐨勫瓙闆嗭紝鏂囦欢鎸夊叏灞€绱㈠紩淇濆瓨
    if num_splits > 1:
        indices = list(range(total_videos))
        chunk = indices[rank::num_splits]
        dataset = torch.utils.data.Subset(full_dataset, chunk)
        logger.info(f"[{split}] rank={rank}/{num_splits}: "
                    f"processing {len(dataset)}/{total_videos} videos "
                    f"(global indices {chunk[0]}..{chunk[-1]})")
    else:
        dataset = full_dataset
        chunk = list(range(total_videos))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=getattr(cfg.DATA, "NUM_WORKERS", 4),
        pin_memory=True,
    )

    logger.info(f"[{split}] {len(dataset)} videos 鈫?extracting CLIP features...")

    labels_list = []
    extracted = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    for local_idx, (video, label) in enumerate(loader):
        idx = chunk[local_idx]  # 鍏ㄥ眬绱㈠紩
        out_path = os.path.join(cache_split_dir, f"{idx:08d}.pt")

        # 鏂偣缁紶
        if os.path.exists(out_path) and not overwrite:
            labels_list.append(label.item() if label.dim() == 0 else label[0].item())
            skipped += 1
            if (idx + 1) % 500 == 0:
                logger.info(f"  [{split}] {idx+1}/{len(dataset)} (skipped {skipped})")
            continue

        try:
            video = video.to(device, non_blocking=True)  # (1, T, 3, H, W)
            B, T, C, H, W = video.shape

            flat = video.view(B * T, C, H, W)           # (T, 3, H, W)
            feats = backbone(flat).float()               # (T, D), FP32

            # 馃敡 鐗瑰緛缂撳瓨锛欶P16 瀛樺偍锛岃妭鐪?50% 纾佺洏绌洪棿
            feats = feats.half().cpu()                   # (T, D), FP16

            torch.save(feats, out_path)
            labels_list.append(label.item() if label.dim() == 0 else label[0].item())
            extracted += 1

        except Exception as e:
            logger.error(f"  [ERROR] idx={idx}: {e}")
            traceback.print_exc()
            errors += 1
            # 瀛樼┖鐗瑰緛鍏滃簳
            dummy = torch.zeros(
                cfg.FRAME_SELECTOR.TOTAL_FRAMES,
                getattr(cfg.FRAME_SELECTOR, "FEAT_DIM", 512),
                dtype=torch.float16,
            )
            torch.save(dummy, out_path)
            labels_list.append(label.item() if label.dim() == 0 else label[0].item())

        if (local_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (local_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(dataset) - local_idx - 1) / speed if speed > 0 else 0
            logger.info(
                f"  [{split}] {local_idx+1}/{len(dataset)} "
                f"(+{extracted} / skipped {skipped} / err {errors}) "
                f"| {speed:.1f} v/s | ETA {datetime.timedelta(seconds=int(remaining))}"
            )

    # 澶氬崱妯″紡涓嬩粎 rank 0 淇濆瓨鍏冩暟鎹紙闈炲畬鏁达紝浠呭惈璇ュ崱瀛愰泦锛?
    if rank == 0 or num_splits == 1:
        if num_splits > 1:
            logger.warning(
                f"[{split}] Multi-GPU mode: metadata saved by rank 0 only "
                f"({len(labels_list)}/{total_videos} labels)"
            )
        meta = {
            "labels": labels_list,
            "total": total_videos,
            "extracted": extracted,
            "skipped": skipped,
            "errors": errors,
            "backbone": cfg.FRAME_SELECTOR.BACKBONE_NAME,
            "total_frames": cfg.FRAME_SELECTOR.TOTAL_FRAMES,
            "feat_dtype": "float16",
        }
        meta_path = os.path.join(os.path.dirname(cache_split_dir), f"{split}_metadata.pt")
        torch.save(meta, meta_path)

    total_time = time.time() - t0
    logger.info(
        f"[{split}] rank={rank}: +{extracted} extracted, {skipped} skipped, {errors} errors "
        f"| {str(datetime.timedelta(seconds=int(total_time)))}"
    )
    return extracted, skipped, errors


def main():
    args = parse_args()

    # 馃敡 澶氬崱鑷姩娲剧敓锛氭娴?GPU 鏁伴噺锛?1 鏃惰嚜鍔?spawn 瀛愯繘绋?
    num_gpus = args.num_gpus
    if num_gpus == 0:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        num_gpus = len(visible.split(',')) if visible else 1

    if num_gpus > 1:
        import subprocess
        import sys as _sys
        cmd = [_sys.executable] + _sys.argv.copy()
        while '--num-gpus' in cmd:
            i = cmd.index('--num-gpus')
            del cmd[i:i + 2]
        for flag in ['--num-splits', '--rank']:
            while flag in cmd:
                i = cmd.index(flag)
                del cmd[i:i + 2]
        cmd += ['--num-splits', str(num_gpus)]
        procs = []
        for rank in range(num_gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(rank)
            p = subprocess.Popen(cmd + ['--rank', str(rank)], env=env)
            procs.append(p)
            print(f"  [Spawn] rank={rank}, CUDA_VISIBLE_DEVICES={rank}, pid={p.pid}")
        for p in procs:
            p.wait()
        print("  [Done] All GPU workers finished.")
        return

    # 馃敡 浠?sys.argv 涓Щ闄?--splits / --num-splits / --rank锛岄伩鍏?Config 鍐呴儴 argparse 鎶ラ敊
    import sys as _sys
    for _flag in ['--splits', '--num-splits', '--rank']:
        while _flag in _sys.argv:
            _i = _sys.argv.index(_flag)
            del _sys.argv[_i:_i + 2]

    cfg = Config(load=True)

    # 馃敡 鐗瑰緛缂撳瓨锛氫粠閰嶇疆璇诲彇缂撳瓨鐩綍
    cache_cfg = getattr(cfg.DATA, "FEAT_CACHE", {})
    if isinstance(cache_cfg, dict):
        cache_root = cache_cfg.get("CACHE_DIR", "data/feat_cache/ssv2")
    else:
        cache_root = getattr(cache_cfg, "CACHE_DIR", "data/feat_cache/ssv2")

    tag = _cache_tag(cfg)
    cache_dir = os.path.join(cache_root, tag)
    os.makedirs(cache_dir, exist_ok=True)

    # 鏃ュ織
    logging.setup_logging(cfg, "extract_features.log")
    logger.info(f"Config: {args.cfg_file}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Tag: {tag} | Storage: FP16")

    # 鍔犺浇 CLIP 涓诲共
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = cfg.FRAME_SELECTOR.BACKBONE_NAME
    logger.info(f"Loading {backbone_name} on {device}...")

    clip_model, _ = clip.load(backbone_name, device="cpu")
    backbone = clip_model.visual.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Backbone loaded: {total_params:,} params (frozen)")

    # 纭畾瑕佹彁鍙栫殑 splits
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",")]
        logger.info(f"Using specified splits: {splits}")
    else:
        splits = ["train"]
        if getattr(cfg.DATA, "VAL_LIST", None):
            splits.append("val")
        if getattr(cfg.DATA, "TEST_LIST", None):
            splits.append("test")
        logger.info(f"Auto-detected splits: {splits}")

    total_new = 0
    for split in splits:
        split_dir = os.path.join(cache_dir, split)
        n, s, e = extract_split(cfg, split, backbone, split_dir, device,
                                 overwrite=args.overwrite,
                                 num_splits=args.num_splits, rank=args.rank)
        total_new += n

    logger.info("=" * 60)
    logger.info(f"Done! {total_new} features 鈫?{cache_dir}")
    logger.info(f"Now run training with FEAT_CACHE.ENABLE=True")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


