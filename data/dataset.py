"""
data/dataset.py
===============
PyTorch Dataset for skeleton-based anomaly detection.

Segments are cached under <pose_root>/.cache_segs/ as .npz files keyed by
a hash of (pose_root, relevant args).  Pass force_build=True to invalidate.
"""
from __future__ import annotations

import json
import os
import re
import time
import hashlib
import traceback
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .normalize import normalize_pose, shanghaitech_hr_skip
from .pose_utils import strip_pose_suffix
from .sequence_utils import clip_to_segments


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_CACHE_KEYS = ["dataset", "seg_len", "seg_stride", "train_seg_conf_th", "split", "filter_conf"]


def _stable_args(d: dict) -> dict:
    return {k: d.get(k) for k in _CACHE_KEYS}


def _cache_paths(root: str, args_dict: dict):
    cache_dir = Path(root) / ".cache_segs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    h.update(str(Path(root).resolve()).encode())
    h.update(json.dumps(_stable_args(args_dict), sort_keys=True).encode())
    key = h.hexdigest()[:16]
    return cache_dir / f"segs_{key}.npz", cache_dir / f"segs_{key}.meta.json"


def _json_manifest(root: str) -> list:
    files = sorted(f for f in os.listdir(root) if f.endswith("tracked_person.json"))
    out = []
    for f in files:
        p = Path(root) / f
        try:
            out.append([f, int(p.stat().st_mtime), int(p.stat().st_size)])
        except FileNotFoundError:
            pass
    return out


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        path_to_json_dir: str,
        path_to_vid_dir: str = None,
        evaluate: bool = False,
        filter_conf: float = 0.0,
        force_build: bool = False,
        **dataset_args,
    ):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        if evaluate:
            filter_conf = 0.0

        (
            self.segs_data_np,
            self.segs_meta,
            self.person_keys,
            self.global_data_np,
            self.global_data,
            self.segs_score_np,
        ) = _gen_dataset(path_to_json_dir, filter_conf=filter_conf, force_build=force_build, **dataset_args)

        self.segs_meta  = np.array(self.segs_meta)
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.metadata   = self.segs_meta
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seg = self.segs_data_np[idx]  # (C, T, V)
        seg = normalize_pose(
            seg.transpose(1, 2, 0)[None], **self.args  # -> (1, T, V, C)
        ).squeeze(0).transpose(2, 0, 1)                # -> (C, T, V)
        return [seg, self.segs_score_np[idx]]


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def get_dataset_and_loader(args):
    common = {
        "seg_len":          args.seg_len,
        "dataset":          args.dataset,
        "train_seg_conf_th": args.train_seg_conf_th,
    }
    force_build = getattr(args, "force_build", False)

    splits = ["train", "validate", "test"] if args.dataset in ("UBnormal", "UBnormal-HR") else ["train", "test"]

    dataset, loader = {}, {}

    for split in splits:
        evaluate = split in ("test", "validate")
        split_args = {**common, "seg_stride": args.seg_stride if split == "train" else 1, "split": split}

        dataset[split] = SkeletonSequenceDataset(
            args.pose_path[split],
            path_to_vid_dir=args.vid_path.get(split, ""),
            evaluate=evaluate,
            filter_conf=getattr(args, "filter_conf", 0.0),
            force_build=force_build,
            **split_args,
        )
        loader[split] = DataLoader(
            dataset[split],
            batch_size=args.batch_size_features,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=(split == "train"),
        )

    # For datasets without a validation split, alias test as validate
    if args.dataset not in ("UBnormal", "UBnormal-HR"):
        dataset["validate"] = dataset["test"]
        loader["validate"]  = loader["test"]

    return dataset, loader


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def _gen_dataset(person_json_root: str, filter_conf=None, force_build=False, **dataset_args):
    args_for_hash = {**dataset_args, "filter_conf": filter_conf}
    npz_path, meta_path = _cache_paths(person_json_root, args_for_hash)
    manifest = _json_manifest(person_json_root)

    # ---- Try cache -----------------------------------------------------------
    if not force_build and npz_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("manifest") == manifest and meta.get("args") == _stable_args(args_for_hash):
                print(f"Cache hit - loading from {npz_path} ...")
                t0  = time.time()
                npz = np.load(npz_path, allow_pickle=True)
                out = (
                    npz["segs_data_np"],
                    list(npz["segs_meta"]),
                    dict(npz["person_keys"].item()),
                    npz["global_data_np"],
                    list(npz["global_data"]),
                    npz["segs_score_np"],
                )
                print(f"Loaded in {time.time() - t0:.2f}s")
                return out
            else:
                print("Cache outdated - rebuilding ...")
        except Exception:
            print("Cache load failed - rebuilding ...")
            traceback.print_exc()

    # ---- Build ---------------------------------------------------------------
    t0           = time.time()
    seg_stride   = dataset_args.get("seg_stride", 1)
    seg_len      = dataset_args.get("seg_len",    12)
    seg_conf_th  = dataset_args.get("train_seg_conf_th", 0.0)
    ds_name      = dataset_args.get("dataset", "ShanghaiTech")

    json_list = sorted(f for f in os.listdir(person_json_root) if f.endswith("tracked_person.json"))
    print(f"Building dataset from {len(json_list)} JSON files ...")

    segs_data_np  = []
    segs_score_np = []
    segs_meta     = []
    global_data   = []
    person_keys   = {}

    for person_dict_fn in tqdm(json_list):
        base_name = strip_pose_suffix(person_dict_fn)
        try:
            scene_id, clip_id = _parse_scene_clip(base_name, ds_name, person_dict_fn)
        except _SkipClip:
            continue
        if scene_id is None:
            continue

        with open(os.path.join(person_json_root, person_dict_fn)) as f:
            clip_dict = json.load(f)

        clip_segs, clip_meta, clip_keys, _, _, clip_scores = clip_to_segments(
            clip_dict, 0, seg_stride, seg_len,
            scene_id=scene_id, clip_id=clip_id,
            dataset=ds_name, filter_conf=filter_conf,
        )

        if clip_segs.shape[0] == 0:
            print(f"  [WARN] No segments for {person_dict_fn} (clip shorter than seg_len={seg_len}?)")

        _, _, _, global_np, global_data, _ = clip_to_segments(
            clip_dict, 0, 1, 1,
            scene_id=scene_id, clip_id=clip_id,
            global_pose_data=global_data,
            dataset=ds_name, filter_conf=filter_conf,
        )

        segs_data_np.append(clip_segs)
        segs_score_np.append(clip_scores)
        segs_meta += clip_meta
        person_keys.update(clip_keys)

    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    segs_data_np   = np.concatenate(segs_data_np,  axis=0)
    segs_score_np  = np.concatenate(segs_score_np, axis=0)

    # COCO17 -> COCO18 (add neck keypoint, reorder)
    segs_data_np   = _kps17_to_coco18(segs_data_np)
    global_data_np = _kps17_to_coco18(global_data_np)
    global_data    = [_kps17_to_coco18(d) for d in global_data]

    # (N, T, V, C) -> (N, C, T, V)
    segs_data_np   = np.transpose(segs_data_np,   (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = _seg_conf_filter(
            segs_data_np, segs_meta, segs_score_np, seg_conf_th
        )

    print(f"Dataset built in {time.time() - t0:.2f}s  ({len(segs_data_np)} segments)")

    # ---- Save cache ----------------------------------------------------------
    print(f"Saving cache to {npz_path} ...")
    try:
        np.savez_compressed(
            npz_path,
            segs_data_np   = segs_data_np,
            segs_meta      = np.array(segs_meta, dtype=object),
            person_keys    = np.array(person_keys, dtype=object),
            global_data_np = global_data_np,
            global_data    = np.array(global_data, dtype=object),
            segs_score_np  = segs_score_np,
        )
        with open(meta_path, "w") as f:
            json.dump({"args": _stable_args(args_for_hash), "manifest": manifest, "created": int(time.time())}, f)
        print("Cache saved.")
    except Exception as e:
        print(f"WARNING: Failed to save cache: {e}")
        traceback.print_exc()

    return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np


# ---------------------------------------------------------------------------
# Per-dataset scene/clip parsing
# ---------------------------------------------------------------------------

class _SkipClip(Exception):
    pass


def _parse_scene_clip(base_name: str, dataset: str, filename: str):
    if dataset in ("UBnormal", "UBnormal-HR"):
        m = re.findall(r"(abnormal|normal)_scene_(\d+)_scenario(.*)$", base_name)
        if not m:
            print(f"  [WARN] Regex failed for {filename}")
            return None, None
        type_str, scene_id, clip_id = m[0]
        return scene_id, type_str + "_" + clip_id

    elif dataset == "avenue":
        m = re.match(r"^(\d{1,2})", base_name)
        if not m:
            print(f"  [WARN] Could not parse avenue clip id from {filename}")
            return None, None
        cid = f"{int(m.group(1)):02d}"
        return cid, cid

    elif dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
        m = re.match(r"^([A-Za-z][A-Za-z0-9_\-]*?)(\d+)$", base_name)
        if not m:
            print(f"  [WARN] Regex failed for MSAD file: {filename}")
            return None, None
        return m.group(1), m.group(2)

    else:  # ShanghaiTech / ShanghaiTech-HR
        parts = base_name.split("_")
        if len(parts) < 2:
            print(f"  [WARN] Cannot parse scene/clip from {filename}")
            return None, None
        scene_id, clip_id = parts[0], parts[1]
        if shanghaitech_hr_skip(dataset == "ShanghaiTech-HR", scene_id, clip_id):
            raise _SkipClip
        return scene_id, clip_id


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_COCO18_ORDER = np.array(
    [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3], dtype=np.int64
)


def _kps17_to_coco18(kps: np.ndarray) -> np.ndarray:
    """
    Convert 17-keypoint COCO skeleton to 18-keypoint COCO18.
    Neck = average of left/right shoulders (indices 5 & 6).
    Reorders to the COCO18 convention used by STG-NF / SeeKer.
    """
    arr  = np.array(kps)
    neck = 0.5 * (arr[..., 5, :] + arr[..., 6, :])
    arr  = np.concatenate([arr, neck[..., None, :]], axis=-2)
    return arr[..., _COCO18_ORDER, :]


def _seg_conf_filter(segs_data_np, segs_meta, segs_score_np, conf_th: float):
    mask = segs_score_np.mean(axis=1) > conf_th
    return segs_data_np[mask], list(np.array(segs_meta)[mask]), segs_score_np[mask]
