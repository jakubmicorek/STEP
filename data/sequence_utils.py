"""
data/sequence_utils.py
======================
Sliding-window segmentation of per-person pose tracks.

Compatible with AlphaPose JSON format used by STG-NF and SeeKer.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clip_to_segments(
    clip_dict,
    start_ofst: int = 0,
    seg_stride: int = 1,
    seg_len: int = 12,
    scene_id="",
    clip_id="",
    global_pose_data=None,
    dataset: str = "ShanghaiTech",
    filter_conf=None,
):
    """
    Generate sliding-window segments from a single clip's pose dictionary.

    Parameters
    ----------
    clip_dict : dict  {track_id: {frame_str: {keypoints, scores}}}
    start_ofst : int  first segment start offset
    seg_stride : int  stride between consecutive segments
    seg_len    : int  frames per segment
    scene_id   : str
    clip_id    : str
    global_pose_data : list  accumulator for global (stride-1) pose arrays
    dataset    : str
    filter_conf : float | None  drop frames with mean keypoint conf < threshold

    Returns
    -------
    pose_segs_data_np  : (N, seg_len, V, C)
    pose_segs_meta     : list of [scene_id, clip_id, person_id, start_frame]
    person_keys        : dict {key_str: sorted_frame_keys}
    global_data_np     : (T, 17, 3)  (concatenated if global_pose_data was provided)
    global_pose_data   : updated accumulator
    score_segs_data_np : (N, seg_len)
    """
    if global_pose_data is None:
        global_pose_data = []

    pose_segs_data  = []
    score_segs_data = []
    pose_segs_meta  = []
    person_keys     = {}

    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = \
            single_pose_dict2np(clip_dict, idx)

        key = _build_person_key(dataset, scene_id, clip_id, idx)
        person_keys[key] = sing_pose_keys

        curr_segs_np, curr_segs_meta, curr_score_np = split_person_sequence_to_segments(
            sing_pose_np, sing_pose_meta, sing_pose_keys,
            start_ofst, seg_stride, seg_len,
            scene_id=scene_id, clip_id=clip_id,
            single_score_np=sing_scores_np,
            dataset=dataset,
            filter_conf=filter_conf,
        )

        if curr_segs_np.shape[0] == 0:
            continue

        pose_segs_data.append(curr_segs_np)
        score_segs_data.append(curr_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_segs_meta

    kp_count, kp_dim = 17, 3
    if pose_segs_data:
        pose_segs_data_np  = np.concatenate(pose_segs_data,  axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    else:
        pose_segs_data_np  = np.empty(0).reshape(0, seg_len, kp_count, kp_dim)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)

    global_data_np = (
        np.concatenate(global_pose_data, axis=0)
        if global_pose_data
        else np.empty(0).reshape(0, kp_count, kp_dim)
    )

    return (
        pose_segs_data_np,
        pose_segs_meta,
        person_keys,
        global_data_np,
        global_pose_data,
        score_segs_data_np,
    )


def single_pose_dict2np(person_dict, idx):
    """Convert one person's sub-dict to numpy arrays."""
    single_person = person_dict[str(idx)]
    if isinstance(single_person, list):
        merged = {}
        for sub_dict in single_person:
            merged.update(sub_dict)
        single_person = merged

    keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(keys[0])]

    poses, scores = [], []
    for key in keys:
        poses.append(np.array(single_person[key]["keypoints"]).reshape(-1, 3))
        scores.append(single_person[key]["scores"])

    return (
        np.stack(poses,  axis=0),
        sing_pose_meta,
        keys,
        np.stack(scores, axis=0),
    )


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    start_idx = sorted_seg_keys.index(start_key)
    expected  = set(range(start_key, start_key + seg_len))
    actual    = set(sorted_seg_keys[start_idx: start_idx + seg_len])
    return len(actual.intersection(expected)) >= seg_len - missing_th


def impute_missing_frames(single_pose_np, single_pose_keys, seg_len):
    """Linear interpolation over short gaps (<= seg_len) in a pose track."""
    diff = np.diff(single_pose_keys)
    missing = np.where(diff > 1)[0]

    if len(missing) == 0:
        return single_pose_np, single_pose_keys

    splits_keys, splits_pose = [], []
    l = 0
    for i in missing:
        splits_keys.append(single_pose_keys[l: i + 1])
        splits_pose.append(single_pose_np[l: i + 1])
        length = diff[i] - 1
        last   = single_pose_keys[i]
        if length > seg_len:
            l = i + 1
            continue
        splits_keys.append(np.arange(last + 1, last + length + 1))
        a, b = single_pose_np[i], single_pose_np[i + 1]
        splits_pose.append(
            np.array([a + (b - a) * t / (length + 1) for t in range(1, length + 1)])
        )
        l = i + 1

    splits_keys.append(single_pose_keys[l:])
    splits_pose.append(single_pose_np[l:])

    return (
        np.concatenate(splits_pose),
        np.concatenate(splits_keys).astype(int).tolist(),
    )


def split_person_sequence_to_segments(
    single_pose_np,
    single_pose_meta,
    single_pose_keys,
    start_ofst: int = 0,
    seg_dist: int = 1,
    seg_len: int = 12,
    scene_id="",
    clip_id="",
    single_score_np=None,
    dataset: str = "ShanghaiTech",
    filter_conf=None,
):
    single_pose_keys = sorted([int(k) for k in single_pose_keys])

    if filter_conf is not None and filter_conf > 0:
        keep = single_pose_np.mean(axis=1)[:, 2] > filter_conf
        single_pose_np   = single_pose_np[keep]
        single_pose_keys = np.array(single_pose_keys)[keep].tolist()

    diff = np.diff(single_pose_keys)
    if len(diff) > 0 and not np.all(diff == 1):
        single_pose_np, single_pose_keys = impute_missing_frames(
            single_pose_np, single_pose_keys, seg_len
        )
        single_score_np = single_pose_np[..., -1][:, 0]

    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np   = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np  = np.empty([0, seg_len])
    pose_segs_meta = []

    num_segs = max(0, int(np.ceil((clip_t - seg_len) / seg_dist)))

    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        if start_ind >= len(single_pose_keys):
            break
        start_key = single_pose_keys[start_ind]

        if not is_seg_continuous(single_pose_keys, start_key, seg_len):
            continue

        curr_seg   = single_pose_np[start_ind: start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
        curr_score = single_score_np[start_ind: start_ind + seg_len].reshape(1, seg_len)
        pose_segs_np   = np.append(pose_segs_np,  curr_seg,   axis=0)
        pose_score_np  = np.append(pose_score_np, curr_score, axis=0)
        pose_segs_meta.append(
            _build_segment_meta(dataset, scene_id, clip_id, single_pose_meta, start_key)
        )

    return pose_segs_np, pose_segs_meta, pose_score_np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_person_key(dataset: str, scene_id, clip_id, idx) -> str:
    if dataset in ("UBnormal", "UBnormal-HR"):
        return "{:02d}_{}_{:02d}".format(int(scene_id), clip_id, int(idx))
    elif dataset == "avenue":
        return "{:02d}".format(int(clip_id))
    elif dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
        return "{}{}_{}".format(scene_id, clip_id, int(idx))
    else:
        return "{:02d}_{:04d}_{:02d}".format(int(scene_id), int(clip_id), int(idx))


def _build_segment_meta(dataset: str, scene_id, clip_id, single_pose_meta, start_key):
    person_id = int(single_pose_meta[0])
    start_key = int(start_key)
    if dataset in ("UBnormal", "UBnormal-HR"):
        return [int(scene_id), clip_id, person_id, start_key]
    elif dataset == "avenue":
        return [int(scene_id), int(clip_id), person_id, start_key]
    elif dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
        return [scene_id, int(clip_id), person_id, start_key]
    else:
        return [int(scene_id), int(clip_id), person_id, start_key]
