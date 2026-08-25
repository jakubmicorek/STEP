"""
data/normalize.py
=================
Pose normalization and ShanghaiTech-HR filtering.

Normalization: zero-mean, scale by std of y-coordinates (per sample).
This is the same normalization used by SeeKer.
"""

import numpy as np


# (scene_id, clip_id) pairs to skip for ShanghaiTech-HR evaluation
SHANGHAITECH_HR_SKIP = [
    (1, 130), (1, 135), (1, 136),
    (6, 144), (6, 145),
    (12, 152),
]


def shanghaitech_hr_skip(is_hr: bool, scene_id, clip_id) -> bool:
    if not is_hr:
        return False
    return (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP


def normalize_pose(
    pose_data: np.ndarray,
    norm_mode: str = "y_axis",
    eps: float = 1e-6,
    **kwargs,
) -> np.ndarray:
    """
    Normalize skeleton windows.

    Parameters
    ----------
    pose_data : (T, V, C) or (B, T, V, C)
        Last channel is (x, y, confidence).  Confidence is left unchanged.
    norm_mode : "y_axis" | "per_axis"
        y_axis   - divide both x and y by std(y)  [default]
        per_axis - divide x by std(x), y by std(y)  [ablation]

    Returns same shape as input.
    """
    arr = np.asarray(pose_data)
    assert arr.ndim in (3, 4), f"Expected (T,V,C) or (B,T,V,C), got {arr.shape}"

    squeeze = arr.ndim == 3
    if squeeze:
        arr = arr[None]

    pd = arr.copy()
    mu_xy = pd[..., :2].mean(axis=(1, 2))   # (B, 2)

    if norm_mode == "per_axis":
        std_xy = np.clip(pd[..., :2].std(axis=(1, 2)), eps, None)
        pd[..., :2] = (pd[..., :2] - mu_xy[:, None, None, :]) / std_xy[:, None, None, :]
    else:
        std_y = np.clip(pd[..., 1].std(axis=(1, 2)), eps, None)
        pd[..., :2] = (pd[..., :2] - mu_xy[:, None, None, :]) / std_y[:, None, None, None]

    return pd[0] if squeeze else pd
