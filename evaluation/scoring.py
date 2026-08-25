"""
evaluation/scoring.py
=====================
Backtrack score filling and optional Gaussian smoothing for per-person frame score arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _is_consecutive(inds: np.ndarray) -> bool:
    return bool(np.all(np.diff(np.sort(inds)) == 1))


def fill_and_smooth_fw(
    s_t: np.ndarray,
    pid_frame_inds: np.ndarray,
    args,
    sigma: int = 1,
    final_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill and smooth per-person frame scores.

    Parameters
    ----------
    s_t            : (N_segs, seg_len)  - per-segment scores
    pid_frame_inds : (N_segs,)          - start frame index of each segment
    args           : namespace with .seg_len
    sigma          : smoothing kernel size (0 or 1 = no smoothing)
    final_idx      : clip last frame index (for truncation in non-consecutive case)

    Returns
    -------
    (scores, frame_indices)
    """
    if _is_consecutive(pid_frame_inds):
        # Extend index to the right by seg_len (for the last window's tail)
        pid_frame_inds_ = np.concatenate([
            pid_frame_inds,
            np.arange(pid_frame_inds[-1] + 1, pid_frame_inds[-1] + args.seg_len + 1)
        ])
        # Prepend seg_len copies of the first segment's score (Backtrack fill)
        first_score = s_t[:, -1][0]
        first_pad   = np.ones(args.seg_len) * first_score
        s = np.concatenate([first_pad, s_t[:, -1]], axis=0)
        s = gaussian_filter1d(s, sigma=sigma) if sigma > 1 else s

    else:
        _scores = s_t[:, -1]
        diff    = np.diff(pid_frame_inds)
        split_indices = np.where(diff != 1)[0]

        # Split into consecutive sub-sequences
        splits_ind, splits_scores = [], []
        l = 0
        for si in split_indices:
            splits_ind.append(pid_frame_inds[l: si + 1])
            splits_scores.append(_scores[l: si + 1])
            l = si + 1
        splits_ind.append(pid_frame_inds[l:])
        splits_scores.append(_scores[l:])

        splits_ind_, splits_scores_ = [], []
        for i in range(len(splits_ind)):
            ind = splits_ind[i]
            sc  = splits_scores[i]

            # Length of the gap BEFORE this sub-sequence.
            # For the first sub-sequence there is no preceding gap -> length 0.
            if i == 0:
                length = 0
            else:
                length = int(diff[split_indices[i - 1]]) - 1

            if length < args.seg_len:
                tail = np.arange(ind[-1] + 1, ind[-1] + args.seg_len + 1)[-length:] if length > 0 else np.array([], dtype=int)
                splits_ind_.append(np.concatenate((ind, tail)))
            else:
                length = args.seg_len
                splits_ind_.append(np.concatenate((ind, np.arange(ind[-1] + 1, ind[-1] + args.seg_len + 1))))

            pad = np.ones(length) * sc[0] if length > 0 else np.array([])
            arr = np.concatenate((pad, sc))
            splits_scores_.append(gaussian_filter1d(arr, sigma=sigma) if sigma > 1 else arr)

        s               = np.concatenate(splits_scores_)
        pid_frame_inds_ = np.concatenate(splits_ind_)

        if final_idx is not None:
            mask            = pid_frame_inds_ <= final_idx
            s               = s[mask]
            pid_frame_inds_ = pid_frame_inds_[mask]

    return s, pid_frame_inds_
