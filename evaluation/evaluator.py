"""
Frame-level anomaly evaluation: GT loading, score flattening, and AUC/AP metrics.
"""
from __future__ import annotations

import os
import re
import glob

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score

from .scoring import fill_and_smooth_fw
from data.normalize import shanghaitech_hr_skip
from data.pose_utils import strip_pose_suffix


MSAD_HR_CLASSES = {
    "Assault", "Fighting", "People_falling", "Robbery",
    "Shooting", "Traffic_accident", "Vandalism",
}
MSAD_NON_HR_CLASSES = {
    "Explosion", "Fire", "Object_falling", "Water_incident",
}
MSAD_ALL_ABNORMAL_CLASSES = MSAD_HR_CLASSES | MSAD_NON_HR_CLASSES


class PoseAnomalyEvaluator:
    def __init__(self, args, split: str = "test"):
        self.args     = args
        self.dataset  = args.dataset
        self.data_dir = args.data_dir
        self.split    = split
        self.clip_path_map = self._build_gt_map()

    # ------------------------------------------------------------------
    # GT map
    # ------------------------------------------------------------------

    def _build_gt_map(self) -> dict:
        map_dict = {}

        if self.dataset in ("UBnormal", "UBnormal-HR"):
            gt_root = os.path.join(self.args.data_dir, "UBnormal_labels")
            for fpath in glob.glob(os.path.join(gt_root, "**", "*.npy"), recursive=True):
                parent = os.path.basename(os.path.dirname(fpath))
                map_dict[parent] = fpath

        elif self.dataset in ("ShanghaiTech", "ShanghaiTech-HR"):
            gt_root = os.path.join(self.data_dir, "ShanghaiTech", "gt", "test_frame_mask")
            if os.path.isdir(gt_root):
                for fn in os.listdir(gt_root):
                    if fn.endswith(".npy"):
                        map_dict[fn[:-4]] = os.path.join(gt_root, fn)

        elif self.dataset == "avenue":
            gt_root = os.path.join(self.data_dir, "avenue", "ground_truth_demo", "testing_label_mask")
            if os.path.isdir(gt_root):
                for fn in os.listdir(gt_root):
                    if fn.endswith("_label.mat"):
                        raw = fn[:-len("_label.mat")]
                        try:
                            key = f"{int(raw):02d}"
                        except ValueError:
                            key = raw
                        map_dict[key] = os.path.join(gt_root, fn)

        elif self.dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
            gt_root = os.path.join(self.data_dir, "MSAD", "gt", "test_frame_mask")
            if os.path.isdir(gt_root):
                for fn in os.listdir(gt_root):
                    if fn.endswith(".npy"):
                        clip_name = fn[:-4]
                        if self._should_include_msad_clip(clip_name):
                            map_dict[clip_name] = os.path.join(gt_root, fn)
            else:
                raise FileNotFoundError(
                    f"MSAD GT directory not found: {gt_root}"
                )
        return map_dict

    # ------------------------------------------------------------------
    # GT loading
    # ------------------------------------------------------------------

    def _load_gt(self, gt_path: str):
        if not os.path.exists(gt_path):
            return None
        if self.dataset == "avenue":
            try:
                mat = loadmat(gt_path)
                return (mat["volLabel"][0] == 1).astype(np.int32)
            except Exception:
                return None
        try:
            return np.load(gt_path, allow_pickle=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Clip-ID parsing
    # ------------------------------------------------------------------

    def _parse_clip_id(self, clip_name: str):
        if self.dataset in ("UBnormal", "UBnormal-HR"):
            m = re.findall(r"(abnormal|normal)_scene_(\d+)_scenario(.*)", clip_name)
            if not m:
                return None, None
            type_str, scene_id_str, suffix = m[0]
            scene_id   = int(scene_id_str)
            clip_id    = type_str + "_" + suffix
            candidates = [clip_id]
            if not suffix.startswith("_"):
                candidates.append(type_str + "__" + suffix)
            if suffix.startswith("_"):
                candidates.append(type_str + "_" + suffix[1:])
            return scene_id, candidates

        elif self.dataset in ("ShanghaiTech", "ShanghaiTech-HR"):
            try:
                parts        = clip_name.split("_")
                scene_id_int = int(parts[0])
                clip_id_int  = int(parts[1])
            except (IndexError, ValueError):
                return None, None
            if shanghaitech_hr_skip(self.dataset == "ShanghaiTech-HR", scene_id_int, clip_id_int):
                return None, None
            return scene_id_int, [str(clip_id_int)]

        elif self.dataset == "avenue":
            try:
                return int(clip_name), [str(int(clip_name))]
            except ValueError:
                return None, None

        elif self.dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
            m = re.match(r"^([A-Za-z][A-Za-z0-9_\-]*?)(\d+)$", clip_name)
            if not m:
                return None, None
            return m.group(1), [m.group(2)]

        return None, None

    # ------------------------------------------------------------------
    # MSAD helpers
    # ------------------------------------------------------------------

    def _msad_anomaly_class(self, clip_name: str):
        for cls in sorted(MSAD_ALL_ABNORMAL_CLASSES, key=len, reverse=True):
            if clip_name.startswith(f"{cls}_"):
                return cls
        return None

    def _is_msad_dataset(self) -> bool:
        return self.dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE")

    def _is_msad_hr_st_style(self) -> bool:
        return self.dataset in ("MSAD-HR", "MSAD-HR-ST-STYLE")

    def _is_msad_hr_ub_style(self) -> bool:
        return self.dataset == "MSAD-HR-UB-STYLE"

    def _should_include_msad_clip(self, clip_name: str) -> bool:
        if not self._is_msad_hr_st_style():
            return True
        cls = self._msad_anomaly_class(clip_name)
        return cls is None or cls in MSAD_HR_CLASSES

    def _select_clip_names(self, clip_names):
        if not self._is_msad_dataset():
            return clip_names, None
        selected = [c for c in clip_names if self._should_include_msad_clip(c)]
        return selected, None

    # ------------------------------------------------------------------
    # UBnormal HR mask
    # ------------------------------------------------------------------

    def _load_ubnormal_hr_mask(self, clip_name: str):
        m = re.search(r"(normal|abnormal)_scene_(\d+)_scenario_?(\d+)(?:_(.+))?", clip_name)
        if not m:
            raise ValueError(f"Failed to parse clip_name: '{clip_name}'.")
        scene_type, scene_num, scenario_num, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        c1  = 1 if scene_type == "normal" else 0
        c3c4 = "00"
        if suffix:
            env_map = {"fog": "51", "fire": "52", "smoke": "53"}
            c3c4 = env_map.get(suffix, f"{int(suffix):02d}" if suffix.isdigit() else "00")
        mapped_fn  = f"{c1}{scene_num:02d}_{scenario_num:02d}{c3c4}.npy"
        split_dir  = "testing" if self.split == "test" else "validating"
        path       = os.path.join(self.args.data_dir, "UBnormal_hr_bool_masks", split_dir, "test_frame_mask", mapped_fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"HR mask not found for '{clip_name}'. Expected: {path}")
        return np.load(path)

    # ------------------------------------------------------------------
    # Frame-level flattening
    # ------------------------------------------------------------------

    def flatten_dataset(
        self,
        raw_scores: np.ndarray,
        metadata,
        person_sigmas: tuple = (0,),
        apply_hr: bool = False,
    ) -> dict:
        apply_hr = apply_hr or (self.dataset == "UBnormal-HR")
        metadata_np = np.array(metadata, dtype=object)

        flat = {"gt": [], "clip_ids": []}
        for p_sig in person_sigmas:
            flat[f"Backtrack_P{p_sig}"]      = []
            flat[f"Online_Center_P{p_sig}"]  = []
            flat[f"Online_Latest_P{p_sig}"]  = []

        pose_dir = self._pose_dir()
        if not os.path.isdir(pose_dir):
            return flat

        clip_names, _ = self._select_clip_names(self._clip_names_from_pose_dir(pose_dir))
        current_clip  = 0

        for clip_name in clip_names:
            if clip_name not in self.clip_path_map:
                raise FileNotFoundError(
                    f"Missing ground truth for clip '{clip_name}' "
                    f"({self.dataset}, split={self.split}). "
                    f"No GT file found under {self.data_dir} - refusing to "
                    "evaluate without it."
                )

            hr_mask = None
            if apply_hr:
                if self.dataset in ("ShanghaiTech", "ShanghaiTech-HR"):
                    sid, cands = self._parse_clip_id(clip_name)
                    if cands and shanghaitech_hr_skip(True, sid, int(cands[0])):
                        continue
                elif self.dataset in ("UBnormal", "UBnormal-HR"):
                    try:
                        hr_mask = self._load_ubnormal_hr_mask(clip_name)
                    except (FileNotFoundError, ValueError) as e:
                        raise FileNotFoundError(
                            f"{e}  Cannot evaluate '{clip_name}' in HR mode "
                            "without its HR boolean mask."
                        ) from e

            gt = self._load_gt(self.clip_path_map[clip_name])
            if gt is None:
                raise RuntimeError(
                    f"Failed to load ground truth for clip '{clip_name}' "
                    f"({self.dataset}, split={self.split}). "
                    f"Expected file: {self.clip_path_map[clip_name]}"
                )

            if self._is_msad_hr_ub_style():
                cls = self._msad_anomaly_class(clip_name)
                if cls in MSAD_NON_HR_CLASSES:
                    gt = np.zeros_like(gt)

            video_len = gt.shape[0]
            scene_id, cands = self._parse_clip_id(clip_name)
            if cands is None:
                continue

            clip_inds = self._find_segment_indices(metadata_np, scene_id, cands)

            vid_back   = {s: np.full(video_len, -np.inf) for s in person_sigmas}
            vid_center = {s: np.full(video_len, -np.inf) for s in person_sigmas}
            vid_latest = {s: np.full(video_len, -np.inf) for s in person_sigmas}

            if clip_inds:
                unique_pids = {metadata[i][2] for i in clip_inds}

                for pid in unique_pids:
                    p_inds = [i for i in clip_inds if metadata[i][2] == pid]
                    if not p_inds:
                        continue

                    s_t    = raw_scores[p_inds]
                    starts = np.array([metadata[i][3] for i in p_inds], dtype=int)
                    dummy  = np.zeros((len(s_t), self.args.seg_len))
                    dummy[:, -1] = s_t

                    # Backtrack
                    s_off, inds_off = fill_and_smooth_fw(dummy, starts, self.args, sigma=0, final_idx=video_len - 1)
                    p_back = np.full(video_len, -np.inf)
                    valid  = (inds_off >= 0) & (inds_off < video_len)
                    if valid.any():
                        p_back[inds_off[valid]] = s_off[valid]

                    # Online-Center (STG-NF): score at center frame of segment
                    p_center = np.full(video_len, -np.inf)
                    for k, start in enumerate(starts):
                        c_f = start + self.args.seg_len // 2
                        if c_f < video_len:
                            p_center[c_f] = max(p_center[c_f], s_t[k])

                    # Online-Latest (ours): score at last frame of segment
                    p_latest = np.full(video_len, -np.inf)
                    for k, start in enumerate(starts):
                        end_f = start + self.args.seg_len - 1
                        if end_f < video_len:
                            p_latest[end_f] = max(p_latest[end_f], s_t[k])

                    def _numeric(arr):
                        a2  = arr.copy()
                        fin = arr[arr != -np.inf]
                        floor = (fin.min() - 1.0) if len(fin) > 0 else -100.0
                        a2[arr == -np.inf] = floor
                        return a2

                    pb_n = _numeric(p_back)
                    pc_n = _numeric(p_center)
                    pl_n = _numeric(p_latest)

                    for p_sig in person_sigmas:
                        if p_sig == 0:
                            sb, sc, sl = p_back, p_center, p_latest
                        else:
                            sb = gaussian_filter1d(pb_n, sigma=p_sig)
                            sc = gaussian_filter1d(pc_n, sigma=p_sig)
                            sl = gaussian_filter1d(pl_n, sigma=p_sig)
                        vid_back[p_sig]   = np.maximum(vid_back[p_sig], sb)
                        vid_center[p_sig] = np.maximum(vid_center[p_sig], sc)
                        vid_latest[p_sig] = np.maximum(vid_latest[p_sig], sl)

            if apply_hr and self.dataset in ("UBnormal", "UBnormal-HR") and hr_mask is not None:
                hr_mask = hr_mask.flatten().astype(bool)
                min_len = min(video_len, len(hr_mask))
                gt      = gt[:min_len][hr_mask]
                ids_arr = np.full(min_len, current_clip)[hr_mask]
                for p_sig in person_sigmas:
                    vid_back[p_sig]   = vid_back[p_sig][:min_len][hr_mask]
                    vid_center[p_sig] = vid_center[p_sig][:min_len][hr_mask]
                    vid_latest[p_sig] = vid_latest[p_sig][:min_len][hr_mask]
            else:
                ids_arr = np.full(video_len, current_clip)

            flat["gt"].append(gt)
            flat["clip_ids"].append(ids_arr)
            for p_sig in person_sigmas:
                flat[f"Backtrack_P{p_sig}"].append(vid_back[p_sig])
                flat[f"Online_Center_P{p_sig}"].append(vid_center[p_sig])
                flat[f"Online_Latest_P{p_sig}"].append(vid_latest[p_sig])
            current_clip += 1

        for k in flat:
            flat[k] = np.concatenate(flat[k]) if flat[k] else np.array([])

        return flat

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calc_metrics(self, gt: np.ndarray, scores: np.ndarray) -> dict:
        scores = scores.copy()
        fin    = scores[np.isfinite(scores)]
        if len(fin):
            scores[scores ==  np.inf] = fin.max() + 1
            scores[scores == -np.inf] = fin.min() - 1
        else:
            scores[scores ==  np.inf] = 1.0
            scores[scores == -np.inf] = 0.0
        try:
            return {"AUC": roc_auc_score(gt, scores), "AP": average_precision_score(gt, scores)}
        except ValueError:
            return {"AUC": 0.0, "AP": 0.0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pose_dir(self) -> str:
        pose_map = getattr(self.args, "pose_path", None)
        for alias in [self.split, "validate" if self.split == "validation" else "validation"]:
            if isinstance(pose_map, dict) and pose_map.get(alias):
                return pose_map[alias]
        ds = self.dataset
        base = self.data_dir
        if ds in ("ShanghaiTech", "ShanghaiTech-HR"):
            return os.path.join(base, "ShanghaiTech", "pose", self.split)
        elif ds in ("UBnormal", "UBnormal-HR"):
            return os.path.join(base, "UBnormal", "pose", self.split)
        elif ds == "avenue":
            return os.path.join(base, "avenue", "pose", self.split)
        return os.path.join(base, ds, "pose", self.split)

    @staticmethod
    def _clip_names_from_pose_dir(pose_dir: str) -> list:
        files = sorted(f for f in os.listdir(pose_dir) if f.endswith("tracked_person.json"))
        return [strip_pose_suffix(f) for f in files]

    def _find_segment_indices(self, metadata_np, scene_id, candidates: list) -> list:
        for cid in candidates:
            if self.dataset in ("UBnormal", "UBnormal-HR"):
                inds = np.where((metadata_np[:, 1] == cid) & (metadata_np[:, 0] == scene_id))[0]
            elif self.dataset in ("ShanghaiTech", "ShanghaiTech-HR"):
                inds = np.where((metadata_np[:, 1] == int(cid)) & (metadata_np[:, 0] == int(scene_id)))[0]
            elif self.dataset == "avenue":
                inds = np.where(metadata_np[:, 1] == int(cid))[0]
            elif self.dataset in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
                inds = np.where(
                    (metadata_np[:, 1].astype(str) == str(cid)) & (metadata_np[:, 0] == scene_id)
                )[0]
            else:
                inds = np.array([], dtype=int)
            if len(inds):
                return list(inds)
        return []
