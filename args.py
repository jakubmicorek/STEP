"""
args.py
=======
Argument definitions for the STEP pipeline.

All defaults reflect the intended configuration.
Ablation flags are explicit (not hidden behind default mismatches).

Dataset path conventions
------------------------
ShanghaiTech / ShanghaiTech-HR:
    <data_dir>/ShanghaiTech/pose/{train,test}/
    <data_dir>/ShanghaiTech/{train,test}/images/

UBnormal:
    <data_dir>/UBnormal/pose/{train,validation,test}/
    <data_dir>/UBnormal/{train,validation,test}/

avenue:
    <data_dir>/avenue/pose/{train,test}/
    <data_dir>/avenue/{training,testing}/videos/

MSAD / MSAD-HR / MSAD-HR-ST-STYLE / MSAD-HR-UB-STYLE:
    <data_dir>/MSAD/pose/alphapose/{train,test}/
"""
from __future__ import annotations

import argparse
import os


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args()
    return _resolve_paths(args)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="STEP")

    # ------------------------------------------------------------------
    # Dataset / paths
    # ------------------------------------------------------------------
    p.add_argument("--dataset", default="UBnormal",
                   choices=["ShanghaiTech", "ShanghaiTech-HR",
                            "UBnormal", "UBnormal-HR", "avenue",
                            "MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--runs_dir", default="runs", help="Root directory for experiment outputs")

    # Optional explicit path overrides (skip dataset auto-detection)
    p.add_argument("--pose_path_train",    default=None)
    p.add_argument("--pose_path_validate", default=None)
    p.add_argument("--pose_path_test",     default=None)

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed. None (default) draws a fresh random seed "
                        "per run and records it in config.json.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True,
                   help="Save a checkpoint whenever validation AUC improves. "
                        "Pass --no-save_model to disable.")
    p.add_argument("--save_every_n_epochs", type=int, default=0,
                   help="Save a periodic checkpoint every N epochs. 0 = disabled.")
    p.add_argument("--force_build", action="store_true",
                   help="Rebuild segment cache even if a valid one exists")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    p.add_argument("--T", "--seg_len", dest="seg_len", type=int, default=12,
                   help="Temporal window T (seg_len). Default: 12")
    p.add_argument("--seg_stride", type=int,  default=1,
                   help="Stride between consecutive training segments")
    p.add_argument("--batch_size_features", type=int, default=1024,
                   help="Batch size for the DataLoader during pose feature extraction")

    # Confidence filtering at the data level (hard drop, ablation only)
    p.add_argument("--filter_conf",       type=float, default=0.0,
                   help="Drop frames with mean keypoint confidence < threshold. "
                        "0 = disabled (use soft confidence weighting instead).")
    p.add_argument("--train_seg_conf_th", type=float, default=0.0,
                   help="Drop entire segments whose mean confidence < threshold.")

    # ------------------------------------------------------------------
    # Projection (PCA / AE / VAE)
    # ------------------------------------------------------------------
    p.add_argument("--K", "--pca_dim", dest="pca_dim", type=int, default=48,
                   help="PCA bottleneck K (pca_dim). 0 = raw coordinates (ablation). "
                        "Default: 48")
    p.add_argument("--whiten", "--standardize", dest="standardize",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Whiten the PCA projection (standardize; unit variance per component).")

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    p.add_argument("--units", nargs="+", type=int, default=[1024, 1024, 1024, 1024],
                   help="Hidden layer units. All must be equal for residual connections.")
    p.add_argument("--layernorm", action=argparse.BooleanOptionalAction, default=False,
                   help="LayerNorm in the data path of the residual blocks. off (default).")

    # ------------------------------------------------------------------
    # Noise schedule (sigma)
    # ------------------------------------------------------------------
    p.add_argument("--L",          type=int,   default=10,
                   help="Number of sigma levels in the geometric schedule")
    p.add_argument("--sigma_low",  type=float, default=0.1,
                   help="Minimum noise level sigma_low. Default: 0.1")
    p.add_argument("--sigma_high", type=float, default=1.0,
                   help="Maximum noise level sigma_high. Default: 1.0")
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    p.add_argument("--epochs",        type=int,   default=400)
    p.add_argument("--lr",            type=float, default=5e-4,
                   help="Learning rate. Default: 5e-4 (use 2e-4 for ShanghaiTech / MSAD-HR)")
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Batch size for the EBM training loop")
    p.add_argument("--ema_decay",     type=float, default=0.999,
                   help="EMA decay for the inference model. 0 = disable EMA.")
    p.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=True,
                   help="Cosine annealing LR schedule with linear warmup over epoch 0.")
    p.add_argument("--gradient_clipping", type=float, default=None)

    p.add_argument("--network_jitter", type=float, default=0.1,
                   help="Gaussian noise on hidden activations between blocks. Default: 0.1")

    # ------------------------------------------------------------------
    # Confidence weighting
    # ------------------------------------------------------------------
    p.add_argument("--train_confidence_weighted",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Weight DSM loss by sequence confidence. "
                        "Disabling degrades soft confidence handling.")
    p.add_argument("--score_confidence_weighted",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Weight anomaly scores at inference by sequence confidence.")

    # ------------------------------------------------------------------
    # Validation / evaluation
    # ------------------------------------------------------------------
    p.add_argument("--check_val_every_n_epoch", type=int, default=20,
                   help="Run validation every N epochs. "
                        "0 = only at the end.")
    # Temporal smoothing applied to anomaly scores at eval time.
    # These are used for training-time checkpoint selection (single fixed point).
    # Default=None -> dataset-specific value set by _resolve_paths().
    p.add_argument("--temporal_gaussian_1d_smoothing_sigma_person", type=float, default=None,
                   help="Gaussian sigma applied along the time axis per person before "
                        "person-level score pooling. None -> dataset default. 0 = disabled.")
    p.add_argument("--temporal_gaussian_1d_smoothing_sigma_frame_level", type=float, default=None,
                   help="Gaussian sigma applied to the final per-frame score curve. "
                        "None -> dataset default. 0 = disabled.")

    return p


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Per-dataset defaults for temporal_gaussian_1d_smoothing_sigma_person and
# temporal_gaussian_1d_smoothing_sigma_frame_level.
#
# HR variants (UBnormal-HR, ShanghaiTech-HR) automatically inherit their base
# dataset's defaults via the _resolve_paths lookup - no separate entry needed.
#
# UBnormal  (person-sigma=1, frame-sigma=21):
#   UBnormal is a synthetic dataset with clean, long, stable person tracks.
#   A small amount of person-level smoothing (sigma=1) improves score continuity
#   without blurring across track boundaries, giving a consistent +2-3 pp gain.
#
# ShanghaiTech  (person-sigma=0, frame-sigma=13):
#   Real surveillance footage with short, frequently interrupted tracks and
#   lower keypoint confidence. Person-level smoothing mixes scores from
#   different detections on the same track slot, hurting discrimination.
#   Frame-level smoothing alone (sigma=13) is good.
_SMOOTHING_DEFAULTS: dict[str, tuple[int, int]] = {
    "UBnormal":         (1,  21),
    "ShanghaiTech":     (0,  13),
    "avenue":           (0,   0),
    "MSAD":             (0,   0),
    "MSAD-HR":          (0,   0),
    "MSAD-HR-ST-STYLE": (0,   0),
    "MSAD-HR-UB-STYLE": (0,   0),
}


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill args.pose_path and args.vid_path dicts from data_dir + dataset."""
    ds = args.dataset

    if ds == "avenue":
        base = os.path.join(args.data_dir, "avenue")
        args.vid_path = {
            "train":    os.path.join(base, "training",  "videos"),
            "validate": os.path.join(base, "testing",   "videos"),
            "test":     os.path.join(base, "testing",   "videos"),
        }
        args.pose_path = {
            "train":    os.path.join(base, "pose", "train"),
            "validate": os.path.join(base, "pose", "test"),
            "test":     os.path.join(base, "pose", "test"),
        }

    elif ds in ("MSAD", "MSAD-HR", "MSAD-HR-ST-STYLE", "MSAD-HR-UB-STYLE"):
        base = os.path.join(args.data_dir, "MSAD")
        args.vid_path = {
            "train":    os.path.join(base, "train_videos"),
            "validate": os.path.join(base, "test_videos"),
            "test":     os.path.join(base, "test_videos"),
        }
        args.pose_path = {
            "train":    os.path.join(base, "pose", "alphapose", "train"),
            "validate": os.path.join(base, "pose", "alphapose", "test"),
            "test":     os.path.join(base, "pose", "alphapose", "test"),
        }

    elif ds in ("UBnormal", "UBnormal-HR"):
        base = os.path.join(args.data_dir, "UBnormal")
        args.vid_path = {
            "train":    os.path.join(base, "train"),
            "validate": os.path.join(base, "validation"),
            "test":     os.path.join(base, "test"),
        }
        args.pose_path = {
            "train":    os.path.join(base, "pose", "train"),
            "validate": os.path.join(base, "pose", "validation"),
            "test":     os.path.join(base, "pose", "test"),
        }

    else:  # ShanghaiTech / ShanghaiTech-HR
        base = os.path.join(args.data_dir, "ShanghaiTech")
        args.vid_path = {
            "train":    os.path.join(base, "train",  "images"),
            "validate": os.path.join(base, "test",   "frames"),
            "test":     os.path.join(base, "test",   "frames"),
        }
        args.pose_path = {
            "train":    os.path.join(base, "pose", "train"),
            "validate": os.path.join(base, "pose", "test"),
            "test":     os.path.join(base, "pose", "test"),
        }

    # Explicit path overrides (e.g. for non-default pose extractors)
    if args.pose_path_train:
        args.pose_path["train"] = args.pose_path_train
    if args.pose_path_validate:
        args.pose_path["validate"] = args.pose_path_validate
    if args.pose_path_test:
        args.pose_path["test"] = args.pose_path_test

    # Apply dataset-specific smoothing defaults when the user did not override.
    # HR variants (e.g. UBnormal-HR, ShanghaiTech-HR) share defaults with their base dataset.
    _smooth_key = ds.replace("-HR", "") if ds.endswith("-HR") and ds.replace("-HR", "") in _SMOOTHING_DEFAULTS else ds
    p_def, f_def = _SMOOTHING_DEFAULTS.get(_smooth_key, (0, 0))
    if getattr(args, "temporal_gaussian_1d_smoothing_sigma_person", None) is None:
        args.temporal_gaussian_1d_smoothing_sigma_person = p_def
    if getattr(args, "temporal_gaussian_1d_smoothing_sigma_frame_level", None) is None:
        args.temporal_gaussian_1d_smoothing_sigma_frame_level = f_def

    return args
