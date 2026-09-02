"""
train.py
========
Main training entry point for STEP.

Usage
-----
# Default dataset - UBnormal
python train.py --dataset UBnormal

# ShanghaiTech (lower LR)
python train.py --dataset ShanghaiTech --lr 2e-4
"""
from __future__ import annotations

import os
import sys
import json
import datetime
import random
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from args import get_args
from data import get_dataset_and_loader
from models.step import STEPNetwork
from training import (
    fit_pca, apply_pca, fit_standardize, apply_standardize, save_pca,
    compute_energy_stats, compute_score_matrix,
    STEPTrainer,
)
from evaluation import PoseAnomalyEvaluator


_PROTOCOLS = ("Backtrack", "Online_Center", "Online_Latest")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(loader):
    """Extract (pose_coords, mean_conf) tensors from a DataLoader."""
    all_vecs, all_confs = [], []
    for batch in loader:
        data = batch[0].float()           # (B, C, T, V)
        poses = data[:, :2].reshape(data.shape[0], -1)
        confs = data[:, 2].mean(dim=(1, 2))
        all_vecs.append(poses.cpu().numpy().astype(np.float32))
        all_confs.append(confs.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(all_vecs,  axis=0),
        np.concatenate(all_confs, axis=0),
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic):  return obj.item()
        if isinstance(obj, torch.device): return str(obj)
        return super().default(obj)


def save_checkpoint(save_dir, model, pca_mat, tr_mean, tr_std, energy_stats, args):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    if pca_mat is not None:
        save_pca(pca_mat, os.path.join(save_dir, "pca_matrix.faiss"))
    if tr_mean is not None:
        np.savez(os.path.join(save_dir, "feature_stats.npz"), mean=tr_mean, std=tr_std)
    clean_stats = {
        str(float(s)): {"mean": float(v["mean"]), "std": float(v["std"])}
        for s, v in energy_stats.items()
    }
    with open(os.path.join(save_dir, "energy_stats.json"), "w") as f:
        json.dump(clean_stats, f, indent=4)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # Reproducibility: no --seed given -> draw a fresh random seed and record
    # it in args (saved to config.json), so the run can be reproduced by
    # re-passing the printed seed.
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"[seed] No --seed given - using random seed {args.seed} ")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)

    # ------------------------------------------------------------------
    # 1. Data loading
    # ------------------------------------------------------------------
    print(f"\n=== Dataset: {args.dataset} ===")
    dataset, loader = get_dataset_and_loader(args)

    train_vecs, train_confs = extract_features(loader["train"])
    print(f"\nTotal training sequences: {len(train_vecs)}")

    target_data: dict = {}   # split_key -> {vecs, confs, meta}
    evaluators:  dict = {}   # split_key -> PoseAnomalyEvaluator

    seen = set()
    for split in ["test", "validate"]:
        if split not in loader or loader[split] is None:
            continue
        lid = id(loader[split])
        if lid in seen:
            continue
        seen.add(lid)
        v, c = extract_features(loader[split])
        meta = dataset[split].metadata if split in dataset else None
        target_data[split] = {"vecs": v, "confs": c, "meta": meta}
        eval_split = "test" if split == "test" else "validation"
        evaluators[split] = PoseAnomalyEvaluator(args, split=eval_split)


    # ------------------------------------------------------------------
    # 2. Projection (PCA)
    # ------------------------------------------------------------------
    pca_mat = tr_mean = tr_std = None

    pca_mat, tr_proj = fit_pca(train_vecs, args.pca_dim)
    te_projs = {k: apply_pca(pca_mat, v["vecs"]) for k, v in target_data.items()}

    # ------------------------------------------------------------------
    # 3. Whitening
    # ------------------------------------------------------------------
    if args.standardize:
        print("Whitening features ...")
        tr_mean, tr_std = fit_standardize(tr_proj)
        tr_proj = apply_standardize(tr_proj, tr_mean, tr_std)
        for k in te_projs:
            te_projs[k] = apply_standardize(te_projs[k], tr_mean, tr_std)

    feature_dim = tr_proj.shape[1]

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    model = STEPNetwork(
        feature_dim    = feature_dim,
        units          = args.units,
        layernorm      = args.layernorm,
        network_jitter = args.network_jitter,
        L              = args.L,
        sigma_low      = args.sigma_low,
        sigma_high     = args.sigma_high,
    ).to(device)

    # ------------------------------------------------------------------
    # 5. Logging
    # ------------------------------------------------------------------
    ts       = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"STEP_{args.dataset}_{ts}_lr{args.lr}_seg{args.seg_len}_pca{args.pca_dim}"
    log_dir  = os.path.join(args.runs_dir, run_name)
    writer   = SummaryWriter(log_dir=log_dir)
    writer.add_text("Args", json.dumps(vars(args), indent=4, cls=_NumpyEncoder))

    best_auc = [0.0]
    # ShanghaiTech has no validation split (test aliased to validate)
    ckpt_split = "validate" if "validate" in evaluators else "test"

    # ------------------------------------------------------------------
    # 6. Validation callback
    # ------------------------------------------------------------------
    sigma_person = args.temporal_gaussian_1d_smoothing_sigma_person
    sigma_frame  = args.temporal_gaussian_1d_smoothing_sigma_frame_level

    def run_validation(inference_model, epoch):
        stats = compute_energy_stats(inference_model, tr_proj, train_confs, str(device))

        for s_val, st in stats.items():
            writer.add_scalar(f"Energy/sigma_{s_val:.4f}/mean", st["mean"], epoch)
            writer.add_scalar(f"Energy/sigma_{s_val:.4f}/std",  st["std"],  epoch)

        te_confs = {k: (v["confs"] if args.score_confidence_weighted else np.ones(len(v["confs"])))
                    for k, v in target_data.items()}

        for split_key, evaluator in evaluators.items():
            split_tb = "Validation" if split_key == "validate" else "Test"
            scores = compute_score_matrix(
                inference_model, te_projs[split_key], te_confs[split_key],
                stats, str(device), confidence_weighted=args.score_confidence_weighted,
            )

            person_sigmas = tuple(dict.fromkeys((0, sigma_person)))
            flat = evaluator.flatten_dataset(
                scores["Agg_Max"], target_data[split_key]["meta"],
                person_sigmas=person_sigmas,
            )
            if len(flat.get("gt", [])) == 0:
                continue

            gt       = np.array(flat["gt"])
            clip_ids = np.array(flat["clip_ids"])

            smooth_rows, raw_rows = {}, {}
            for pipe in _PROTOCOLS:
                raw = flat[f"{pipe}_P0"]
                finite = raw[raw != -np.inf]
                filled = raw.copy()
                filled[filled == -np.inf] = finite.min() - 1.0 if len(finite) else 0.0

                smooth_source = flat[f"{pipe}_P{sigma_person}"]
                finite = smooth_source[smooth_source != -np.inf]
                smooth_filled = smooth_source.copy()
                smooth_filled[smooth_filled == -np.inf] = (
                    finite.min() - 1.0 if len(finite) else 0.0
                )
                if sigma_frame > 0:
                    smoothed = smooth_filled.copy()
                    for clip_id in np.unique(clip_ids):
                        mask = clip_ids == clip_id
                        smoothed[mask] = gaussian_filter1d(
                            smooth_filled[mask], sigma=sigma_frame,
                        )
                else:
                    smoothed = smooth_filled

                metrics_smooth = evaluator.calc_metrics(gt, smoothed)
                metrics_raw    = evaluator.calc_metrics(gt, filled)
                smooth_rows[pipe] = (metrics_smooth["AUC"], metrics_smooth["AP"])
                raw_rows[pipe]    = (metrics_raw["AUC"],    metrics_raw["AP"])

                # Headline metrics: underscore-prefixed tags sort to the top of
                # TensorBoard, the "/protocol" suffix groups the 3 protocols under
                # one tag. Smooth = with frame-level Gaussian, Raw = without.
                writer.add_scalar(f"_AUC_{split_tb}_Smooth/{pipe}", metrics_smooth["AUC"], epoch)
                writer.add_scalar(f"_AP_{split_tb}_Smooth/{pipe}",  metrics_smooth["AP"],  epoch)
                writer.add_scalar(f"_AUC_{split_tb}_Raw/{pipe}",    metrics_raw["AUC"],    epoch)
                writer.add_scalar(f"_AP_{split_tb}_Raw/{pipe}",     metrics_raw["AP"],     epoch)

                if pipe == "Online_Latest" and split_key == ckpt_split and metrics_smooth["AUC"] > best_auc[0]:
                    best_auc[0] = metrics_smooth["AUC"]
                    print(f"[Epoch {epoch}] [{split_tb}] New best: "
                          f"AUC={metrics_smooth['AUC']:.4f}  AP={metrics_smooth['AP']:.4f}")
                    if args.save_model:
                        save_checkpoint(
                            os.path.join(log_dir, "best_model"),
                            inference_model, pca_mat, tr_mean, tr_std, stats, args,
                        )

            # Console summary per eval: AUC/AP for all protocols, smooth & raw.
            def _fmt(rows):
                return "  ".join(
                    f"{p:15s} {rows[p][0]:.4f}/{rows[p][1]:.4f}" for p in _PROTOCOLS
                )
            print(f"[Epoch {epoch}] [{split_tb}] Smooth AUC/AP: " + _fmt(smooth_rows))
            print(f"[Epoch {epoch}] [{split_tb}] Raw    AUC/AP: " + _fmt(raw_rows))

        return stats

    def periodic_save(inference_model, epoch):
        stats = compute_energy_stats(inference_model, tr_proj, train_confs, str(device))
        save_checkpoint(
            os.path.join(log_dir, f"checkpoint_epoch_{epoch:04d}"),
            inference_model, pca_mat, tr_mean, tr_std, stats, args,
        )
        print(f"[Epoch {epoch}] Periodic checkpoint saved.")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    trainer = STEPTrainer.build(model, args, writer=writer)
    model   = trainer.train(
        tr_proj, train_confs,
        val_callback=run_validation,
        periodic_save_fn=periodic_save if args.save_every_n_epochs > 0 else None,
    )

    # ------------------------------------------------------------------
    # 8. Final evaluation + end-of-training report
    # ------------------------------------------------------------------
    print("\n--- Final evaluation ---")
    stats = compute_energy_stats(model, tr_proj, train_confs, str(device))
    run_validation(model, args.epochs)

    # Save the final model for evaluation
    final_dir = os.path.join(log_dir, "final_model")
    save_checkpoint(final_dir, model, pca_mat, tr_mean, tr_std, stats, args)

    # ------------------------------------------------------------------
    # End-of-training report: all 3 protocols at the dataset-default sigma
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  END-OF-TRAINING REPORT")

    te_confs_final = {
        k: (v["confs"] if args.score_confidence_weighted else np.ones(len(v["confs"])))
        for k, v in target_data.items()
    }

    for split_key, evaluator in evaluators.items():
        split_tb = "Validation" if split_key == "validate" else "Test"
        scores_final = compute_score_matrix(
            model, te_projs[split_key], te_confs_final[split_key],
            stats, str(device), confidence_weighted=args.score_confidence_weighted,
        )
        flat = evaluator.flatten_dataset(
            scores_final["Agg_Max"], target_data[split_key]["meta"],
            person_sigmas=(sigma_person,),
        )
        if len(flat.get("gt", [])) == 0:
            continue

        gt       = np.array(flat["gt"])
        clip_ids = np.array(flat["clip_ids"])
        print(f"\n  {args.dataset} | {split_key}  (p-sigma={sigma_person}, f-sigma={sigma_frame})")
        print(f"  {'Protocol':<22} {'AUC':>8} {'AP':>8}")
        print(f"  {'-'*40}")

        for pipe in _PROTOCOLS:
            raw    = flat[f"{pipe}_P{sigma_person}"]
            fin    = raw[raw != -np.inf]
            floor  = (fin.min() - 1.0) if len(fin) else 0.0
            filled = raw.copy()
            filled[filled == -np.inf] = floor

            if sigma_frame > 0:
                smoothed = filled.copy()
                for cid in np.unique(clip_ids):
                    m = clip_ids == cid
                    smoothed[m] = gaussian_filter1d(filled[m], sigma=sigma_frame)
            else:
                smoothed = filled
            m = evaluator.calc_metrics(gt, smoothed)
            print(f"  {pipe:<22} {m['AUC']*100:>7.2f}% {m['AP']*100:>7.2f}%")
            writer.add_scalar(f"{split_tb}_AUC_final/{pipe}", m["AUC"], args.epochs)
            writer.add_scalar(f"{split_tb}_AP_final/{pipe}",  m["AP"],  args.epochs)

    writer.close()
    print(f"\nDone. Results in: {log_dir}")


if __name__ == "__main__":
    main()
