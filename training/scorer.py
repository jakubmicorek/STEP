"""
training/scorer.py
==================
Energy statistics and score matrix computation.

All functions are pure (no side effects, no model mutation).
"""

import numpy as np
import torch


def compute_energy_stats(
    model,
    features: np.ndarray,
    confs: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> dict:
    """
    Compute per-sigma mean and std of the energy on the training set.
    Used for z-score normalisation at inference.

    Returns
    -------
    dict  {sigma_value: {"mean": float, "std": float}}
    """
    model.eval()
    sigmas   = model.get_sigma_list()
    acc      = {s: [] for s in sigmas}

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[i: i + batch_size]).float().to(device)
            for s_val in sigmas:
                s_t    = torch.full((batch.shape[0], 1), s_val, device=device)
                energy = model.forward(batch, s_t).cpu().numpy().flatten()
                acc[s_val].append(energy)

    stats = {}
    for s_val in sigmas:
        if acc[s_val]:
            all_e = np.concatenate(acc[s_val])
            stats[s_val] = {"mean": float(np.mean(all_e)), "std": float(np.std(all_e)) + 1e-6}
        else:
            stats[s_val] = {"mean": 0.0, "std": 1.0}
    return stats


def compute_score_matrix(
    model,
    features: np.ndarray,
    confs: np.ndarray,
    stats: dict,
    device: str,
    confidence_weighted: bool = True,
    batch_size: int = 2048,
) -> dict:
    """
    Compute per-segment anomaly scores for all sigma levels.

    Scores are z-normalised: (E - mu_sigma) / std_sigma, optionally x confidence.

    Returns
    -------
    dict with keys:
        "Agg_Max"         : np.ndarray[N]   - max z-score across sigma
        "Agg_Sum"         : np.ndarray[N]   - sum of z-scores across sigma
        "Sigma_<value>"   : np.ndarray[N]   - individual sigma scores
    """
    model.eval()
    sigmas = model.get_sigma_list()
    all_norm_batches = []

    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch      = torch.from_numpy(features[i: i + batch_size]).float().to(device)
            batch_conf = confs[i: i + batch_size] if confidence_weighted else np.ones(batch.shape[0])
            batch_norm = np.zeros((batch.shape[0], len(sigmas)), dtype=np.float32)

            for idx, s_val in enumerate(sigmas):
                s_t    = torch.full((batch.shape[0], 1), s_val, device=device)
                energy = model.forward(batch, s_t).cpu().numpy().flatten()
                z      = (energy - stats[s_val]["mean"]) / stats[s_val]["std"]
                z      = z * batch_conf
                batch_norm[:, idx] = z

            all_norm_batches.append(batch_norm)

    matrix = np.concatenate(all_norm_batches, axis=0)

    results = {f"Sigma_{s:.4f}": matrix[:, i] for i, s in enumerate(sigmas)}
    results["Agg_Sum"] = matrix.sum(axis=1)
    results["Agg_Max"] = matrix.max(axis=1)
    return results
