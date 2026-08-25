"""training/projection.py - Faiss PCA + whitening (default)."""

from __future__ import annotations

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# PCA (Faiss)
# ---------------------------------------------------------------------------

def fit_pca(
    train_vecs: np.ndarray,
    target_dim: int,
) -> tuple[object, np.ndarray]:
    """Fit Faiss PCA on train_vecs. Returns (pca_mat, tr_proj)."""
    d_in = train_vecs.shape[1]
    if target_dim == 0 or target_dim >= d_in:
        print(f"Skipping PCA (target_dim={target_dim}, input_dim={d_in}) - identity projection.")
        return None, train_vecs.copy()

    print(f"Fitting PCA {d_in} -> {target_dim} on {len(train_vecs)} vectors ...")
    mat = faiss.PCAMatrix(d_in, target_dim)
    mat.train(train_vecs)

    tr_proj = mat.apply_py(train_vecs)
    return mat, tr_proj


def apply_pca(pca_mat, vecs: np.ndarray) -> np.ndarray:
    if pca_mat is None:
        return vecs
    return pca_mat.apply_py(vecs)


def save_pca(pca_mat, path: str):
    if pca_mat is not None:
        faiss.write_VectorTransform(pca_mat, path)


def load_pca(path: str):
    """Load a saved Faiss PCA matrix."""
    return faiss.read_VectorTransform(path)


# ---------------------------------------------------------------------------
# Whitening (standardize to zero-mean, unit-variance per component)
# ---------------------------------------------------------------------------

def fit_standardize(tr_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(tr_proj, axis=0, keepdims=True)
    std  =  np.std(tr_proj, axis=0, keepdims=True) + 1e-6
    return mean, std


def apply_standardize(proj: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (proj - mean) / std
