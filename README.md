# STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection

[//]: # ([![ECCV]&#40;https://img.shields.io/badge/ECCV-2026-blue.svg&#41;]&#40;&#41;)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://jakubmicorek.github.io/STEP-demo)
[![arXiv](https://img.shields.io/badge/arXiv-2608.19987-b31b1b.svg)](https://arxiv.org/abs/2608.19987)

This is the **official implementation** of our ECCV 2026 paper *"STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection"*.

> **Abstract.** Skeleton-based Video Anomaly Detection (VAD) offers a robust, privacy-preserving solution for identifying abnormal behaviors. To model the distribution of normal static and moving poses, recent methods train Energy-Based Models (EBMs) via Denoising Score Matching (DSM). However, directly injecting noise, required for training, into raw joint coordinates creates physically impossible poses, and this structural collapse severely worsens as the temporal window expands. To address this, we introduce STEP, a simple framework that utilizes Principal Component Analysis (PCA) to project pose sequences into a compact, whitened PC-space. Learning the data density within this well-behaved PC-space ensures that the injected noise translates into physically plausible variations, which allows the model to process longer video sequences without the performance collapse of raw coordinate baselines. Additionally, to mitigate inherent pose estimation inaccuracies arising from occlusions or motion blur, we integrate a sequence-level weighting mechanism based on the estimator's confidence scores. Operating at real-time computational efficiency, our simple and lightweight framework outperforms the previous skeleton-based state-of-the-art by 12.2% (90.1% AUROC) on the challenging UBnormal dataset and achieves highly competitive results by improving on the ShanghaiTech benchmark.


---

## Install & Train

```bash
# Install (conda env "step"):  or use pip install -r requirements.txt
bash setup.sh && conda activate step

# 1. Obtain the pose data (see data/README.md)
# 2. Train (paper defaults are already in args.py; checkpoints save automatically)
python train.py --dataset UBnormal
python train.py --dataset ShanghaiTech --lr 2e-4

# 3. Monitor training in TensorBoard (open http://localhost:6006)
tensorboard --logdir runs
```

Checkpoints (model, PCA matrix, feature/energy statistics) are saved to
`runs/STEP_<dataset>_<timestamp>_lr<lr>_seg<seg_len>_pca<pca_dim>/`.

---

## Data

Pose sequences (UBnormal, ShanghaiTech) are obtained from the
[STG-NF](https://github.com/orhir/STG-NF) repository; the UBnormal HR boolean masks
from [MoCoDAD](https://github.com/aleflabo/MoCoDAD). `data/UBnormal_labels/` ships
with this repository. Full layout and download instructions:
[`data/README.md`](data/README.md).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{micorek2026step,
  title     = {{STEP: Score-Based Temporal Energy for Human Pose Video Anomaly Detection}},
  author    = {Micorek, Jakub and Kozi{\'n}ski, Mateusz and Possegger, Horst},
  booktitle = {Proc. of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Acknowledgements

We thank the authors of
[STG-NF](https://github.com/orhir/STG-NF), [SeeKer](https://github.com/adelic99/seeker), and [MULDE](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection). Parts of this codebase are used and adapted from these repositories.

---

## Disclaimer

The code in this repository has been refactored from our research codebase and may contain errors.
