#!/usr/bin/env bash
# Creates the "step" conda environment with all dependencies.
# Run from anywhere:  bash STEP/setup.sh

set -e

conda create -n step python=3.10 -y
conda run -n step pip install torch --index-url https://download.pytorch.org/whl/cu121
conda run -n step pip install \
    numpy==1.26.4 \
    scipy \
    scikit-learn \
    faiss-gpu \
    tqdm \
    tensorboard

echo ""
echo "Environment ready.  Activate with:  conda activate step"
