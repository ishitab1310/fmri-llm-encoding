#!/bin/bash
# =============================================================================
# STEP 0: Environment Setup
# Run this ONCE on the Ada LOGIN NODE (not a compute node)
# Usage: bash setup_env.sh
# =============================================================================

set -e   # stop on any error

echo "=== A3 Project Environment Setup on Ada ==="
echo "This installs miniconda + all packages in your home directory."
echo ""

# ── 1. Install Miniconda if not already present ──────────────────────────────
if [ ! -d "$HOME/miniconda3" ]; then
    echo "[1/5] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    echo "export PATH=\"$HOME/miniconda3/bin:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
    echo "   Miniconda installed."
else
    echo "[1/5] Miniconda already installed — skipping."
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# ── 2. Create conda environment ───────────────────────────────────────────────
ENV_NAME="a3_llada"
if conda env list | grep -q "$ENV_NAME"; then
    echo "[2/5] Conda env '$ENV_NAME' already exists — skipping."
else
    echo "[2/5] Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
    echo "   Environment created."
fi

# Activate
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# ── 3. Install PyTorch with CUDA 11.6 (matches Ada's available module) ────────
echo "[3/5] Installing PyTorch + CUDA..."
# Ada has cuda/11.6 module — use the matching torch wheel
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116 -q
echo "   PyTorch installed."

# ── 4. Install all project packages ──────────────────────────────────────────
echo "[4/5] Installing project packages..."
pip install -q \
    transformers==4.40.2 \
    accelerate==0.29.3 \
    bitsandbytes==0.43.0 \
    sentencepiece \
    nibabel \
    nilearn \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scipy \
    tqdm \
    awscli \
    huggingface_hub
echo "   Packages installed."

# ── 5. Create project directory structure ─────────────────────────────────────
echo "[5/5] Creating project directories..."
mkdir -p $HOME/a3_project/scripts
mkdir -p $HOME/a3_project/logs
mkdir -p /share1/$USER/ds003643
mkdir -p /share1/$USER/features
mkdir -p /share1/$USER/results
mkdir -p /share1/$USER/llada_model
mkdir -p /scratch/$USER   # may not exist yet, sbatch creates it

echo ""
echo "=== Setup complete ==="
echo "To activate the environment in future sessions:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate a3_llada"
echo ""
echo "Next step: bash download_data.sh"
