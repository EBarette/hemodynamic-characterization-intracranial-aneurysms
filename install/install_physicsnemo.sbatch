#!/bin/bash
#SBATCH --job-name=install_physicsnemo
#SBATCH --output=%x_%j.out
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=1

set -euo pipefail

module --force purge
module load releases/2024a # for python 312
module load tis/2025 # for cuda 126


module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# Optional GPU visibility check (won't fail the job if unavailable)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

VENV="$HOME/venvs/physicsnemo-py312"
REPO="$HOME/src/physicsnemo-sym"

module list
nvcc --version
python --version
which python

# create venv
if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch

# Needed to build some dependencies from source (e.g., numpoly)
python -m pip install "Cython>=0.29"

# Work around packages that don't declare Cython properly in pyproject build requirements
# python -m pip install --no-build-isolation "numpoly==1.3.4"

# clone repo
if [ ! -d "$REPO/.git" ]; then
  git clone --depth 1 https://github.com/NVIDIA/physicsnemo-sym.git "$REPO"
else
  cd "$REPO"
  git pull --ff-only
fi

cd "$REPO"


python -m pip install warp-lang
python -m pip install "nvidia-physicsnemo>=1.0.0"
python -m pip install -e . --no-build-isolation

# sanity check
python -c "import physicsnemo, physicsnemo.sym; print('PN file', physicsnemo.__file__); print('PNS file', physicsnemo.sym.__file__)"

# reproducibility
python -m pip freeze > "$HOME/pnemo_requirements_312.txt"