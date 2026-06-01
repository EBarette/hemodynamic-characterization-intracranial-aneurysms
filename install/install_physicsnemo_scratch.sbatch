#!/bin/bash
#SBATCH --job-name=install_physicsnemo_scratch
#SBATCH --output=%x_%j.out
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1

set -euo pipefail

module --force purge
module load releases/2024a # for python 312
module load tis/2025 # for cuda 126

module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV="$GLOBALSCRATCH/physicsnemo-py312"

module list
nvcc --version
python --version
which python

# create venv in scratch if needed
if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"
export MAX_JOBS=1

python -m pip install --upgrade pip "setuptools<82" wheel ninja
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch

# Needed to build some dependencies from source (e.g., numpoly)
python -m pip install "Cython>=0.29"

python -m pip install nvidia-physicsnemo.sym --no-build-isolation

# sanity check
python -c "import physicsnemo, physicsnemo.sym; print('PN file', physicsnemo.__file__); print('PNS file', physicsnemo.sym.__file__)"
