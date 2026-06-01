#!/bin/bash
#SBATCH --job-name=pinn_0203_hybrid_sp0100
#SBATCH --output=/home/ucl/elen/ebarette/outputs/%x_%j.out
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# Sparsity-sweep variant of pinn_0203_hybrid: samples_per_step=0100.
set -euo pipefail
module --force purge
module load releases/2024a
module load tis/2025
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
source $GLOBALSCRATCH/physicsnemo-py312/bin/activate
python -c "import physicsnemo, physicsnemo.sym; print('PhysicsNeMo OK')"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=${SLURM_NTASKS:-1}
export RANK=${SLURM_PROCID:-0}
export LOCAL_RANK=${SLURM_LOCALID:-0}
cd $HOME/src/PhysicsNeMo
srun --ntasks=$WORLD_SIZE --gpus=1 \
    python aneurysm_pulsatile.py \
    --config-name config_0203_hybrid_sp0100
echo "Done: pinn_0203_hybrid_sp0100"
