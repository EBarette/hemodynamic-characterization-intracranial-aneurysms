#!/bin/bash
#SBATCH --job-name=sv_0209_pulsatile
#SBATCH --output=/home/ucl/elen/ebarette/outputs/%x_%j.out
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem=16G

# svMultiPhysics pulsatile run for patient 0209 (4 cardiac cycles = 3.8 s).
# PREREQUISITE: mesh_0209.sbatch must complete successfully first.
# XML:  src/SimVascular/inputs/pulsatile/0209_pulsatile_solver.xml
# PINN counterpart: jobs/pinn_0209_pulsatile.sbatch
# Submit from workspace root: sbatch jobs/sv_0209_pulsatile.sbatch

set -euo pipefail

# Verify mesh exists before launching
MESH="$HOME/src/SimVascular/0209_H_CERE_CA/Simulations/ANR_008V1cgs/mesh-complete/mesh-complete.mesh.vtu"
if [ ! -f "$MESH" ]; then
    echo "ERROR: FEM mesh not found: $MESH"
    echo "  Run jobs/mesh_0209.sbatch first."
    exit 1
fi

module --force purge
module load releases/2024a
module load tis/2025
module load GCC/13.3.0
module load OpenBLAS/0.3.27-GCC-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load Python/3.12.3-GCCcore-13.3.0

SV_BIN="$GLOBALSCRATCH/apps/simvascular/svMultiPhysics-source/bin/svmultiphysics"
XML="$HOME/src/SimVascular/inputs/pulsatile/0209_pulsatile_solver.xml"
OUTDIR="$HOME/src/SimVascular/outputs/0209_H_CERE_CA_pulsatile_refined"
FLOW_SRC="$HOME/src/SimVascular/0207_H_CERE_CA/Simulations/Anr_006V1cgsPulsePres/ICA.flow"
FLOW_EXT="$HOME/src/SimVascular/0207_H_CERE_CA/Simulations/Anr_006V1cgsPulsePres/ICA_extended.flow"

export LD_LIBRARY_PATH="$GLOBALSCRATCH/apps/simvascular/vtk-9.3.1/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUTDIR"

# Generate ICA_extended.flow if not already present
"$GLOBALSCRATCH/physicsnemo-py312/bin/python" - "$FLOW_SRC" "$FLOW_EXT" << 'PYEOF'
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
data = np.loadtxt(src)
t0, Q0 = data[:, 0], data[:, 1]
T = float(t0[-1])
N = 1
t_ext, Q_ext = [], []
for n in range(N):
    t_ext.extend(t0[:-1] + n * T)
    Q_ext.extend(Q0[:-1])
t_ext.append(N * T)
Q_ext.append(float(Q0[0]))
with open(dst, "w") as _f:
    _f.write(f"{len(t_ext)}   10\n")
    np.savetxt(_f, np.column_stack([t_ext, Q_ext]), fmt="%.15e", delimiter="   ")
print(f"Written {len(t_ext)} time points to {dst}  (T_total={N*T:.4f} s)")
PYEOF

cd "$OUTDIR"
mpirun -np 4 "$SV_BIN" "$XML"

echo "Done: sv_0209_pulsatile"
