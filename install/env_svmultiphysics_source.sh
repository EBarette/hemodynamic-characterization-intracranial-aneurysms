#!/bin/bash
set -euo pipefail

module --force purge
module load releases/2024a
module load tis/2025
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS/0.3.27-GCC-13.3.0

prepend_unique_path() {
  local p="$1"
  case ":$PATH:" in
    *":$p:"*) ;;
    *) export PATH="$p:$PATH" ;;
  esac
}

prepend_unique_ld() {
  local p="$1"
  local cur="${LD_LIBRARY_PATH:-}"
  case ":$cur:" in
    *":$p:"*) ;;
    *) export LD_LIBRARY_PATH="$p${cur:+:$cur}" ;;
  esac
}

export SVMULTIPHYSICS_ROOT="/globalsc/ucl/elen/ebarette/apps/simvascular/svMultiPhysics-source"
export VTK_ROOT="/globalsc/ucl/elen/ebarette/apps/simvascular/vtk-9.3.1"

prepend_unique_path "$SVMULTIPHYSICS_ROOT/bin"

if [ -d "$VTK_ROOT/lib" ]; then
  prepend_unique_ld "$VTK_ROOT/lib"
fi
if [ -d "$VTK_ROOT/lib64" ]; then
  prepend_unique_ld "$VTK_ROOT/lib64"
fi
