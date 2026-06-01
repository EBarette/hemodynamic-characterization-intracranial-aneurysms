#!/bin/bash
#SBATCH --job-name=build_svmp_source
#SBATCH --output=outputs/%x_%j.out
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

load_module_or_fail() {
  local mod="$1"
  if ! module load "$mod" >/dev/null 2>&1; then
    echo "Failed to load module: $mod" >&2
    exit 2
  fi
}

module --force purge
load_module_or_fail releases/2024a
load_module_or_fail tis/2025
load_module_or_fail GCC/13.3.0
load_module_or_fail CMake/3.31.8-GCCcore-13.3.0
load_module_or_fail OpenMPI/5.0.3-GCC-13.3.0
load_module_or_fail OpenBLAS/0.3.27-GCC-13.3.0

if ! command -v git >/dev/null 2>&1; then
  if module load git/2.45.1-GCCcore-13.3.0 >/dev/null 2>&1; then
    :
  else
    echo "git is not available in PATH and git module could not be loaded" >&2
    exit 2
  fi
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$WORKDIR/installers}"
APP_ROOT="${INSTALL_ROOT:-$GLOBALSCRATCH/apps/simvascular}"

VTK_VERSION="${VTK_VERSION:-9.3.1}"
VTK_URL="${VTK_URL:-https://www.vtk.org/files/release/9.3/VTK-${VTK_VERSION}.tar.gz}"
VTK_TARBALL="${VTK_TARBALL:-$ARCHIVE_DIR/VTK-${VTK_VERSION}.tar.gz}"
VTK_PREFIX="$APP_ROOT/vtk-${VTK_VERSION}"

SVMP_REPO_URL="${SVMP_REPO_URL:-https://github.com/SimVascular/svMultiPhysics.git}"
SVMP_SRC="$APP_ROOT/src/svMultiPhysics"
SVMP_BUILD="$APP_ROOT/build/svMultiPhysics-build"
SVMP_PREFIX="$APP_ROOT/svMultiPhysics-source"

VTK_SRC_STAGE="$APP_ROOT/build/vtk-src"
VTK_BUILD="$APP_ROOT/build/vtk-build"

NPROC="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$ARCHIVE_DIR" "$APP_ROOT/build" "$APP_ROOT/src"

download_file() {
  local url="$1"
  local out="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$out" "$url"
  else
    echo "Neither wget nor curl is available for downloading $url" >&2
    return 1
  fi
}

echo "=== Build settings ==="
echo "WORKDIR: $WORKDIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"
echo "APP_ROOT: $APP_ROOT"
echo "VTK_TARBALL: $VTK_TARBALL"
echo "VTK_PREFIX: $VTK_PREFIX"
echo "SVMP_SRC: $SVMP_SRC"
echo "SVMP_BUILD: $SVMP_BUILD"
echo "SVMP_PREFIX: $SVMP_PREFIX"

if [ ! -f "$VTK_TARBALL" ]; then
  echo "VTK tarball not found locally, downloading from: $VTK_URL"
  download_file "$VTK_URL" "$VTK_TARBALL"
fi

echo "=== Building VTK ${VTK_VERSION} ==="
rm -rf "$VTK_SRC_STAGE" "$VTK_BUILD"
mkdir -p "$VTK_SRC_STAGE"
tar -xzf "$VTK_TARBALL" -C "$VTK_SRC_STAGE"

VTK_SRC_DIR="$VTK_SRC_STAGE/VTK-${VTK_VERSION}"
if [ ! -d "$VTK_SRC_DIR" ]; then
  echo "Expected VTK source directory not found: $VTK_SRC_DIR" >&2
  exit 3
fi

cmake -S "$VTK_SRC_DIR" -B "$VTK_BUILD" \
  -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DVTK_USE_SYSTEM_EXPAT:BOOL=ON \
  -DVTK_USE_SYSTEM_ZLIB:BOOL=ON \
  -DVTK_LEGACY_REMOVE=ON \
  -DVTK_Group_Rendering=OFF \
  -DVTK_Group_StandAlone=OFF \
  -DVTK_RENDERING_BACKEND=None \
  -DVTK_WRAP_PYTHON=OFF \
  -DModule_vtkChartsCore=ON \
  -DModule_vtkCommonCore=ON \
  -DModule_vtkCommonDataModel=ON \
  -DModule_vtkCommonExecutionModel=ON \
  -DModule_vtkFiltersCore=ON \
  -DModule_vtkFiltersFlowPaths=ON \
  -DModule_vtkFiltersModeling=ON \
  -DModule_vtkIOLegacy=ON \
  -DModule_vtkIOXML=ON \
  -DVTK_GROUP_ENABLE_Views=NO \
  -DVTK_GROUP_ENABLE_Web=NO \
  -DVTK_GROUP_ENABLE_Imaging=NO \
  -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
  -DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
  -DCMAKE_INSTALL_PREFIX="$VTK_PREFIX"

cmake --build "$VTK_BUILD" -j "$NPROC"
cmake --install "$VTK_BUILD"

echo "=== Preparing svMultiPhysics source ==="
if [ -d "$SVMP_SRC/.git" ]; then
  git -C "$SVMP_SRC" fetch --all --tags
  git -C "$SVMP_SRC" pull --ff-only || true
else
  rm -rf "$SVMP_SRC"
  git clone --depth 1 "$SVMP_REPO_URL" "$SVMP_SRC"
fi

echo "=== Building svMultiPhysics ==="
rm -rf "$SVMP_BUILD" "$SVMP_PREFIX"
mkdir -p "$SVMP_BUILD" "$SVMP_PREFIX/bin"

cmake -S "$SVMP_SRC" -B "$SVMP_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SVMP_PREFIX" \
  -DSV_LOCAL_VTK_PATH="$VTK_PREFIX"

cmake --build "$SVMP_BUILD" -j "$NPROC"
cmake --install "$SVMP_BUILD" || true

SVMP_BIN="$(find "$SVMP_BUILD" -type f -name svmultiphysics | head -n 1 || true)"
if [ -z "$SVMP_BIN" ]; then
  echo "Could not find svmultiphysics binary in build tree: $SVMP_BUILD" >&2
  exit 4
fi

cp -f "$SVMP_BIN" "$SVMP_PREFIX/bin/svmultiphysics"
chmod +x "$SVMP_PREFIX/bin/svmultiphysics"

ENV_FILE="$WORKDIR/install/env_svmultiphysics_source.sh"
cat > "$ENV_FILE" <<EOF
#!/bin/bash
set -euo pipefail

module --force purge
module load releases/2024a
module load tis/2025
module load GCC/13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
module load OpenBLAS/0.3.27-GCC-13.3.0

prepend_unique_path() {
  local p="\$1"
  case ":\$PATH:" in
    *":\$p:"*) ;;
    *) export PATH="\$p:\$PATH" ;;
  esac
}

prepend_unique_ld() {
  local p="\$1"
  local cur="\${LD_LIBRARY_PATH:-}"
  case ":\$cur:" in
    *":\$p:"*) ;;
    *) export LD_LIBRARY_PATH="\$p\${cur:+:\$cur}" ;;
  esac
}

export SVMULTIPHYSICS_ROOT="$SVMP_PREFIX"
export VTK_ROOT="$VTK_PREFIX"

prepend_unique_path "\$SVMULTIPHYSICS_ROOT/bin"

if [ -d "\$VTK_ROOT/lib" ]; then
  prepend_unique_ld "\$VTK_ROOT/lib"
fi
if [ -d "\$VTK_ROOT/lib64" ]; then
  prepend_unique_ld "\$VTK_ROOT/lib64"
fi
EOF

chmod +x "$ENV_FILE"

source "$ENV_FILE"
echo "svmultiphysics path: $(command -v svmultiphysics)"

if ldd "$(command -v svmultiphysics)" | grep -q "not found"; then
  echo "Runtime dependency check failed: unresolved libraries detected" >&2
  ldd "$(command -v svmultiphysics)" | grep "not found" >&2 || true
  exit 5
fi

echo "Build completed successfully"
echo "Binary: $SVMP_PREFIX/bin/svmultiphysics"
echo "Env script: $ENV_FILE"
