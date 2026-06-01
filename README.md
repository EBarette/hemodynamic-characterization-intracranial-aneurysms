# Hemodynamic Characterization of Intracranial Aneurysms

Code and simulation setup for a master's thesis comparing patient-specific CFD (SimVascular/svMultiPhysics) and Physics-Informed Neural Networks (PhysicsNeMo) for hemodynamic characterization of intracranial aneurysms.

## Repository Structure

```
src/
  PhysicsNeMo/        # PINN training scripts and Hydra configs
    aneurysm.py             # Steady-state PINN (reference)
    aneurysm_pulsatile.py   # Pulsatile PINN
    conf/                   # Hydra configuration files
    stl_files/              # STL surface geometry inputs
  SimVascular/        # SimVascular project files (6 patient cases)
    0203_H_CERE_CA/   # Patient case (Paths, Segmentations, Models, Meshes, Simulations)
    0204_H_CERE_CA/
    ...
    inputs/           # Solver XML / boundary condition files
jobs/                 # SLURM batch scripts
install/              # HPC environment setup scripts
```

## Cases

Six intracranial aneurysm geometries from the [Vascular Model Repository](https://www.vascularmodel.com):
`0203`, `0204`, `0207`, `0208`, `0209`, `0210`.

Each case contains:
- SimVascular project (paths, segmentations, surface model, mesh)
- svMultiPhysics solver configuration (pulsatile Navier-Stokes, VMS formulation)
- PINN training configuration (steady + pulsatile, full and hybrid data-informed variants)

## Requirements

### PhysicsNeMo (PINNs)
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (formerly Modulus)
- See `install/install_physicsnemo.sbatch` for the HPC setup

### SimVascular (CFD)
- [SimVascular](https://simvascular.github.io/) with svMultiPhysics solver
- See `install/build_svmultiphysics_from_src.sbatch` for HPC build instructions

## Usage

SLURM job scripts in `jobs/` cover the full pipeline:
- `sv_<case>_pulsatile.sbatch` — run CFD simulation
- `pinn_<case>_pulsatile.sbatch` / `pinn_<case>_hybrid.sbatch` — run PINN training
- `hemo_fem.sbatch` / `hemo_pinn.sbatch` — compute hemodynamic indices

## License

See individual SimVascular case folders for data licenses (VMR data).
