"""
compute_hemodynamics_pinn.py - PINN hemodynamic indices and PINN-vs-FEM
comparison for the pulsatile aneurysm simulations.

For each patient, the script:
  * Loads the pulsatile Modified Fourier checkpoint (4 x 256)
  * Samples N_WALL_SAMPLES points on aneurysm_noslip.stl
  * Evaluates the PINN at N_TIME_SAMPLES + 1 time levels spanning the
    cardiac period and computes the WSS vector via torch.autograd.grad
  * Reduces the (N_t, N_pts, 3) WSS history to TAWSS, OSI, RRT, ECAP, LSA
  * Compares the PINN cycle-averaged velocity field against the FEM
    cycle-averaged field on a regular voxel grid restricted to the lumen
    (relative L2 error)
  * Renders TAWSS / OSI scatter plots against the FEM CSV

Units: PINN is trained in CGS with rho_pinn = 1 and nu = 0.03774 cm^2/s.
Wall shear stress is computed as tau = nu * (grad u + grad u^T) . n, with
tangent component retained. The result is in dyne/cm^2 in the PINN's
internal "rho = 1" system, then multiplied by 0.1 to obtain Pa.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv

from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.key import Key

from region_labels import label_points


DYNE_PER_CM2_TO_PA = 0.1
LSA_THRESHOLD_PA = 0.4
PERIOD = 0.95
N_TIME_SAMPLES = 100
N_WALL_SAMPLES = 5000
BATCH_SIZE = 2048
NU_CGS = 0.03774

# FEM cycle parameters (must match compute_hemodynamics_fem.py).
DT_FEM = 0.0005
SAVE_EVERY = 19
STEPS_PER_CYCLE = int(round(PERIOD / DT_FEM))
FILES_PER_CYCLE = STEPS_PER_CYCLE // SAVE_EVERY

PROJECT_ROOT = os.path.expanduser("~/src/PhysicsNeMo")
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "outputs", "aneurysm_pulsatile")
STL_ROOT = os.path.join(PROJECT_ROOT, "stl_files")
FEM_ROOT = os.path.expanduser("~/src/SimVascular/outputs")


@dataclass
class Patient:
    pid: str
    status: str
    pair: int


PATIENTS: List[Patient] = [
    Patient("0203", "growing", 1),
    Patient("0204", "stable",  1),
    Patient("0207", "growing", 2),
    Patient("0208", "stable",  2),
    Patient("0209", "growing", 3),
    Patient("0210", "stable",  3),
]


# Network loading

def load_pulsatile_network(checkpoint_dir: str,
                           device: torch.device) -> torch.nn.Module:
    pth = os.path.join(checkpoint_dir, "flow_network.0.pth")
    if not os.path.isfile(pth):
        raise FileNotFoundError(pth)
    state = torch.load(pth, map_location=device)
    arch = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        layer_size=256,
        nr_layers=4,
    ).to(device)
    arch.load_state_dict(state, strict=False)
    arch.eval()
    return arch


# Wall sampling and WSS via autograd

def sample_wall(stl_path: str, n_pts: int) -> Dict[str, np.ndarray]:
    mesh = Tessellation.from_stl(stl_path, airtight=False)
    s = mesh.sample_boundary(n_pts)
    out = {
        "x":  np.asarray(s["x"]).reshape(-1).astype(np.float32),
        "y":  np.asarray(s["y"]).reshape(-1).astype(np.float32),
        "z":  np.asarray(s["z"]).reshape(-1).astype(np.float32),
        "nx": np.asarray(s["normal_x"]).reshape(-1).astype(np.float32),
        "ny": np.asarray(s["normal_y"]).reshape(-1).astype(np.float32),
        "nz": np.asarray(s["normal_z"]).reshape(-1).astype(np.float32),
    }
    if "area" in s:
        out["area"] = np.asarray(s["area"]).reshape(-1).astype(np.float64)
    return out


def wss_vector(arch, x, y, z, nx, ny, nz, t_val: float, nu: float):
    t = torch.full_like(x, t_val)
    out = arch({"x": x, "y": y, "z": z, "t": t})
    u, v, w = out["u"], out["v"], out["w"]
    ones = torch.ones_like(u)

    def g(s, var):
        return torch.autograd.grad(s, var, grad_outputs=ones,
                                   create_graph=False, retain_graph=True)[0]

    ux, uy, uz = g(u, x), g(u, y), g(u, z)
    vx, vy, vz = g(v, x), g(v, y), g(v, z)
    wx, wy, wz = g(w, x), g(w, y), g(w, z)

    tu = nu * (2.0 * ux * nx + (uy + vx) * ny + (uz + wx) * nz)
    tv = nu * ((vx + uy) * nx + 2.0 * vy * ny + (vz + wy) * nz)
    tw = nu * ((wx + uz) * nx + (wy + vz) * ny + 2.0 * wz * nz)

    t_dot_n = tu * nx + tv * ny + tw * nz
    return tu - t_dot_n * nx, tv - t_dot_n * ny, tw - t_dot_n * nz


def time_integrated_indices(arch, wall, device, nu: float) -> Dict[str, np.ndarray]:
    n = wall["x"].shape[0]
    t_grid = np.linspace(0.0, PERIOD, N_TIME_SAMPLES + 1, dtype=np.float32)
    dt = PERIOD / N_TIME_SAMPLES

    sum_vec = np.zeros((n, 3), dtype=np.float64)
    sum_mag = np.zeros((n,), dtype=np.float64)

    for it, t_val in enumerate(t_grid):
        w_quad = 0.5 if (it == 0 or it == len(t_grid) - 1) else 1.0
        tu = np.zeros(n, dtype=np.float32)
        tv = np.zeros(n, dtype=np.float32)
        tw = np.zeros(n, dtype=np.float32)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            sl = slice(start, end)
            xb = torch.tensor(wall["x"][sl], dtype=torch.float32, device=device,
                              requires_grad=True).reshape(-1, 1)
            yb = torch.tensor(wall["y"][sl], dtype=torch.float32, device=device,
                              requires_grad=True).reshape(-1, 1)
            zb = torch.tensor(wall["z"][sl], dtype=torch.float32, device=device,
                              requires_grad=True).reshape(-1, 1)
            nxb = torch.tensor(wall["nx"][sl], dtype=torch.float32,
                               device=device).reshape(-1, 1)
            nyb = torch.tensor(wall["ny"][sl], dtype=torch.float32,
                               device=device).reshape(-1, 1)
            nzb = torch.tensor(wall["nz"][sl], dtype=torch.float32,
                               device=device).reshape(-1, 1)
            t1, t2, t3 = wss_vector(arch, xb, yb, zb, nxb, nyb, nzb,
                                    float(t_val), nu)
            tu[sl] = t1.detach().cpu().numpy().reshape(-1)
            tv[sl] = t2.detach().cpu().numpy().reshape(-1)
            tw[sl] = t3.detach().cpu().numpy().reshape(-1)
        mag = np.sqrt(tu * tu + tv * tv + tw * tw)
        sum_vec[:, 0] += w_quad * tu
        sum_vec[:, 1] += w_quad * tv
        sum_vec[:, 2] += w_quad * tw
        sum_mag += w_quad * mag

    mean_vec = (dt / PERIOD) * sum_vec
    tawss_cgs = (dt / PERIOD) * sum_mag                       # dyne/cm^2 (rho=1)
    mean_mag = np.linalg.norm(mean_vec, axis=1)
    osi = 0.5 * (1.0 - mean_mag / np.maximum(tawss_cgs, 1e-30))
    osi = np.clip(osi, 0.0, 0.5)
    tawss_pa = tawss_cgs * DYNE_PER_CM2_TO_PA
    tawss_floor = np.maximum(tawss_pa, 0.05)
    rrt = 1.0 / (np.maximum(1.0 - 2.0 * osi, 1e-3) * tawss_floor)
    ecap = osi / tawss_floor
    return {"tawss": tawss_pa, "osi": osi, "rrt": rrt, "ecap": ecap}


# FEM cycle-averaged velocity for L2 comparison

def list_fem_analysis_vtus(pid: str) -> List[str]:
    """Same selection rule as compute_hemodynamics_fem.py: last
    FILES_PER_CYCLE result_*.vtu files."""
    refined = os.path.join(FEM_ROOT, f"{pid}_H_CERE_CA_pulsatile_refined",
                           "4-procs")
    if os.path.isdir(refined):
        folder = refined
    else:
        folder = os.path.join(FEM_ROOT, f"{pid}_H_CERE_CA_pulsatile",
                              "4-procs")
    candidates = sorted(glob(os.path.join(folder, "result_*.vtu")))
    parsed = []
    for f in candidates:
        try:
            n = int(os.path.basename(f).replace("result_", "").replace(".vtu", ""))
        except ValueError:
            continue
        parsed.append((n, f))
    parsed.sort()
    if not parsed:
        raise FileNotFoundError(f"No FEM VTUs for {pid} in {folder}")
    return [p for _, p in parsed[-FILES_PER_CYCLE:]]


def cycle_mean_velocity_on_grid(pid: str, n_grid: int = 48
                                ) -> Tuple[pv.ImageData, np.ndarray, np.ndarray]:
    files = list_fem_analysis_vtus(pid)
    first = pv.read(files[0])
    b = first.bounds
    grid = pv.ImageData(
        dimensions=(n_grid, n_grid, n_grid),
        spacing=(
            (b[1] - b[0]) / (n_grid - 1),
            (b[3] - b[2]) / (n_grid - 1),
            (b[5] - b[4]) / (n_grid - 1),
        ),
        origin=(b[0], b[2], b[4]),
    )
    accum = None
    mask = None
    for path in files:
        vol = pv.read(path)
        sampled = grid.sample(vol)
        u = np.asarray(sampled.point_data["Velocity"], dtype=np.float64)
        if accum is None:
            accum = u.copy()
            mask = np.asarray(sampled.point_data["vtkValidPointMask"],
                              dtype=bool)
        else:
            accum += u
    accum /= len(files)
    return grid, mask, accum


def pinn_mean_velocity_on_grid(arch, grid: pv.ImageData, mask: np.ndarray,
                               device, n_t: int = 25) -> np.ndarray:
    pts = np.asarray(grid.points, dtype=np.float32)
    pts_in = pts[mask]
    accum = np.zeros((pts_in.shape[0], 3), dtype=np.float64)
    t_samples = np.linspace(0.0, PERIOD, n_t, endpoint=False, dtype=np.float32)
    for t_val in t_samples:
        u_part = np.zeros_like(pts_in)
        for start in range(0, pts_in.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, pts_in.shape[0])
            sl = slice(start, end)
            xb = torch.tensor(pts_in[sl, 0], dtype=torch.float32,
                              device=device).reshape(-1, 1)
            yb = torch.tensor(pts_in[sl, 1], dtype=torch.float32,
                              device=device).reshape(-1, 1)
            zb = torch.tensor(pts_in[sl, 2], dtype=torch.float32,
                              device=device).reshape(-1, 1)
            tb = torch.full_like(xb, float(t_val))
            with torch.no_grad():
                out = arch({"x": xb, "y": yb, "z": zb, "t": tb})
            u_part[sl, 0] = out["u"].cpu().numpy().reshape(-1)
            u_part[sl, 1] = out["v"].cpu().numpy().reshape(-1)
            u_part[sl, 2] = out["w"].cpu().numpy().reshape(-1)
        accum += u_part
    accum /= n_t
    full = np.zeros((pts.shape[0], 3), dtype=np.float64)
    full[mask] = accum
    return full


def relative_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - ref)[mask]
    denom = np.linalg.norm(ref[mask])
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(diff) / denom)


# Per-patient pipeline

def process_patient(patient: Patient, device: torch.device,
                    checkpoint_suffix: str = "pulsatile") -> Dict[str, float]:
    print(f"\n[{patient.pid}] loading PINN checkpoint ({checkpoint_suffix})")
    ckpt_dir = os.path.join(CHECKPOINT_ROOT,
                            f"checkpoint_{patient.pid}_{checkpoint_suffix}")
    arch = load_pulsatile_network(ckpt_dir, device)

    stl = os.path.join(STL_ROOT, patient.pid, "aneurysm_noslip.stl")
    wall = sample_wall(stl, N_WALL_SAMPLES)

    print(f"[{patient.pid}] WSS over {N_TIME_SAMPLES + 1} time levels...")
    idx = time_integrated_indices(arch, wall, device, NU_CGS)

    print(f"[{patient.pid}] PINN-vs-FEM L2 velocity error...")
    grid, lumen_mask, fem_mean = cycle_mean_velocity_on_grid(patient.pid)
    pinn_mean = pinn_mean_velocity_on_grid(arch, grid, lumen_mask, device)
    rel_err = relative_l2(pinn_mean, fem_mean, lumen_mask)

    lsa = float((idx["tawss"] < LSA_THRESHOLD_PA).mean())
    # Region labels for the wall sample (dome / parent / other)
    wall_pts = np.column_stack([wall["x"], wall["y"], wall["z"]])
    try:
        regions = label_points(wall_pts, patient.pid)
    except Exception as exc:
        print(f"[{patient.pid}] region labelling failed: {exc}")
        regions = np.full(wall_pts.shape[0], "other", dtype="<U6")
    w = wall.get("area", None)
    def _avg(arr, m):
        if not m.any():
            return float("nan")
        if w is not None:
            ww = w[m]; total = ww.sum()
            return float(np.sum(arr[m] * ww) / total) if total > 0 else float("nan")
        return float(arr[m].mean())
    row = {
        "patient": patient.pid,
        "status": patient.status,
        "pair": patient.pair,
        "tawss_pa": float(idx["tawss"].mean()),
        "tawss_max_pa": float(idx["tawss"].max()),
        "osi": float(idx["osi"].mean()),
        "rrt": float(np.median(idx["rrt"])),
        "ecap": float(idx["ecap"].mean()),
        "lsa": lsa,
        "rel_l2_velocity": rel_err,
    }
    for region in ("dome", "parent"):
        m = (regions == region)
        row[f"n_{region}"] = int(m.sum())
        row[f"tawss_{region}"] = _avg(idx["tawss"], m)
        row[f"tawss_max_{region}"] = float(idx["tawss"][m].max()) if m.any() else float("nan")
        row[f"osi_{region}"] = _avg(idx["osi"], m)
        row[f"rrt_{region}"] = _avg(idx["rrt"], m)
        row[f"ecap_{region}"] = _avg(idx["ecap"], m)
        row[f"lsa_{region}"] = float((idx["tawss"][m] < LSA_THRESHOLD_PA).mean()) if m.any() else float("nan")
    def _safe_ratio(a, b):
        return float(a / b) if (b is not None and b == b and b > 0) else float("nan")
    row["ratio_tawss_dome_parent"] = _safe_ratio(row.get("tawss_dome"), row.get("tawss_parent"))
    row["ratio_osi_dome_parent"] = _safe_ratio(row.get("osi_dome"), row.get("osi_parent"))
    print(f"[{patient.pid}] TAWSS={row['tawss_pa']:.3f} Pa  "
          f"OSI={row['osi']:.4f}  L2_err={rel_err:.3f}")
    return row


# CSV / plots

def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    cols = ["patient", "status", "pair", "tawss_pa", "tawss_max_pa",
            "osi", "rrt", "ecap", "lsa", "rel_l2_velocity",
            "n_dome", "tawss_dome", "tawss_max_dome",
            "osi_dome", "rrt_dome", "ecap_dome", "lsa_dome",
            "n_parent", "tawss_parent", "tawss_max_parent",
            "osi_parent", "rrt_parent", "ecap_parent", "lsa_parent",
            "ratio_tawss_dome_parent", "ratio_osi_dome_parent"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"[pinn] CSV written: {path}")


def scatter_pinn_vs_fem(pinn_csv: str, fem_csv: str, out_dir: str) -> None:
    def load(p):
        with open(p) as fh:
            return list(csv.DictReader(fh))

    pinn = {r["patient"]: r for r in load(pinn_csv)}
    fem = {r["patient"]: r for r in load(fem_csv)}
    common = sorted(set(pinn) & set(fem))
    if not common:
        print("[pinn] no overlap with FEM CSV; skipping scatter plots.")
        return

    os.makedirs(out_dir, exist_ok=True)
    for field, title, unit in [("tawss_pa", "TAWSS", "Pa"),
                               ("osi", "OSI", "")]:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        xs = [float(fem[p][field]) for p in common]
        ys = [float(pinn[p][field]) for p in common]
        ax.scatter(xs, ys, color="black")
        for x, y, p in zip(xs, ys, common):
            ax.annotate(p, (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        lim = max(max(xs), max(ys), 1e-6) * 1.1
        ax.plot([0, lim], [0, lim], color="grey", linestyle="--",
                linewidth=1, label="y = x")
        ax.set_xlabel(f"FEM {title}" + (f" ({unit})" if unit else ""))
        ax.set_ylabel(f"PINN {title}" + (f" ({unit})" if unit else ""))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{title} on the aneurysm wall")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        out = os.path.join(out_dir, f"pinn_vs_fem_{field}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[pinn] figure written: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--out", default="results/hemodynamics_pinn.csv")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fem_csv", default="results/hemodynamics_fem.csv")
    ap.add_argument("--checkpoint_suffix", default="pulsatile")
    args = ap.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"[pinn] device: {device}")

    selected = [p for p in PATIENTS
                if args.patients is None or p.pid in args.patients]

    rows: List[Dict[str, float]] = []
    for patient in selected:
        try:
            rows.append(process_patient(patient, device,
                                        checkpoint_suffix=args.checkpoint_suffix))
        except Exception as exc:
            print(f"[{patient.pid}] FAILED: {exc}")

    write_csv(rows, args.out)
    if os.path.isfile(args.fem_csv):
        scatter_pinn_vs_fem(args.out, args.fem_csv,
                            os.path.join("results", "figures"))


if __name__ == "__main__":
    main()
