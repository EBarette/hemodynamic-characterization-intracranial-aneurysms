"""
compute_hemodynamics_fem.py - Hemodynamic indices from pulsatile svMultiPhysics.

For each patient, the script:
  * Locates available result_*.vtu files in 4-procs/
  * Selects the last full cardiac cycle (or, if not available, the latest
    100 saved steps as a partial cycle)
  * Extracts the wall surface from each VTU
  * Accumulates the WSS vector history (N_t, N_pts, 3) in dyne/cm^2
    (svMultiPhysics writes the wall shear stress vector in the `WSS`
    point array; the `Traction` array includes the pressure component
    and is therefore not used here)
  * Converts to Pa and computes per-node TAWSS, OSI, RRT, ECAP, then
    area-weighted averages and the LSA area fraction
  * Renders TAWSS and OSI surface maps with PyVista

Units (svMultiPhysics CGS): rho=1.06 g/cm^3, mu=0.04 g/(cm.s).
1 dyne/cm^2 = 0.1 Pa.

Output: results/hemodynamics_fem.csv (one row per patient).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from region_labels import label_wall_regions


DYNE_PER_CM2_TO_PA = 0.1
LSA_THRESHOLD_PA = 0.4
# three thresholds for LSA sensitivity (0.4 Pa = literature standard; 1.0 and 1.5 Pa additional)
LSA_THRESHOLDS_PA = (0.4, 1.0, 1.5)
DT_FEM = 0.0005
CYCLE_PERIOD = 0.95
SAVE_EVERY = 19
STEPS_PER_CYCLE = int(round(CYCLE_PERIOD / DT_FEM))  # 1900
FILES_PER_CYCLE = STEPS_PER_CYCLE // SAVE_EVERY      # 100


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


def vtu_directory(pid: str) -> str:
    root = os.environ.get("SV_OUTPUTS_ROOT",
                          os.path.expanduser("~/src/SimVascular/outputs"))
    refined = os.path.join(root, f"{pid}_H_CERE_CA_pulsatile_refined", "4-procs")
    if os.path.isdir(refined):
        return refined
    return os.path.join(root, f"{pid}_H_CERE_CA_pulsatile", "4-procs")


def list_analysis_vtus(pid: str) -> Tuple[List[str], Dict[str, int]]:
    """Return the last FILES_PER_CYCLE result_*.vtu paths and step metadata."""
    folder = vtu_directory(pid)
    candidates = sorted(glob(os.path.join(folder, "result_*.vtu")))
    parsed: List[Tuple[int, str]] = []
    for f in candidates:
        try:
            n = int(os.path.basename(f).replace("result_", "").replace(".vtu", ""))
        except ValueError:
            continue
        parsed.append((n, f))
    parsed.sort()
    if not parsed:
        raise FileNotFoundError(f"No result_*.vtu in {folder}")
    chosen = parsed[-FILES_PER_CYCLE:]
    meta = {
        "n_files": len(chosen),
        "first_step": chosen[0][0],
        "last_step": chosen[-1][0],
        "complete_cycle": int(len(chosen) == FILES_PER_CYCLE),
    }
    return [p for _, p in chosen], meta


WSS_ARRAY = "WSS"


def extract_wall(volume: pv.UnstructuredGrid) -> pv.PolyData:
    surf = volume.extract_surface()
    if WSS_ARRAY not in surf.point_data:
        raise RuntimeError(f"{WSS_ARRAY} array missing from VTU surface")
    return surf


def per_node_indices(wss_dyne: np.ndarray) -> Dict[str, np.ndarray]:
    tau = wss_dyne * DYNE_PER_CM2_TO_PA            # to Pa
    mag = np.linalg.norm(tau, axis=2)              # (N_t, N_pts)
    tawss = mag.mean(axis=0)
    mean_vec = tau.mean(axis=0)
    mean_mag = np.linalg.norm(mean_vec, axis=1)
    eps = 1e-12
    osi = 0.5 * (1.0 - mean_mag / np.maximum(tawss, eps))
    osi = np.clip(osi, 0.0, 0.5)
    # floor at 0.05 Pa to avoid 1/0 in RRT and ECAP at near-zero-WSS nodes
    tawss_floor = np.maximum(tawss, 0.05)
    rrt = 1.0 / (np.maximum(1.0 - 2.0 * osi, 1e-3) * tawss_floor)
    ecap = osi / tawss_floor
    # PSWSS: peak-systole |WSS| — frame with max area-mean WSS
    frame_mean = mag.mean(axis=1)
    peak_idx = int(np.argmax(frame_mean))
    pswss = mag[peak_idx]                          # (N_pts,) in Pa
    # Systolic-diastolic WSS range per node, in Pa.
    wss_range = mag.max(axis=0) - mag.min(axis=0)
    return {
        "tawss": tawss,
        "osi": osi,
        "rrt": rrt,
        "ecap": ecap,
        "pswss": pswss,
        "wss_range": wss_range,
        "_peak_idx": np.array([peak_idx], dtype=np.int32),
    }


def point_areas(surface: pv.PolyData) -> np.ndarray:
    surf = surface.triangulate().compute_cell_sizes(
        length=False, area=True, volume=False
    )
    cell_area = np.asarray(surf.cell_data["Area"])
    cells = surf.faces.reshape(-1, 4)
    nodal = np.zeros(surf.n_points, dtype=np.float64)
    np.add.at(nodal, cells[:, 1], cell_area / 3.0)
    np.add.at(nodal, cells[:, 2], cell_area / 3.0)
    np.add.at(nodal, cells[:, 3], cell_area / 3.0)
    return nodal


def area_weighted(values: np.ndarray, areas: np.ndarray) -> float:
    total = areas.sum()
    if total <= 0:
        return float("nan")
    return float(np.sum(values * areas) / total)


def surface_gradient_magnitude(surface: pv.PolyData, scalar_name: str) -> np.ndarray:
    """Per-node magnitude of the surface gradient of scalar_name (Pa/cm for TAWSS -> WSSG)."""
    triangulated = surface.triangulate()
    triangulated.point_data[scalar_name] = surface.point_data[scalar_name]
    deriv = triangulated.compute_derivative(scalars=scalar_name)
    grad = np.asarray(deriv.point_data["gradient"])
    return np.linalg.norm(grad, axis=1)


def render_surface_map(surface: pv.PolyData, field: str, out_path: str,
                       label: str, clim: Tuple[float, float]) -> None:
    p = pv.Plotter(off_screen=True, window_size=(900, 700))
    p.add_mesh(surface, scalars=field, cmap="viridis", clim=clim,
               scalar_bar_args={"title": label, "n_labels": 5})
    p.view_isometric()
    p.screenshot(out_path)
    p.close()


def process_patient(patient: Patient, fig_dir: str) -> Dict[str, float]:
    files, meta = list_analysis_vtus(patient.pid)
    print(f"[{patient.pid}] using {meta['n_files']} VTUs "
          f"(steps {meta['first_step']}..{meta['last_step']}, "
          f"complete_cycle={meta['complete_cycle']})")

    first = pv.read(files[0])
    wall = extract_wall(first)
    n_pts = wall.n_points
    wss_hist = np.zeros((len(files), n_pts, 3), dtype=np.float32)
    wss_hist[0] = np.asarray(wall.point_data[WSS_ARRAY], dtype=np.float32)

    for i, path in enumerate(files[1:], start=1):
        vol = pv.read(path)
        surf = vol.extract_surface()
        t = np.asarray(surf.point_data[WSS_ARRAY], dtype=np.float32)
        if t.shape[0] != n_pts:
            raise RuntimeError(
                f"[{patient.pid}] surface point count changed: "
                f"{n_pts} vs {t.shape[0]} at {path}"
            )
        wss_hist[i] = t

    idx = per_node_indices(wss_hist)
    peak_idx = int(idx.pop("_peak_idx")[0])
    areas = point_areas(wall)
    # Wall shear stress gradient (WSSG) in Pa/cm: surface gradient of TAWSS.
    wall.point_data["tawss"] = idx["tawss"]
    wssg = surface_gradient_magnitude(wall, "tawss")
    idx["wssg"] = wssg
    try:
        regions = label_wall_regions(wall, patient.pid)
    except Exception as exc:
        print(f"[{patient.pid}] region labelling failed: {exc}")
        regions = np.full(wall.n_points, "other", dtype="<U6")

    for name, arr in idx.items():
        wall.point_data[name] = arr
    region_int = np.zeros(wall.n_points, dtype=np.int8)
    region_int[regions == "dome"] = 1
    region_int[regions == "parent"] = 2
    wall.point_data["region"] = region_int

    os.makedirs(fig_dir, exist_ok=True)
    render_surface_map(wall, "tawss",
                       os.path.join(fig_dir, f"tawss_{patient.pid}.png"),
                       "TAWSS (Pa)", (0.0, 2.0))
    render_surface_map(wall, "osi",
                       os.path.join(fig_dir, f"osi_{patient.pid}.png"),
                       "OSI", (0.0, 0.5))

    def _lsa_key(thr: float) -> str:
        return f"lsa_t{int(round(thr * 10)):02d}"

    out = {
        "patient": patient.pid,
        "status": patient.status,
        "pair": patient.pair,
        "n_files": meta["n_files"],
        "first_step": meta["first_step"],
        "last_step": meta["last_step"],
        "complete_cycle": meta["complete_cycle"],
        "peak_systole_frame": peak_idx,
        "area_cm2": float(areas.sum()),
        "tawss_pa": area_weighted(idx["tawss"], areas),
        "tawss_max_pa": float(idx["tawss"].max()),
        "osi": area_weighted(idx["osi"], areas),
        "rrt": area_weighted(idx["rrt"], areas),
        "ecap": area_weighted(idx["ecap"], areas),
        "wssg": area_weighted(idx["wssg"], areas),
        "pswss": area_weighted(idx["pswss"], areas),
        "wss_range": area_weighted(idx["wss_range"], areas),
    }
    for thr in LSA_THRESHOLDS_PA:
        out[_lsa_key(thr)] = float(
            np.sum((idx["tawss"] < thr) * areas) / areas.sum()
        )
    out["lsa"] = out[_lsa_key(LSA_THRESHOLD_PA)]  # 0.4 Pa threshold

    # Per-region area-weighted indices (dome / parent)
    for region in ("dome", "parent"):
        m = (regions == region)
        if m.any():
            a = areas[m]
            tawss_r = idx["tawss"][m]
            out[f"area_{region}_cm2"] = float(a.sum())
            out[f"tawss_{region}"] = area_weighted(tawss_r, a)
            out[f"tawss_max_{region}"] = float(tawss_r.max())
            out[f"osi_{region}"] = area_weighted(idx["osi"][m], a)
            out[f"rrt_{region}"] = area_weighted(idx["rrt"][m], a)
            out[f"ecap_{region}"] = area_weighted(idx["ecap"][m], a)
            out[f"wssg_{region}"] = area_weighted(idx["wssg"][m], a)
            out[f"pswss_{region}"] = area_weighted(idx["pswss"][m], a)
            out[f"wss_range_{region}"] = area_weighted(idx["wss_range"][m], a)
            for thr in LSA_THRESHOLDS_PA:
                out[f"{_lsa_key(thr)}_{region}"] = float(
                    np.sum((tawss_r < thr) * a) / a.sum()
                )
            out[f"lsa_{region}"] = out[f"{_lsa_key(LSA_THRESHOLD_PA)}_{region}"]
        else:
            for k in ("area_cm2", "tawss", "tawss_max", "osi", "rrt",
                     "ecap", "wssg", "pswss", "wss_range", "lsa"):
                key = f"area_{region}_cm2" if k == "area_cm2" else f"{k}_{region}"
                out[key] = float("nan")
            for thr in LSA_THRESHOLDS_PA:
                out[f"{_lsa_key(thr)}_{region}"] = float("nan")

    # Dome-to-parent ratios for every base index.
    def _safe_ratio(a, b):
        return float(a / b) if (b is not None and b == b and b > 0) else float("nan")
    for base in ("tawss", "osi", "rrt", "ecap", "wssg", "pswss",
                 "wss_range", "lsa"):
        out[f"ratio_{base}_dome_parent"] = _safe_ratio(
            out.get(f"{base}_dome"), out.get(f"{base}_parent")
        )
    for thr in LSA_THRESHOLDS_PA:
        key = _lsa_key(thr)
        out[f"ratio_{key}_dome_parent"] = _safe_ratio(
            out.get(f"{key}_dome"), out.get(f"{key}_parent")
        )

    # PSWSS/TAWSS peaking factor
    for region in ("", "_dome", "_parent"):
        ts = out.get(f"tawss{region}" if region else "tawss_pa")
        ps = out.get(f"pswss{region}" if region else "pswss")
        key = f"pswss_over_tawss{region}" if region else "pswss_over_tawss"
        out[key] = _safe_ratio(ps, ts)
    return out


def write_csv(rows: List[Dict[str, float]], path: str) -> None:
    # Use the union of all row keys, in insertion order from the first row,
    # so the additional metric columns appear automatically.
    seen: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(",".join(seen) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in seen) + "\n")
    print(f"[fem] CSV written: {path}")


def box_plots(rows: List[Dict[str, float]], out_path: str) -> None:
    fields = ["tawss_pa", "osi", "rrt", "ecap", "lsa"]
    titles = ["TAWSS (Pa)", "OSI", "RRT (Pa$^{-1}$)",
              "ECAP (Pa$^{-1}$)", "LSA"]
    fig, axes = plt.subplots(1, len(fields), figsize=(3.0 * len(fields), 3.5))
    for ax, f, t in zip(axes, fields, titles):
        grow = [r[f] for r in rows if r["status"] == "growing"]
        stab = [r[f] for r in rows if r["status"] == "stable"]
        ax.boxplot([grow, stab], tick_labels=["Growing", "Stable"])
        for i, vals in enumerate([grow, stab], start=1):
            ax.scatter([i] * len(vals), vals, color="black", zorder=3, s=15)
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Wall hemodynamic indices, FEM")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--out", default="results/hemodynamics_fem.csv")
    ap.add_argument("--fig_dir", default="results/figures")
    args = ap.parse_args()

    selected = [p for p in PATIENTS
                if args.patients is None or p.pid in args.patients]

    rows: List[Dict[str, float]] = []
    for patient in selected:
        try:
            rows.append(process_patient(patient, args.fig_dir))
        except Exception as exc:
            print(f"[{patient.pid}] FAILED: {exc}")

    write_csv(rows, args.out)
    box_plots(rows, os.path.join(args.fig_dir, "boxplots_fem_wall.png"))


if __name__ == "__main__":
    main()
