import argparse
from pathlib import Path

import numpy as np

from physicsnemo.sym.geometry.tessellation import Tessellation


REQUIRED_STL_NAMES = [
    "aneurysm_inlet.stl",
    "aneurysm_outlet_01.stl",
    "aneurysm_noslip.stl",
    "aneurysm_integral.stl",
    "aneurysm_closed.stl",
]


def mesh_bounds(mesh):
    bounds_dict = {str(k): v for k, v in mesh.bounds.bound_ranges.items()}
    return {
        "x": tuple(bounds_dict["x"]),
        "y": tuple(bounds_dict["y"]),
        "z": tuple(bounds_dict["z"]),
    }


def bounds_center_max_extent(bounds):
    center = tuple((bounds[axis][0] + bounds[axis][1]) * 0.5 for axis in ("x", "y", "z"))
    max_extent = max(bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z"))
    return center, float(max_extent)


def estimate_circular_inlet(mesh, npoints):
    sampled = mesh.sample_boundary(npoints)
    x = np.asarray(sampled["x"]).reshape(-1)
    y = np.asarray(sampled["y"]).reshape(-1)
    z = np.asarray(sampled["z"]).reshape(-1)

    if "area" in sampled:
        w = np.asarray(sampled["area"]).reshape(-1)
        w_sum = float(np.sum(w))
        if w_sum > 0:
            cx = float(np.sum(w * x) / w_sum)
            cy = float(np.sum(w * y) / w_sum)
            cz = float(np.sum(w * z) / w_sum)
            area = w_sum
        else:
            cx, cy, cz = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
            area = float(np.pi)
    else:
        cx, cy, cz = float(np.mean(x)), float(np.mean(y)), float(np.mean(z))
        area = float(np.pi)

    normal = None
    if all(k in sampled for k in ("normal_x", "normal_y", "normal_z")):
        nx = float(np.mean(np.asarray(sampled["normal_x"]).reshape(-1)))
        ny = float(np.mean(np.asarray(sampled["normal_y"]).reshape(-1)))
        nz = float(np.mean(np.asarray(sampled["normal_z"]).reshape(-1)))
        n_norm = float(np.linalg.norm([nx, ny, nz]))
        if n_norm > 1e-12:
            normal = (nx / n_norm, ny / n_norm, nz / n_norm)

    radius = float(np.sqrt(max(area, 1e-16) / np.pi))
    return (cx, cy, cz), normal, radius, float(area)


def load_set(stl_dir):
    meshes = {}
    for name in REQUIRED_STL_NAMES:
        p = Path(stl_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"Missing STL file: {p}")
        airtight = name == "aneurysm_closed.stl"
        meshes[name] = Tessellation.from_stl(str(p), airtight=airtight)
    return meshes


def summarize_set(label, meshes, inlet_vel, npoints):
    closed_bounds = mesh_bounds(meshes["aneurysm_closed.stl"])
    closed_center, closed_max_extent = bounds_center_max_extent(closed_bounds)

    inlet_center, inlet_normal, inlet_radius, inlet_area = estimate_circular_inlet(
        meshes["aneurysm_inlet.stl"], npoints
    )
    q_base = 0.5 * np.pi * inlet_radius**2 * inlet_vel

    print(f"[{label}] aneurysm_closed bounds: {closed_bounds}")
    print(f"[{label}] recommended center (bbox midpoint): {closed_center}")
    print(f"[{label}] closed max extent: {closed_max_extent:.8g}")
    print(f"[{label}] inlet center estimate: {inlet_center}")
    print(f"[{label}] inlet normal estimate: {inlet_normal}")
    print(f"[{label}] inlet area estimate: {inlet_area:.8g}")
    print(f"[{label}] inlet radius estimate: {inlet_radius:.8g}")
    print(f"[{label}] flux estimate Q=0.5*pi*R^2*Umax with Umax={inlet_vel:.8g}: {q_base:.8g}")

    return {
        "center": closed_center,
        "max_extent": closed_max_extent,
        "inlet_center": inlet_center,
        "inlet_normal": inlet_normal,
        "inlet_radius": inlet_radius,
        "flux": float(q_base),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect aneurysm STL sets and compare scales")
    parser.add_argument("--source", required=True, help="Path to source STL directory (e.g. ./stl_files)")
    parser.add_argument("--target", required=True, help="Path to target STL directory (e.g. ./stanford_stl_files)")
    parser.add_argument("--inlet-vel", type=float, default=1.5, help="Inlet max velocity Umax")
    parser.add_argument("--sample-n", type=int, default=8192, help="Boundary samples for inlet estimation")
    parser.add_argument(
        "--target-max-extent",
        type=float,
        default=8.0,
        help="Reference max extent after normalization (used to suggest scale)",
    )
    args = parser.parse_args()

    src_meshes = load_set(args.source)
    tgt_meshes = load_set(args.target)

    src = summarize_set("source", src_meshes, args.inlet_vel, args.sample_n)
    tgt = summarize_set("target", tgt_meshes, args.inlet_vel, args.sample_n)

    if tgt["max_extent"] > 1e-12:
        extent_ratio = src["max_extent"] / tgt["max_extent"]
    else:
        extent_ratio = float("nan")

    if tgt["max_extent"] > 1e-12:
        suggested_scale_target = args.target_max_extent / tgt["max_extent"]
    else:
        suggested_scale_target = float("nan")

    print("[compare]")
    print(f"[compare] max_extent_ratio(source/target): {extent_ratio:.8g}")
    print(
        "[compare] suggested target normalization params: "
        f"center={tgt['center']}, scale={suggested_scale_target:.8g} "
        f"(for target_max_extent={args.target_max_extent:.8g})"
    )


if __name__ == "__main__":
    main()
