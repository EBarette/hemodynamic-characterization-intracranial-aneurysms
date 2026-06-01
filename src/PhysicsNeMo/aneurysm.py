# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import warnings

import torch
import numpy as np
from omegaconf import OmegaConf
from sympy import Symbol, sqrt, Max

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.domain.inferencer import VoxelInferencer # ADDED


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    def _cfg_tuple(key, default):
        value = OmegaConf.select(cfg, key, default=default)
        return tuple(value)

    def _cfg_float(key, default):
        value = OmegaConf.select(cfg, key, default=default)
        return float(value)

    def _boundary_center_and_normal(mesh, npoints=10000):
        sampled = mesh.sample_boundary(npoints)
        cx = float(np.mean(np.asarray(sampled["x"]).reshape(-1)))
        cy = float(np.mean(np.asarray(sampled["y"]).reshape(-1)))
        cz = float(np.mean(np.asarray(sampled["z"]).reshape(-1)))
        nx = float(np.mean(np.asarray(sampled["normal_x"]).reshape(-1)))
        ny = float(np.mean(np.asarray(sampled["normal_y"]).reshape(-1)))
        nz = float(np.mean(np.asarray(sampled["normal_z"]).reshape(-1)))
        n_len = float(np.linalg.norm([nx, ny, nz]))
        if n_len < 1e-12:
            return (cx, cy, cz), (0.0, 0.0, 0.0)
        return (cx, cy, cz), (nx / n_len, ny / n_len, nz / n_len)

    def _normal_points_outward(center, normal, interior_mesh):
        """True if `normal` at `center` points outside `interior_mesh`.
        Nudges the centroid along the normal and checks SDF sign.
        Tessellation.sdf is positive inside the mesh, so the nudged
        point is outside when its SDF value is negative.
        Queries two points so Tessellation.sdf doesn't divide-by-zero.
        """
        bounds = {str(k): v for k, v in interior_mesh.bounds.bound_ranges.items()}
        max_extent = max(b[1] - b[0] for b in bounds.values())
        eps = max_extent * 0.01
        nx, ny, nz = normal
        cx, cy, cz = center
        test = {
            "x": np.array([[cx + eps * nx], [cx - eps * nx]]),
            "y": np.array([[cy + eps * ny], [cy - eps * ny]]),
            "z": np.array([[cz + eps * nz], [cz - eps * nz]]),
        }
        sdf = interior_mesh.sdf(test, {})
        sdf_vals = np.asarray(sdf["sdf"]).reshape(-1)
        return float(sdf_vals[0]) < 0

    def _estimate_circular_inlet(mesh, npoints):
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
        return (cx, cy, cz), normal, radius, area

    # read stl files to make geometry
    stl_dir = OmegaConf.select(cfg, "custom.stl_dir", default="./stl_files")
    point_path = to_absolute_path(stl_dir)
    inlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_inlet.stl", airtight=False
    )
    noslip_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_noslip.stl", airtight=False
    )
    integral_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_integral.stl", airtight=False
    )
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    # Auto-discover all outlet STLs (aneurysm_outlet_01.stl, _02.stl, ...).
    outlet_paths = sorted(glob.glob(point_path + "/aneurysm_outlet_*.stl"))
    if not outlet_paths:
        raise FileNotFoundError(
            f"No outlet STLs found matching {point_path}/aneurysm_outlet_*.stl"
        )
    outlet_meshes = [
        (os.path.splitext(os.path.basename(p))[0], Tessellation.from_stl(p, airtight=False))
        for p in outlet_paths
    ]

    # params
    nu = _cfg_float("custom.nu", 0.025)
    inlet_vel = _cfg_float("custom.inlet_vel", 1.5)

    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # estimate inlet geometry from mesh
    npoints = int(OmegaConf.select(cfg, "custom.auto_inlet_npoints", default=10000))
    inlet_center, inlet_normal, inlet_radius, _ = _estimate_circular_inlet(
        inlet_mesh, npoints
    )

    # Inlet: ensure the normal points INTO the vessel (inward).
    inlet_normal_flipped = False
    if _normal_points_outward(inlet_center, inlet_normal, interior_mesh):
        inlet_normal = tuple(-n for n in inlet_normal)
        inlet_normal_flipped = True

    # Per-outlet: boundary cap, SDF tells us if normal points outward.
    # Also compute the cap area for area-weighted flux distribution.
    outlet_info = []  # list of dicts: name, mesh, center, normal, sign, area
    for name, mesh in outlet_meshes:
        c, n = _boundary_center_and_normal(mesh)
        sign = 1.0 if _normal_points_outward(c, n, interior_mesh) else -1.0
        _, _, _, area = _estimate_circular_inlet(mesh, npoints)
        outlet_info.append(
            {"name": name, "mesh": mesh, "center": c, "normal": n,
             "sign": sign, "area": area}
        )

    # Integral: internal cross-section — both sides are inside the vessel
    # so SDF cannot distinguish.  Use the inlet -> integral direction instead
    integral_c, integral_n = _boundary_center_and_normal(integral_mesh)
    integral_flux_sign = 1.0 if np.dot(integral_n, np.array(inlet_normal)) > 0 else -1.0

    # For a circular parabolic profile with max velocity Umax, volumetric flux is
    # Q = (1/2) * pi * R^2 * Umax.
    base_flux = 0.5 * np.pi * inlet_radius**2 * inlet_vel

    # Distribute inlet flux among outlets.
    # Default: area-weighted so sum(Q_outlet) == Q_inlet.
    # Override: cfg.custom.outlet_flux_fractions = [f1, f2, ...] (must match
    # number of outlets and ideally sum to 1).
    n_outlets = len(outlet_info)
    fractions_cfg = OmegaConf.select(cfg, "custom.outlet_flux_fractions", default=None)
    if fractions_cfg is not None:
        fractions = [float(f) for f in fractions_cfg]
        if len(fractions) != n_outlets:
            raise ValueError(
                f"custom.outlet_flux_fractions has {len(fractions)} entries "
                f"but {n_outlets} outlets were discovered."
            )
        s = sum(fractions)
        if s <= 0:
            raise ValueError("custom.outlet_flux_fractions must sum to > 0.")
        if not np.isclose(s, 1.0, atol=1e-6):
            warnings.warn(
                f"custom.outlet_flux_fractions sum to {s:.6g} (expected 1.0); "
                "using values as given."
            )
    else:
        total_area = sum(o["area"] for o in outlet_info)
        fractions = [o["area"] / total_area for o in outlet_info]

    for o, f in zip(outlet_info, fractions):
        o["fraction"] = f
        o["flux_target"] = o["sign"] * f * base_flux

    integral_flux_target = integral_flux_sign * base_flux
    integral_lambda = _cfg_float("custom.integral_lambda", 0.1)

    outlets_str = ", ".join(
        f"{o['name']}(frac={o['fraction']:.4g}, sign={o['sign']:+.0f}, "
        f"Q={o['flux_target']:.6g})"
        for o in outlet_info
    )
    print(
        "[aneurysm-startup] "
        f"stl_dir={stl_dir} "
        f"inlet_center={inlet_center} inlet_normal={inlet_normal} "
        f"(flipped={inlet_normal_flipped}) inlet_radius={inlet_radius:.8g} "
        f"Q_base={base_flux:.8g} "
        f"n_outlets={n_outlets} outlets=[{outlets_str}] "
        f"integral_Q={integral_flux_target:.8g} (sign={integral_flux_sign:+.0f}) "
        f"integral_lambda={integral_lambda:.8g}"
    )

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    # Resolve the arch config dynamically so any arch (fully_connected,
    # modified_fourier, etc.) can be selected purely via the config file.
    _arch_key = next(iter(cfg.arch))
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch[_arch_key],
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet, "inlet")

    # outlets (p = 0 on each outlet cap).  Split the configured outlet batch
    # size across all discovered outlets so total work stays comparable to
    # the single-outlet case.
    per_outlet_bs = max(1, cfg.batch_size.outlet // n_outlets)
    for o in outlet_info:
        outlet = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=o["mesh"],
            outvar={"p": 0},
            batch_size=per_outlet_bs,
        )
        domain.add_constraint(outlet, o["name"])

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
    )
    domain.add_constraint(interior, "interior")

    # Integral continuity: one constraint per outlet + one on the internal
    # integral cross-section.
    for idx, o in enumerate(outlet_info, start=1):
        ic = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=o["mesh"],
            outvar={"normal_dot_vel": o["flux_target"]},
            batch_size=1,
            integral_batch_size=cfg.batch_size.integral_continuity,
            lambda_weighting={"normal_dot_vel": integral_lambda},
        )
        domain.add_constraint(ic, f"integral_continuity_outlet_{idx:02d}")

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": integral_flux_target},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": integral_lambda},
    )
    domain.add_constraint(integral_continuity, "integral_continuity_internal")

    # add validation data
    file_path = "./openfoam/aneurysm_parabolicInlet_sol0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p": "p",
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        openfoam_invar = {
            key: value for key, value in openfoam_var.items() if key in ["x", "y", "z"]
        }
        openfoam_outvar = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar,
            true_outvar=openfoam_outvar,
            batch_size=4096,
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/physicsnemo_sym_examples_supplemental_materials"
        )

    # add pressure monitor
    pressure_monitor = PointwiseMonitor(
        inlet_mesh.sample_boundary(16),
        output_names=["p"],
        metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    # WSS monitor via finite differences.
    # PointwiseMonitor evaluates without requires_grad, so autograd derivatives
    # (u__x, etc.) are not available here.  Instead we exploit the no-slip BC
    # (u = 0 at the wall) to estimate the wall-normal velocity gradient via FD:
    #   |∂u/∂n| ≈ |u(x - ε·n̂)| / ε
    # where n̂ is the outward wall normal (pointing into the solid), so
    # (x - ε·n̂) lies just inside the fluid domain.
    # WSS ≈ ρ·ν · |∂u/∂n|_wall  (with ρ=1 in the normalised problem)
    eps_fd = _cfg_float("custom.wss_fd_eps", 1e-3)
    _wall_sample = noslip_mesh.sample_boundary(2000)
    wss_invar = {
        "x": np.asarray(_wall_sample["x"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_x"]).reshape(-1, 1),
        "y": np.asarray(_wall_sample["y"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_y"]).reshape(-1, 1),
        "z": np.asarray(_wall_sample["z"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_z"]).reshape(-1, 1),
    }

    def _wss_mean_fd(var):
        u, v, w = var["u"], var["v"], var["w"]
        # |u| / ε  gives the wall-normal velocity gradient magnitude
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + w ** 2)
        return torch.mean(nu * vel_mag / eps_fd)

    wss_monitor = PointwiseMonitor(
        wss_invar,
        output_names=["u", "v", "w"],
        metrics={"wss_mean": _wss_mean_fd},
        nodes=nodes,
    )
    domain.add_monitor(wss_monitor, "wss_monitor")
    
    # ADDED
    bounds_dict = {str(k): v for k, v in interior_mesh.bounds.bound_ranges.items()}
    bounds = [bounds_dict["x"], bounds_dict["y"], bounds_dict["z"]]

    # optional
    pad = 0.05
    bounds = [(lo - (hi - lo) * pad, hi + (hi - lo) * pad) for (lo, hi) in bounds]


    def mask_fn(x, y, z):
        sdf = interior_mesh.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] < 0

    voxel_inf = VoxelInferencer(
        bounds=bounds,
        npoints=[128, 128, 128],              # increase to 192/256 if you can afford it
        nodes=nodes,
        output_names=["u", "v", "w", "p"],
        export_map={"U": ["u", "v", "w"], "p": ["p"]},  # U becomes a VTK vector
        batch_size=8192,
        mask_fn=mask_fn,
    )
    domain.add_inferencer(voxel_inf, "aneurysm_voxel")
    # ADDED

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()




# in $GLOBALSCRATCH/physicsnemo-py312/lib/python3.12/site-packages/physicsnemo/sym/loss/aggregator.py
# line 369
# - if self.ref_key is None:
# -     ref_idx = 0
# - else:
# + ref_idx = 0
# + if self.ref_key is not None: