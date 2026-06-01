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
# Pulsatile (time-dependent) variant of aneurysm.py.
#
# The geometry / inlet-cap / outlet-cap discovery logic is *copied verbatim*
# from the steady-state script (per the task brief).  Only the parts that
# differ for unsteady (t-dependent) flow are commented inline with the marker
# "# PULSATILE:" so a diff against aneurysm.py is easy to follow.
#
# Key differences vs. the steady version:
#   1. Network takes 4 inputs (x, y, z, t).
#   2. NavierStokes is built with time=True (adds the ∂u/∂t terms).
#   3. All physics constraints carry a Parameterization over t ∈ [0, T].
#   4. The inlet parabolic profile is modulated by a periodic waveform g(t),
#      so Q_in(t) = base_flux · g(t); outlet & integral-cap flux targets get
#      the same g(t) factor so global mass balance still holds at every t.
#   5. We do NOT add a VoxelInferencer here — at training time we don't know
#      which time slice the user wants; OSI/TAWSS are recovered in the
#      separate post-processing script `compute_osi.py`.
#   6. We do NOT touch / load the OpenFOAM steady validator (it has no t
#      column).
import os
import glob
import warnings

import torch
import numpy as np
from omegaconf import OmegaConf
import sympy
from sympy import Symbol, sqrt, Max, sin, cos, pi

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.geometry.tessellation import Tessellation
# PULSATILE: Parameterization is how PhysicsNeMo Sym exposes extra symbolic
# inputs (here: the time variable t) to every constraint.
from physicsnemo.sym.geometry.parameterization import Parameterization


@physicsnemo.sym.main(config_path="conf", config_name="config_pulsatile")
def run(cfg: PhysicsNeMoConfig) -> None:
    def _cfg_float(key, default):
        value = OmegaConf.select(cfg, key, default=default)
        return float(value)

    # ---- geometry helpers (verbatim from aneurysm.py) -----------------------
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

    # ---- STL loading (verbatim) --------------------------------------------
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

    outlet_paths = sorted(glob.glob(point_path + "/aneurysm_outlet_*.stl"))
    if not outlet_paths:
        raise FileNotFoundError(
            f"No outlet STLs found matching {point_path}/aneurysm_outlet_*.stl"
        )
    outlet_meshes = [
        (os.path.splitext(os.path.basename(p))[0], Tessellation.from_stl(p, airtight=False))
        for p in outlet_paths
    ]

    # ---- physical params ----------------------------------------------------
    nu = _cfg_float("custom.nu", 0.025)
    inlet_vel = _cfg_float("custom.inlet_vel", 1.5)   # peak / mean (see below)

    # PULSATILE: cardiac-cycle parameters
    # ------------------------------------
    # T          : period of one cardiac cycle [s].
    # T_period    : cardiac cycle [s]; overridden by flow_file last time if used.
    # n_train_periods : training t-range is [0, n * T] (1 = one full cycle).
    # Waveform modes (mutually exclusive config keys):
    #   A. custom.flow_file  → multi-mode Fourier series fitted to a two-column
    #      time/Q file (e.g. the SimVascular ICA.flow). n_fourier_modes modes.
    #      waveform(t) = Q(t)/Q_mean; always > 0 since Q < 0 (inflow) always.
    #   B. custom.pulse_amp  → single sinusoidal (legacy / backward compat).
    #      waveform(t) = 1 + pulse_amp * sin(2*pi*t/T)
    T_period = _cfg_float("custom.period", 0.8)
    n_train_periods = _cfg_float("custom.n_train_periods", 1.0)

    # t_sym must be defined before the waveform expression is built.
    t_sym = Symbol("t")

    flow_file_cfg = OmegaConf.select(cfg, "custom.flow_file", default=None)
    if flow_file_cfg is not None:
        # ---- A: Fourier-series waveform from a flow file -------------------
        _flow_path = to_absolute_path(str(flow_file_cfg))
        _flow_data = np.loadtxt(_flow_path)
        _t_flow, _Q_flow = _flow_data[:, 0], _flow_data[:, 1]
        T_period = float(_t_flow[-1])        # override T from file
        _t_fine = np.linspace(0.0, T_period, 4000, endpoint=False)
        _Q_fine = np.interp(_t_fine, _t_flow, _Q_flow)
        n_modes = int(OmegaConf.select(cfg, "custom.n_fourier_modes", default=10))
        _a = np.zeros(n_modes + 1)
        _b = np.zeros(n_modes + 1)
        _a[0] = float(np.mean(_Q_fine))   # DC = Q_mean (< 0 for inflow)
        for _k in range(1, n_modes + 1):
            _a[_k] = 2.0 * float(np.mean(_Q_fine * np.cos(2 * np.pi * _k * _t_fine / T_period)))
            _b[_k] = 2.0 * float(np.mean(_Q_fine * np.sin(2 * np.pi * _k * _t_fine / T_period)))
        _Q_mean = _a[0]  # < 0; Q(t)/_Q_mean > 0 (both negative), so waveform > 0
        waveform = 1
        for _k in range(1, n_modes + 1):
            _ck = _a[_k] / _Q_mean
            _sk = _b[_k] / _Q_mean
            if abs(_ck) > 1e-12:
                waveform = waveform + _ck * cos(2 * pi * _k * t_sym / T_period)
            if abs(_sk) > 1e-12:
                waveform = waveform + _sk * sin(2 * pi * _k * t_sym / T_period)
        _waveform_desc = f"Fourier({n_modes} modes, Q_mean={_Q_mean:.4f})"
        print(
            f"[aneurysm-pulsatile] waveform from {_flow_path}: "
            f"T={T_period:.4f}s, Q_mean={_Q_mean:.4f} cm\u00b3/s, {n_modes} modes"
        )
    else:
        # ---- B: legacy single sinusoidal mode ------------------------------
        pulse_amp = _cfg_float("custom.pulse_amp", 0.5)
        waveform = 1 + pulse_amp * sin(2 * pi * t_sym / T_period)
        _waveform_desc = f"sinusoidal(amp={pulse_amp})"

    t_max = n_train_periods * T_period
    time_range = (0.0, t_max)

    # ---- inlet parabolic profile (verbatim except for waveform factor) -----
    def circular_parabola(x, y, z, center, normal, radius, max_vel, time_factor):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        # PULSATILE: scale the parabola by waveform(t).
        parabola = max_vel * time_factor * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    npoints = int(OmegaConf.select(cfg, "custom.auto_inlet_npoints", default=10000))
    inlet_center, inlet_normal, inlet_radius, _ = _estimate_circular_inlet(
        inlet_mesh, npoints
    )

    inlet_normal_flipped = False
    if _normal_points_outward(inlet_center, inlet_normal, interior_mesh):
        inlet_normal = tuple(-n for n in inlet_normal)
        inlet_normal_flipped = True

    outlet_info = []
    for name, mesh in outlet_meshes:
        c, n = _boundary_center_and_normal(mesh)
        sign = 1.0 if _normal_points_outward(c, n, interior_mesh) else -1.0
        _, _, _, area = _estimate_circular_inlet(mesh, npoints)
        outlet_info.append(
            {"name": name, "mesh": mesh, "center": c, "normal": n,
             "sign": sign, "area": area}
        )

    integral_c, integral_n = _boundary_center_and_normal(integral_mesh)
    integral_flux_sign = 1.0 if np.dot(integral_n, np.array(inlet_normal)) > 0 else -1.0

    base_flux = 0.5 * np.pi * inlet_radius**2 * inlet_vel

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
        # PULSATILE: time-varying flux target. _compute_outvar() handles
        # SymPy expressions involving the parameterization symbols.
        o["flux_target"] = o["sign"] * f * base_flux * waveform

    integral_flux_target = integral_flux_sign * base_flux * waveform
    integral_lambda = _cfg_float("custom.integral_lambda", 0.1)

    outlets_str = ", ".join(
        f"{o['name']}(frac={o['fraction']:.4g}, sign={o['sign']:+.0f})"
        for o in outlet_info
    )
    print(
        "[aneurysm-pulsatile-startup] "
        f"stl_dir={stl_dir} "
        f"inlet_center={inlet_center} inlet_normal={inlet_normal} "
        f"(flipped={inlet_normal_flipped}) inlet_radius={inlet_radius:.8g} "
        f"Q_base(mean)={base_flux:.8g} "
        f"T_period={T_period} waveform=[{_waveform_desc}] "
        f"t_range=[0,{t_max}] (n_periods={n_train_periods}) "
        f"n_outlets={n_outlets} outlets=[{outlets_str}] "
        f"integral_sign={integral_flux_sign:+.0f} "
        f"integral_lambda={integral_lambda:.8g}"
    )

    # ---- domain / network --------------------------------------------------
    domain = Domain()
    # PULSATILE: time=True activates ∂/∂t terms in continuity/momentum.
    ns = NavierStokes(nu=nu, rho=1.0, dim=3, time=True)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])

    _arch_key = next(iter(cfg.arch))
    # PULSATILE: 4-D input (add Key("t")).
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch[_arch_key],
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # PULSATILE: single parameterization object reused by every constraint.
    param_t = Parameterization({t_sym: time_range})

    # ---- constraints --------------------------------------------------------
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
        time_factor=waveform,
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
        parameterization=param_t,
    )
    domain.add_constraint(inlet, "inlet")

    per_outlet_bs = max(1, cfg.batch_size.outlet // n_outlets)
    for o in outlet_info:
        outlet = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=o["mesh"],
            outvar={"p": 0},
            batch_size=per_outlet_bs,
            parameterization=param_t,
        )
        domain.add_constraint(outlet, o["name"])

    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=param_t,
    )
    domain.add_constraint(no_slip, "no_slip")

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=param_t,
    )
    domain.add_constraint(interior, "interior")

    for idx, o in enumerate(outlet_info, start=1):
        ic = IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=o["mesh"],
            outvar={"normal_dot_vel": o["flux_target"]},
            batch_size=1,
            integral_batch_size=cfg.batch_size.integral_continuity,
            lambda_weighting={"normal_dot_vel": integral_lambda},
            parameterization=param_t,
        )
        domain.add_constraint(ic, f"integral_continuity_outlet_{idx:02d}")

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": integral_flux_target},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": integral_lambda},
        parameterization=param_t,
    )
    domain.add_constraint(integral_continuity, "integral_continuity_internal")

    # ---- hybrid supervised constraint (optional) ---------------------------
    # PULSATILE+HYBRID: if `custom.hybrid_dataset` is set, load an NPZ file
    # containing pre-sampled FEM (x,y,z,t,u,v,w,p) tuples and add them as a
    # supervised PointwiseConstraint.  The intent is to anchor the network
    # on a sparse subset of the FEM reference while still enforcing the
    # Navier-Stokes residuals everywhere else.  See
    # src/Meshing/build_hybrid_dataset.py for the dataset format and units.
    hybrid_npz = OmegaConf.select(cfg, "custom.hybrid_dataset", default=None)
    if hybrid_npz is not None:
        hybrid_lambda = _cfg_float("custom.hybrid_lambda", 1.0)
        hybrid_bs = int(OmegaConf.select(cfg, "custom.hybrid_batch_size",
                                         default=2000))
        hybrid_supervise_p = bool(OmegaConf.select(
            cfg, "custom.hybrid_supervise_pressure", default=True))
        _h_path = to_absolute_path(str(hybrid_npz))
        _h = np.load(_h_path)
        _h_meta = {k: _h[k] for k in _h.files if k.startswith("meta_")}
        invar_h = {
            "x": _h["x"].astype(np.float32),
            "y": _h["y"].astype(np.float32),
            "z": _h["z"].astype(np.float32),
            "t": _h["t"].astype(np.float32),
        }
        outvar_h = {
            "u": _h["u"].astype(np.float32),
            "v": _h["v"].astype(np.float32),
            "w": _h["w"].astype(np.float32),
        }
        if hybrid_supervise_p:
            outvar_h["p"] = _h["p"].astype(np.float32)
        lam_h = {k: np.full_like(v, hybrid_lambda) for k, v in outvar_h.items()}
        n_h = invar_h["x"].shape[0]
        bs_eff = min(hybrid_bs, n_h)
        hybrid_constraint = PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=invar_h,
            outvar=outvar_h,
            batch_size=bs_eff,
            lambda_weighting=lam_h,
        )
        domain.add_constraint(hybrid_constraint, "hybrid_supervised")
        print(
            f"[aneurysm-pulsatile-hybrid] loaded {_h_path} "
            f"(N={n_h}, batch_size={bs_eff}, lambda={hybrid_lambda}, "
            f"supervise_p={hybrid_supervise_p}) meta={_h_meta}"
        )

    # ---- monitors -----------------------------------------------------------
    # PULSATILE: monitors at a single time slice (t = t_mon).  This gives a
    # cheap TensorBoard signal during training; full OSI is done offline.
    t_mon = float(OmegaConf.select(cfg, "custom.monitor_time", default=0.0))

    _inlet_sample = inlet_mesh.sample_boundary(16)
    n_inlet_mon = np.asarray(_inlet_sample["x"]).reshape(-1, 1).shape[0]
    pressure_invar = {
        "x": np.asarray(_inlet_sample["x"]).reshape(-1, 1),
        "y": np.asarray(_inlet_sample["y"]).reshape(-1, 1),
        "z": np.asarray(_inlet_sample["z"]).reshape(-1, 1),
        "t": np.full((n_inlet_mon, 1), t_mon, dtype=np.float32),
    }
    pressure_monitor = PointwiseMonitor(
        pressure_invar,
        output_names=["p"],
        metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
        nodes=nodes,
    )
    domain.add_monitor(pressure_monitor)

    eps_fd = _cfg_float("custom.wss_fd_eps", 1e-3)
    _wall_sample = noslip_mesh.sample_boundary(2000)
    n_wall_mon = np.asarray(_wall_sample["x"]).reshape(-1, 1).shape[0]
    wss_invar = {
        "x": np.asarray(_wall_sample["x"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_x"]).reshape(-1, 1),
        "y": np.asarray(_wall_sample["y"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_y"]).reshape(-1, 1),
        "z": np.asarray(_wall_sample["z"]).reshape(-1, 1)
             - eps_fd * np.asarray(_wall_sample["normal_z"]).reshape(-1, 1),
        "t": np.full((n_wall_mon, 1), t_mon, dtype=np.float32),
    }

    def _wss_mean_fd(var):
        u_, v_, w_ = var["u"], var["v"], var["w"]
        vel_mag = torch.sqrt(u_ ** 2 + v_ ** 2 + w_ ** 2)
        return torch.mean(nu * vel_mag / eps_fd)

    wss_monitor = PointwiseMonitor(
        wss_invar,
        output_names=["u", "v", "w"],
        metrics={"wss_mean": _wss_mean_fd},
        nodes=nodes,
    )
    domain.add_monitor(wss_monitor, "wss_monitor")

    # VTK volume snapshots are produced offline by compute_osi.py
    # (use --vtk_snapshots to request specific time slices).
    # Keeping inferencers out of the training loop avoids evaluating
    # ~64^3 grid points at every rec_results_freq checkpoint.

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()


# UPSTREAM PATCH (required for IntegralBoundaryConstraint with parameterization
# under numpy >= 2.0):
#   in $GLOBALSCRATCH/physicsnemo-py312/lib/python3.12/site-packages/
#       physicsnemo/sym/domain/constraint/continuous.py
#   replace at lines 708 and 758:
#     -    sp.Symbol(key): float(value)
#     +    sp.Symbol(key): float(np.asarray(value).reshape(-1)[0])
#   numpy 2.x refuses float() on non-0-D arrays; Parameterization.sample(1)
#   returns shape (1,1).

