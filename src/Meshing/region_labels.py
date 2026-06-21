"""region_labels.py - Map a pulsatile wall surface to dome/parent labels.

Each patient case ships with a SimVascular folder
``<pid>_H_CERE_CA_3D_RIGID_VTP/`` containing three polydata files:

* ``<pid>_H_CERE_CA_3D_RIGID_last.vtp`` - whole-wall reference surface
* ``<pid>_H_CERE_CA_3D_RIGID_dome.vtp`` - aneurysm sac subset
* ``<pid>_H_CERE_CA_3D_RIGID_parent.vtp`` - parent artery subset

The values stored in the *.vtp files come from a separate (steady) RIGID
simulation and are NOT used; only the per-point coordinates are read so
that the dome and parent regions can be transferred to the pulsatile
wall mesh by nearest-neighbour lookup. The pulsatile wall and the
``_last`` surface share the same geometry (max nearest-neighbour
distance < 0.02 cm in all six cases), so the mapping is unambiguous.

label_wall_regions(surface, pid) returns a per-point string array with
values in {"dome", "parent", "other"} aligned with ``surface.points``.
"""

from __future__ import annotations

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


VTP_ROOT_TEMPLATE = os.path.expanduser(
    "~/src/SimVascular/{pid}_H_CERE_CA/{pid}_H_CERE_CA_3D_RIGID_VTP"
)


def _vtp_path(pid: str, kind: str) -> str:
    root_env = os.environ.get("SV_CASES_ROOT")
    if root_env:
        folder = os.path.join(root_env,
                              f"{pid}_H_CERE_CA",
                              f"{pid}_H_CERE_CA_3D_RIGID_VTP")
    else:
        folder = VTP_ROOT_TEMPLATE.format(pid=pid)
    return os.path.join(folder, f"{pid}_H_CERE_CA_3D_RIGID_{kind}.vtp")


def label_points(points: np.ndarray, pid: str,
                  max_distance: float = 0.05) -> np.ndarray:
    """Like ``label_wall_regions`` but accepts an (N, 3) array directly.

    Points farther than ``max_distance`` from any ``_last`` node are
    silently labelled ``"other"`` instead of raising; the PINN samples
    its wall from an STL that is slightly offset from the SimVascular
    mesh, so a small fraction of points can fall just outside the
    reference cloud.
    """
    last_path = _vtp_path(pid, "last")
    dome_path = _vtp_path(pid, "dome")
    parent_path = _vtp_path(pid, "parent")
    for p in (last_path, dome_path, parent_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    last = pv.read(last_path)
    dome = pv.read(dome_path)
    parent = pv.read(parent_path)

    last_pts = np.asarray(last.points)
    tree = cKDTree(last_pts)
    _, dome_idx = tree.query(np.asarray(dome.points), k=1)
    _, par_idx = tree.query(np.asarray(parent.points), k=1)
    dome_set = set(dome_idx.tolist())
    par_set = set(par_idx.tolist()) - dome_set

    d, i = tree.query(np.asarray(points), k=1)
    labels = np.full(len(points), "other", dtype="<U6")
    for k, (dd, ii) in enumerate(zip(d.tolist(), i.tolist())):
        if dd > max_distance:
            continue
        if ii in dome_set:
            labels[k] = "dome"
        elif ii in par_set:
            labels[k] = "parent"
    return labels


def label_wall_regions(surface: pv.PolyData, pid: str,
                        max_wall_to_last: float = 0.05) -> np.ndarray:
    """Return dome/parent/other labels for surface.points via nearest-neighbour lookup into the VMR VTPs."""
    last_path = _vtp_path(pid, "last")
    dome_path = _vtp_path(pid, "dome")
    parent_path = _vtp_path(pid, "parent")
    for p in (last_path, dome_path, parent_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    last = pv.read(last_path)
    dome = pv.read(dome_path)
    parent = pv.read(parent_path)

    last_pts = np.asarray(last.points)
    tree = cKDTree(last_pts)

    # dome/parent VTPs are point-subsets of last (verified offline)
    _, dome_idx = tree.query(np.asarray(dome.points), k=1)
    _, par_idx = tree.query(np.asarray(parent.points), k=1)
    dome_set = set(dome_idx.tolist())
    par_set = set(par_idx.tolist())
    overlap = dome_set & par_set
    if overlap:
        # Resolve overlap deterministically: treat as dome
        par_set -= overlap

    wp = np.asarray(surface.points)
    d, i = tree.query(wp, k=1)
    if d.max() > max_wall_to_last:
        raise RuntimeError(
            f"[{pid}] wall->last max distance {d.max():.4f} cm exceeds "
            f"tolerance {max_wall_to_last} cm; check that VTPs and "
            f"pulsatile mesh come from the same geometry"
        )

    labels = np.full(len(wp), "other", dtype=object)
    for k, ii in enumerate(i.tolist()):
        if ii in dome_set:
            labels[k] = "dome"
        elif ii in par_set:
            labels[k] = "parent"
    return np.asarray(labels, dtype="<U6")
