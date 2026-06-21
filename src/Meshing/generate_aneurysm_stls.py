"""Generate the 5 STL files required by NVIDIA PhysicsNeMo aneurysm tutorial.

Given either:
- an *open* vascular surface mesh (typically vessel wall with open ends), or
- a *closed* vascular surface mesh with inlet/outlet caps already present,

this script generates:

- <prefix>_inlet.stl     : inlet cap disk (open surface)
- <prefix>_outlet_XX.stl : one outlet cap disk per outlet (open surfaces)
- <prefix>_noslip.stl    : vessel wall surface (open surface)
- <prefix>_integral.stl  : internal cross-section disk at distance d=k*D from inlet (open surface)
- <prefix>_closed.stl    : watertight surface = wall + inlet cap + outlet caps

Python-only (PyVista/VTK), no VMTK. Open meshes are handled from boundary loops;
watertight meshes via normal-angle cap detection. STL normals are made consistent
but should be validated before use.

Usage: python generate_aneurysm_stls.py --input Selected_aneurysm.vtk --k 1.0

C:/Users/edoua/AppData/Local/Programs/Python/Python311/python.exe generate_aneurysm_stls.py --input Selected_aneurysm.vtk --k 1.0 --out_dir stl_files --prefix aneurysm --clean_tol 1e-6

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise RuntimeError(
        "PyVista is required. Install with: pip install pyvista"
    ) from exc


@dataclass(frozen=True)
class Loop:
    points: np.ndarray  # (N, 3) ordered points
    centroid: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) best-fit plane normal


@dataclass(frozen=True)
class CapPatch:
    region_id: int
    mesh: pv.PolyData
    area: float
    planarity_rel: float


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("Zero-length vector")
    return v / n


def _best_fit_plane_normal(points: np.ndarray) -> np.ndarray:
    """Return best-fit plane normal via SVD/PCA."""
    centered = points - points.mean(axis=0, keepdims=True)
    # For N x 3, Vt[-1] is the smallest-variance direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return _normalize(normal)


def _extract_boundary_edge_components(surface: pv.PolyData) -> list[pv.PolyData]:
    edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    if edges.n_cells == 0:
        return []

    conn = edges.connectivity()
    if "RegionId" not in conn.cell_data:
        raise RuntimeError("VTK connectivity did not produce RegionId cell_data")

    region_ids = np.unique(conn.cell_data["RegionId"])
    components: list[pv.PolyData] = []
    for rid in region_ids:
        part = conn.threshold([float(rid), float(rid)], scalars="RegionId")
        # threshold may return UnstructuredGrid; extract geometry to PolyData
        part = part.extract_geometry()
        components.append(part)

    return components


def _order_loop_points_from_segments(edges_component: pv.PolyData) -> np.ndarray:
    """Walk boundary segments into an ordered closed loop of points."""
    if edges_component.n_cells == 0:
        raise ValueError("Empty edge component")

    # Build adjacency using cell -> point ids
    adjacency: dict[int, list[int]] = {}
    for cell_id in range(edges_component.n_cells):
        cell = edges_component.get_cell(cell_id)
        ids = list(cell.point_ids)
        if len(ids) < 2:
            continue
        # Most often these are 2-point segments; if longer, connect consecutive points
        for a, b in zip(ids[:-1], ids[1:]):
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

    if not adjacency:
        raise RuntimeError("Could not build adjacency from boundary edges")

    # Heuristic: in a clean loop, every node has degree 2.
    degrees = np.array([len(v) for v in adjacency.values()], dtype=int)
    if degrees.min() < 2:
        raise RuntimeError(
            "Boundary edges do not form a single closed loop (degree < 2 found). "
            "Mesh may be noisy or have a cut/branch."
        )

    start = next(iter(adjacency.keys()))
    ordered = [start]
    prev = None
    current = start

    # Walk until we return to start.
    # Upper bound prevents infinite loops in malformed adjacency.
    max_steps = len(adjacency) + 5
    for _ in range(max_steps):
        nbrs = adjacency[current]
        # choose next neighbor not equal prev
        if prev is None:
            nxt = nbrs[0]
        else:
            if len(nbrs) == 1:
                raise RuntimeError("Open chain encountered while ordering loop")
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]

        if nxt == start:
            break

        ordered.append(nxt)
        prev, current = current, nxt
    else:
        raise RuntimeError("Failed to close the boundary loop (max_steps exceeded)")

    pts = np.asarray(edges_component.points)[np.array(ordered, dtype=int)]

    # Remove any accidental duplicates while preserving order
    # (VTK sometimes repeats the first point at the end; we don't include it here)
    dedup = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - dedup[-1]) > 1e-12:
            dedup.append(p)

    return np.asarray(dedup)


def _polygon_cap_from_ordered_loop(loop_points: np.ndarray) -> pv.PolyData:
    """Create a triangulated cap disk from an ordered boundary loop."""
    if loop_points.shape[0] < 3:
        raise ValueError("Loop must have >=3 points")

    # One polygon face referencing all points
    faces = np.concatenate(
        ([loop_points.shape[0]], np.arange(loop_points.shape[0], dtype=np.int64))
    )
    poly = pv.PolyData(loop_points, faces)
    return poly.triangulate().clean()


def _loop_cap_area(loop: Loop) -> float:
    return float(_polygon_cap_from_ordered_loop(loop.points).area)


def _loop_from_edges_component(edges_component: pv.PolyData) -> Loop:
    pts = _order_loop_points_from_segments(edges_component)
    centroid = pts.mean(axis=0)
    normal = _best_fit_plane_normal(pts)
    return Loop(points=pts, centroid=centroid, normal=normal)


def _mesh_boundary_edges_count(surface: pv.PolyData) -> int:
    edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return int(edges.n_cells)


def _mesh_nonmanifold_edges_count(surface: pv.PolyData) -> int:
    edges = surface.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    return int(edges.n_cells)


def _append_polydata(meshes: list[pv.PolyData]) -> pv.PolyData:
    """Append/merge multiple PolyData objects in a PyVista-version-compatible way."""
    if not meshes:
        raise ValueError("No meshes to append")
    out = meshes[0].copy(deep=True)
    for mesh in meshes[1:]:
        # merge_points helps avoid cracks when caps share boundary vertices
        out = out.merge(mesh, merge_points=True)
    return out


def _extract_cells_by_ids(surface: pv.PolyData, cell_ids: np.ndarray) -> pv.PolyData:
    if cell_ids.size == 0:
        return pv.PolyData()
    part = surface.extract_cells(cell_ids.astype(np.int64))
    return part.extract_surface().triangulate().clean()


def _extract_cells_by_mask(surface: pv.PolyData, cell_mask: np.ndarray) -> pv.PolyData:
    if cell_mask.shape[0] != surface.n_cells:
        raise ValueError("cell_mask size does not match number of cells")
    cell_ids = np.flatnonzero(cell_mask).astype(np.int64)
    return _extract_cells_by_ids(surface, cell_ids)


def _relative_planarity(surface: pv.PolyData) -> float:
    """Planarity proxy: normal-thickness / in-plane radius scale."""
    pts = np.asarray(surface.points)
    if pts.shape[0] < 3:
        return float("inf")

    centroid = pts.mean(axis=0)
    try:
        normal = _best_fit_plane_normal(pts)
    except ValueError:
        return float("inf")

    thickness = float(np.max(np.abs((pts - centroid) @ normal)))
    radii = np.linalg.norm(pts - centroid, axis=1)
    scale = float(np.percentile(radii, 90.0)) if radii.size else 0.0
    if scale < 1e-12:
        scale = float(np.linalg.norm(np.ptp(pts, axis=0)))
    return thickness / max(scale, 1e-12)


def _expand_cell_mask_by_point_neighbors(
    surface: pv.PolyData,
    initial_mask: np.ndarray,
    rings: int,
) -> np.ndarray:
    """Dilate a cell mask by adding cells sharing at least one point."""
    if initial_mask.shape[0] != surface.n_cells:
        raise ValueError("initial_mask size does not match number of cells")

    if rings <= 0:
        return initial_mask.copy()

    point_to_cells: list[list[int]] = [[] for _ in range(surface.n_points)]
    for cell_id in range(surface.n_cells):
        cell = surface.get_cell(cell_id)
        for point_id in cell.point_ids:
            point_to_cells[int(point_id)].append(cell_id)

    selected = set(np.flatnonzero(initial_mask).tolist())
    frontier = set(selected)

    for _ in range(rings):
        if not frontier:
            break

        expanded: set[int] = set()
        for cell_id in frontier:
            cell = surface.get_cell(cell_id)
            for point_id in cell.point_ids:
                expanded.update(point_to_cells[int(point_id)])

        expanded.difference_update(selected)
        if not expanded:
            break

        selected.update(expanded)
        frontier = expanded

    out = np.zeros(surface.n_cells, dtype=bool)
    if selected:
        out[np.fromiter(selected, dtype=np.int64)] = True
    return out


def _build_cell_edge_neighbors(surface: pv.PolyData) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(surface.n_cells)]
    edge_to_cells: dict[tuple[int, int], list[int]] = {}

    for cell_id in range(surface.n_cells):
        ids = list(surface.get_cell(cell_id).point_ids)
        if len(ids) < 3:
            continue

        ring = ids + [ids[0]]
        for a, b in zip(ring[:-1], ring[1:]):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            edge_to_cells.setdefault(key, []).append(cell_id)

    for cell_list in edge_to_cells.values():
        if len(cell_list) < 2:
            continue
        for i in range(len(cell_list)):
            for j in range(i + 1, len(cell_list)):
                c1 = cell_list[i]
                c2 = cell_list[j]
                neighbors[c1].add(c2)
                neighbors[c2].add(c1)

    return neighbors


def _normal_angle_regions(surface: pv.PolyData, cap_angle_deg: float) -> list[np.ndarray]:
    if not (0.0 < cap_angle_deg <= 180.0):
        raise ValueError("cap_angle_deg must be in (0, 180]")

    with_normals = surface.compute_normals(
        point_normals=False,
        cell_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
    )
    if "Normals" not in with_normals.cell_data:
        raise RuntimeError("Could not compute cell normals for angle-based segmentation")

    normals = np.asarray(with_normals.cell_data["Normals"], dtype=float)
    normal_norms = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_norms[:, None], 1e-12)

    neighbors = _build_cell_edge_neighbors(surface)
    cos_threshold = math.cos(math.radians(float(cap_angle_deg)))

    visited = np.zeros(surface.n_cells, dtype=bool)
    regions: list[np.ndarray] = []

    for seed in range(surface.n_cells):
        if visited[seed]:
            continue

        stack = [seed]
        visited[seed] = True
        region_cells: list[int] = []

        while stack:
            current = stack.pop()
            region_cells.append(current)
            n_current = normals[current]

            for nbr in neighbors[current]:
                if visited[nbr]:
                    continue
                if float(np.dot(n_current, normals[nbr])) >= cos_threshold:
                    visited[nbr] = True
                    stack.append(nbr)

        regions.append(np.asarray(region_cells, dtype=np.int64))

    return regions


def _largest_boundary_loop_from_patch(surface: pv.PolyData) -> Loop:
    edge_components = _extract_boundary_edge_components(surface)
    if not edge_components:
        raise RuntimeError("Cap patch does not contain boundary edges")

    loops = [_loop_from_edges_component(comp) for comp in edge_components]
    loop_areas = np.asarray([_loop_cap_area(loop) for loop in loops], dtype=float)
    return loops[int(np.argmax(loop_areas))]


def _recover_openings_from_closed_by_angle(
    surface: pv.PolyData,
    cap_angle_deg: float,
    planar_tol_rel: float,
    cap_border_rings: int,
) -> tuple[pv.PolyData, list[Loop], list[CapPatch]]:
    if planar_tol_rel <= 0.0:
        raise ValueError("planar_tol_rel must be > 0")
    if cap_border_rings < 0:
        raise ValueError("cap_border_rings must be >= 0")

    regions = _normal_angle_regions(surface, cap_angle_deg=cap_angle_deg)
    cap_patches: list[CapPatch] = []
    cap_cell_mask = np.zeros(surface.n_cells, dtype=bool)

    for region_id, region_cells in enumerate(regions):
        part = _extract_cells_by_ids(surface, region_cells)
        if part.n_cells == 0:
            continue

        boundaries = _extract_boundary_edge_components(part)
        if len(boundaries) != 1:
            continue

        planarity = _relative_planarity(part)
        if planarity > planar_tol_rel:
            continue

        area = float(part.area)
        if area <= 0.0:
            continue

        cap_cell_mask[region_cells] = True
        cap_patches.append(
            CapPatch(
                region_id=int(region_id),
                mesh=part,
                area=area,
                planarity_rel=planarity,
            )
        )

    if len(cap_patches) < 2:
        raise RuntimeError(
            "Input appears watertight (no boundary loops), but fewer than 2 planar cap patches "
            "were detected. Try tuning --cap_angle_deg and/or --planar_tol_rel."
        )

    if cap_border_rings > 0:
        cap_cell_mask = _expand_cell_mask_by_point_neighbors(surface, cap_cell_mask, cap_border_rings)

    wall_cell_mask = ~cap_cell_mask
    noslip = _extract_cells_by_mask(surface, wall_cell_mask)
    if noslip.n_cells == 0:
        raise RuntimeError(
            "Cap removal left an empty wall surface. Reduce --cap_border_rings or adjust cap detection."
        )

    loops = [_largest_boundary_loop_from_patch(cap.mesh) for cap in cap_patches]
    return noslip, loops, cap_patches


def _choose_inlet_outlet(loops: list[Loop]) -> tuple[Loop, list[Loop], np.ndarray]:
    if len(loops) < 2:
        raise ValueError(f"Expected at least 2 boundary loops, got {len(loops)}")

    areas = np.asarray([_loop_cap_area(loop) for loop in loops], dtype=float)
    inlet_idx = int(np.argmax(areas))
    inlet = loops[inlet_idx]

    outlet_indices = [idx for idx in range(len(loops)) if idx != inlet_idx]
    outlet_loops = [loops[idx] for idx in outlet_indices]
    outlet_areas = areas[outlet_indices]

    # Use area-weighted outlet centroid to define a stable inward direction.
    outlet_centroids = np.asarray([loop.centroid for loop in outlet_loops], dtype=float)
    if float(outlet_areas.sum()) > 0.0:
        target = np.average(outlet_centroids, axis=0, weights=outlet_areas)
    else:
        target = outlet_centroids.mean(axis=0)

    direction = target - inlet.centroid
    if float(np.linalg.norm(direction)) < 1e-12:
        distances = np.linalg.norm(outlet_centroids - inlet.centroid[None, :], axis=1)
        target = outlet_centroids[int(np.argmax(distances))]
        direction = target - inlet.centroid

    axis_inward = _normalize(direction)
    return inlet, outlet_loops, axis_inward


def _slice_to_largest_loop(closed: pv.PolyData, origin: np.ndarray, normal: np.ndarray) -> Loop:
    sliced = closed.slice(origin=origin, normal=normal)
    if sliced.n_cells == 0:
        raise RuntimeError("Slice produced no intersection; integral plane may be outside geometry")

    # slice() returns polylines (intersection curves), not a surface.
    # Split into connected curve components using connectivity.
    conn = sliced.connectivity()
    region_array = None
    if "RegionId" in conn.cell_data:
        region_array = conn.cell_data["RegionId"]
    elif "RegionId" in conn.point_data:
        region_array = conn.point_data["RegionId"]

    if region_array is None:
        # If connectivity did not label components, treat entire slice as one component.
        components = [conn.extract_geometry()]
    else:
        region_ids = np.unique(region_array)
        components = [
            conn.threshold([float(rid), float(rid)], scalars="RegionId").extract_geometry()
            for rid in region_ids
        ]

    if not components:
        raise RuntimeError("Could not extract integral loop from slice")

    # Pick the loop with largest triangulated area
    best_loop: Loop | None = None
    best_area = -1.0
    for comp in components:
        try:
            loop = _loop_from_edges_component(comp)
            cap = _polygon_cap_from_ordered_loop(loop.points)
            area = float(cap.area)
        except Exception:
            continue
        if area > best_area:
            best_area = area
            best_loop = loop

    if best_loop is None:
        raise RuntimeError("Failed to build a valid integral loop from sliced contours")

    return best_loop


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate inlet/outlet/noslip/integral/closed STLs from open or watertight vascular surfaces"
    )
    parser.add_argument("--input", "-i", required=True, type=str, help="Input surface mesh (.vtk/.vtp/.stl etc)")
    parser.add_argument("--out_dir", "-o", default="stl_files", type=str, help="Output directory")
    parser.add_argument("--prefix", default="aneurysm", type=str, help="Output filename prefix")
    parser.add_argument("--k", default=1.0, type=float, help="Integral plane distance factor d=k*D (D inferred from inlet cap area)")
    parser.add_argument("--clean_tol", default=None, type=float, help="Optional clean tolerance (PyVista clean(tolerance=...))")
    parser.add_argument(
        "--cap_angle_deg",
        default=45.0,
        type=float,
        help="Neighbor-triangle angle threshold (degrees) for watertight cap segmentation",
    )
    parser.add_argument(
        "--planar_tol_rel",
        default=0.02,
        type=float,
        help="Relative planarity tolerance for watertight cap detection",
    )
    parser.add_argument(
        "--cap_border_rings",
        default=0,
        type=int,
        help="Neighbor triangle rings to also remove around caps when building noslip",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surface = pv.read(str(in_path))
    surface = surface.extract_surface().triangulate()
    if args.clean_tol is None:
        surface = surface.clean()
    else:
        surface = surface.clean(tolerance=float(args.clean_tol))

    # Detect openings from boundaries (open mesh) or from angle-based cap patches (watertight mesh).
    edge_components = _extract_boundary_edge_components(surface)
    cap_patches: list[CapPatch] = []
    cap_source = "open-boundary-loops"

    if len(edge_components) >= 2:
        loops = [_loop_from_edges_component(ec) for ec in edge_components]
        # Noslip is the original wall surface for open input.
        noslip = surface
    elif len(edge_components) == 0:
        noslip, loops, cap_patches = _recover_openings_from_closed_by_angle(
            surface=surface,
            cap_angle_deg=float(args.cap_angle_deg),
            planar_tol_rel=float(args.planar_tol_rel),
            cap_border_rings=int(args.cap_border_rings),
        )
        cap_source = "watertight/normal-angle"
    else:
        raise RuntimeError(
            f"Expected >=2 boundary loops for an open mesh, got {len(edge_components)}. "
            "For watertight meshes, cap detection requires geometric segmentation to find at least 2 cap patches."
        )

    inlet_loop, outlet_loops, axis_inward = _choose_inlet_outlet(loops)

    inlet_cap = _polygon_cap_from_ordered_loop(inlet_loop.points)
    outlet_caps = [_polygon_cap_from_ordered_loop(loop.points) for loop in outlet_loops]

    # Closed mesh = wall + caps
    if cap_source == "open-boundary-loops":
        closed = _append_polydata([noslip, inlet_cap, *outlet_caps]).clean()
    else:
        # For watertight input we keep the original closed surface.
        closed = surface.copy(deep=True)

    try:
        closed = closed.compute_normals(auto_orient_normals=True, consistent_normals=True)
    except Exception:
        # Not fatal for STL generation
        pass

    # Integral plane at d=k*D from inlet, along axis toward outlet
    inlet_area = float(inlet_cap.area)
    inlet_radius = math.sqrt(max(inlet_area, 0.0) / math.pi)
    inlet_diameter = 2.0 * inlet_radius
    d = float(args.k) * inlet_diameter
    integral_origin = inlet_loop.centroid + axis_inward * d

    integral_loop = _slice_to_largest_loop(closed, origin=integral_origin, normal=axis_inward)
    integral_cap = _polygon_cap_from_ordered_loop(integral_loop.points)

    # Write STLs
    prefix = args.prefix
    inlet_path = out_dir / f"{prefix}_inlet.stl"
    outlet_paths = [
        out_dir / f"{prefix}_outlet_{idx:02d}.stl"
        for idx in range(1, len(outlet_caps) + 1)
    ]
    noslip_path = out_dir / f"{prefix}_noslip.stl"
    integral_path = out_dir / f"{prefix}_integral.stl"
    closed_path = out_dir / f"{prefix}_closed.stl"

    inlet_cap.save(str(inlet_path))
    for outlet_cap, outlet_path in zip(outlet_caps, outlet_paths):
        outlet_cap.save(str(outlet_path))
    noslip.save(str(noslip_path))
    integral_cap.save(str(integral_path))
    closed.save(str(closed_path))

    # Sanity checks
    closed_boundary_edges = _mesh_boundary_edges_count(closed)
    closed_nonmanifold = _mesh_nonmanifold_edges_count(closed)

    print("Wrote:")
    print(" -", inlet_path)
    for outlet_path in outlet_paths:
        print(" -", outlet_path)
    print(" -", noslip_path)
    print(" -", integral_path)
    print(" -", closed_path)
    print()
    print("Derived geometry:")
    print(" cap_source:", cap_source)
    print(" boundary_loops_found:", len(loops))
    print(" outlet_count:", len(outlet_loops))
    print(" inlet_area:", inlet_area)
    print(" inlet_diameter D:", inlet_diameter)
    print(" integral_distance d=k*D:", d)
    print(" boundary_edges(closed):", closed_boundary_edges)
    print(" nonmanifold_edges(closed):", closed_nonmanifold)
    if cap_patches:
        print(" cap_regions:", [cap.region_id for cap in cap_patches])
        print(" cap_planarity_rel:", [cap.planarity_rel for cap in cap_patches])

    if closed_boundary_edges != 0:
        print(
            "WARNING: closed mesh still has boundary edges; it may not be watertight. "
            "Try --clean_tol or inspect cap generation."
        )

    if closed_nonmanifold != 0:
        print(
            "WARNING: closed mesh has non-manifold edges; PhysicsNeMo sampling may be unstable. "
            "Consider additional mesh cleaning."
        )

    print(" found_outlets_final:", len(outlet_loops))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
