"""
compute_wss.py — Post-processing: exact WSS via automatic differentiation.

Loads a trained PhysicsNeMo flow_network checkpoint and computes wall shear
stress (WSS) at every point on the no-slip surface using torch.autograd.grad.

Usage (run from src/PhysicsNeMo/):
    python compute_wss.py \\
        --checkpoint outputs/aneurysm/checkpoint_fc4x256 \\
        --noslip     stl_files/0203_k1/aneurysm_noslip.stl \\
        --npoints    5000 \\
        --nu         0.00025 \\
        --out        wss_0203_k1.npz

The output .npz contains:
    x, y, z        : wall point coordinates
    normal_x/y/z   : outward wall normals
    wss            : WSS magnitude [same units as nu * inlet_vel / length_scale]
    tu, tv, tw     : full traction vector components (for WSS direction maps)
"""

import argparse
import os

import numpy as np
import torch

from physicsnemo.sym.geometry.tessellation import Tessellation
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key



def load_network(checkpoint_dir: str, device: torch.device) -> torch.nn.Module:
    """Load the flow_network weights from a PhysicsNeMo checkpoint directory."""
    pth = os.path.join(checkpoint_dir, "flow_network.0.pth")
    if not os.path.isfile(pth):
        raise FileNotFoundError(f"Checkpoint not found: {pth}")

    state = torch.load(pth, map_location=device)

    # infer arch size from weight shapes (keys like "layers.0._orig_mod.weight", shape [out, in])
    layer_sizes = []
    for k, v in state.items():
        if "weight" in k and v.ndim == 2:
            layer_sizes.append((k, v.shape))

    hidden_dim = layer_sizes[0][1][0]  # first weight: [hidden_dim, input_dim]
    nr_layers  = sum(1 for k, _ in layer_sizes if "layers" in k) - 1  # exclude output
    print(f"[wss] Detected architecture: layer_size={hidden_dim}, nr_layers={nr_layers}")

    arch = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        layer_size=hidden_dim,
        nr_layers=nr_layers,
    ).to(device)

    # PhysicsNeMo wraps the arch in a Node; the raw Arch is what we want.
    arch.load_state_dict(state, strict=False)
    arch.eval()
    return arch


def compute_wss_autograd(
    arch: torch.nn.Module,
    x_np: np.ndarray,
    y_np: np.ndarray,
    z_np: np.ndarray,
    nx_np: np.ndarray,
    ny_np: np.ndarray,
    nz_np: np.ndarray,
    nu: float,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
):
    """Compute WSS magnitude and traction components at wall points via autograd.

    Returns arrays of shape (N,): wss, tu, tv, tw.
    """
    n_pts = x_np.shape[0]
    wss_all = np.zeros(n_pts, dtype=np.float32)
    tu_all  = np.zeros(n_pts, dtype=np.float32)
    tv_all  = np.zeros(n_pts, dtype=np.float32)
    tw_all  = np.zeros(n_pts, dtype=np.float32)

    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        sl  = slice(start, end)

        x = torch.tensor(x_np[sl], dtype=torch.float32, device=device,
                         requires_grad=True).reshape(-1, 1)
        y = torch.tensor(y_np[sl], dtype=torch.float32, device=device,
                         requires_grad=True).reshape(-1, 1)
        z = torch.tensor(z_np[sl], dtype=torch.float32, device=device,
                         requires_grad=True).reshape(-1, 1)

        # Forward pass — the network is purely coordinate-based
        out = arch({"x": x, "y": y, "z": z})
        u, v, w = out["u"], out["v"], out["w"]

        ones = torch.ones_like(u)

        def grad_of(scalar, wrt):
            return torch.autograd.grad(
                scalar, wrt,
                grad_outputs=ones,
                create_graph=False,
                retain_graph=True,
            )[0]

        ux = grad_of(u, x); uy = grad_of(u, y); uz = grad_of(u, z)
        vx = grad_of(v, x); vy = grad_of(v, y); vz = grad_of(v, z)
        wx = grad_of(w, x); wy = grad_of(w, y); wz = grad_of(w, z)

        nx = torch.tensor(nx_np[sl], dtype=torch.float32, device=device).reshape(-1, 1)
        ny = torch.tensor(ny_np[sl], dtype=torch.float32, device=device).reshape(-1, 1)
        nz = torch.tensor(nz_np[sl], dtype=torch.float32, device=device).reshape(-1, 1)

        # traction: t = nu * (grad_u + grad_u^T) . n
        tu = nu * (2.0*ux*nx + (uy + vx)*ny + (uz + wx)*nz)
        tv = nu * ((vx + uy)*nx + 2.0*vy*ny + (vz + wy)*nz)
        tw = nu * ((wx + uz)*nx + (wy + vz)*ny + 2.0*wz*nz)

        t_dot_n = tu*nx + tv*ny + tw*nz
        wss_sq  = torch.clamp(tu**2 + tv**2 + tw**2 - t_dot_n**2, min=0.0)
        wss     = torch.sqrt(wss_sq)

        wss_all[sl] = wss.detach().cpu().numpy().reshape(-1)
        tu_all[sl]  = tu.detach().cpu().numpy().reshape(-1)
        tv_all[sl]  = tv.detach().cpu().numpy().reshape(-1)
        tw_all[sl]  = tw.detach().cpu().numpy().reshape(-1)

        print(f"  [{end}/{n_pts}] batch WSS mean = {wss_all[start:end].mean():.4g}", end="\r")

    print()
    return wss_all, tu_all, tv_all, tw_all



def main():
    parser = argparse.ArgumentParser(description="Compute PINN WSS via autograd")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory (contains flow_network.0.pth)")
    parser.add_argument("--noslip", required=True,
                        help="Path to aneurysm_noslip.stl")
    parser.add_argument("--npoints", type=int, default=5000,
                        help="Number of wall sample points (default: 5000)")
    parser.add_argument("--nu", type=float, default=0.00025,
                        help="Kinematic viscosity in normalised units (default: 0.00025)")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for autograd evaluation (default: 2048)")
    parser.add_argument("--out", default="wss_output.npz",
                        help="Output .npz file path (default: wss_output.npz)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (default: use CUDA if available)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    print(f"[wss] Device: {device}")

    # Load geometry
    print(f"[wss] Loading no-slip mesh: {args.noslip}")
    noslip_mesh = Tessellation.from_stl(args.noslip, airtight=False)
    sample = noslip_mesh.sample_boundary(args.npoints)

    x_np  = np.asarray(sample["x"]).reshape(-1).astype(np.float32)
    y_np  = np.asarray(sample["y"]).reshape(-1).astype(np.float32)
    z_np  = np.asarray(sample["z"]).reshape(-1).astype(np.float32)
    nx_np = np.asarray(sample["normal_x"]).reshape(-1).astype(np.float32)
    ny_np = np.asarray(sample["normal_y"]).reshape(-1).astype(np.float32)
    nz_np = np.asarray(sample["normal_z"]).reshape(-1).astype(np.float32)

    print(f"[wss] Sampled {x_np.shape[0]} wall points")

    # Load network
    arch = load_network(args.checkpoint, device)

    # Compute WSS
    print("[wss] Computing WSS via autograd...")
    wss, tu, tv, tw = compute_wss_autograd(
        arch, x_np, y_np, z_np, nx_np, ny_np, nz_np,
        nu=args.nu,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"[wss] WSS stats: mean={wss.mean():.4g}, min={wss.min():.4g}, "
          f"max={wss.max():.4g}")

    np.savez(
        args.out,
        x=x_np, y=y_np, z=z_np,
        normal_x=nx_np, normal_y=ny_np, normal_z=nz_np,
        wss=wss,
        tu=tu, tv=tv, tw=tw,
    )
    print(f"[wss] Saved to {args.out}")


if __name__ == "__main__":
    main()
