#!/usr/bin/env python3
"""
run_re_grid_sweep.py
====================
Batch runner: lid-driven cavity at **Re = 100** and **Re = 400**, each on
**81×81**, **129×129**, and **256×256** node grids (six cases total).

For each case, writes fields and diagnostics under:

  results/sweep/Re<Re>_n<nx>/

and saves PNGs there (non-interactive Matplotlib backend).

Usage (from repo root):

  python run_re_grid_sweep.py

Optional:

  python run_re_grid_sweep.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from solver.grid import Grid
from solver.fields import Fields
from solver.boundary_conditions import apply_velocity_bcs
from solver.convection_schemes import SCHEME_SOU, convection_scheme_label, SCHEME_CDS
from solver.simple import run_simple

# -----------------------------------------------------------------------------
# Sweep specification
# -----------------------------------------------------------------------------
REYNOLDS_NUMBERS = (100, 400)
GRID_SIZES = (81, 129, 256)

RHO = 1.0
U_LID = 1.0
L = 1.0

# Solver settings (SOU is stable across Re and mesh sizes in this code base)
URF_U = 0.5
URF_V = 0.5
URF_P = 0.2
GS_MOM = 20
GS_P = 100
N_ITER = 3000
TOL = 1e-9
PRINT_EVERY = 200
CONVECTION_SCHEME = SCHEME_CDS

SWEEP_ROOT = os.path.join("results", "sweep")


def _case_dir(re: int, nx: int) -> str:
    return os.path.join(SWEEP_ROOT, f"Re{re}_n{nx}")


def _vortex_centre_psi(fields_u: np.ndarray, grid: Grid) -> tuple[float, float]:
    psi = np.zeros_like(fields_u)
    for j in range(1, grid.ny):
        psi[:, j] = psi[:, j - 1] + grid.dy * 0.5 * (
            fields_u[:, j - 1] + fields_u[:, j]
        )
    psi_int = psi[1:-1, 1:-1]
    i_in, j_in = np.unravel_index(np.argmin(psi_int), psi_int.shape)
    i_vc, j_vc = i_in + 1, j_in + 1
    return float(grid.x[i_vc]), float(grid.y[j_vc])


def run_one_case(
    re: int,
    nx: int,
    *,
    do_plots: bool,
) -> Dict[str, Any]:
    """Run SIMPLE for one (Re, nx) with ny = nx. Saves under results/sweep/Re*_n*."""
    ny = nx
    nu = U_LID * L / re
    mu = nu * RHO

    out_dir = _case_dir(re, nx)
    os.makedirs(out_dir, exist_ok=True)

    grid = Grid(nx=nx, ny=ny, L=L)
    fields = Fields(grid)
    apply_velocity_bcs(fields.u, fields.v, U_LID)
    apply_velocity_bcs(fields.u_star, fields.v_star, U_LID)

    t0 = time.time()
    residuals = run_simple(
        fields=fields,
        grid=grid,
        rho=RHO,
        mu=mu,
        urf_u=URF_U,
        urf_v=URF_V,
        urf_p=URF_P,
        gs_mom=GS_MOM,
        gs_p=GS_P,
        n_iter=N_ITER,
        U_lid=U_LID,
        print_every=PRINT_EVERY,
        tol=TOL,
        convection_scheme=CONVECTION_SCHEME,
    )
    elapsed = time.time() - t0

    np.save(os.path.join(out_dir, "u.npy"), fields.u)
    np.save(os.path.join(out_dir, "v.npy"), fields.v)
    np.save(os.path.join(out_dir, "p.npy"), fields.p)
    np.save(os.path.join(out_dir, "grid_x.npy"), grid.x)
    np.save(os.path.join(out_dir, "grid_y.npy"), grid.y)
    np.save(os.path.join(out_dir, "residuals.npy"), np.asarray(residuals))

    vx, vy = _vortex_centre_psi(fields.u, grid)
    meta: Dict[str, Any] = {
        "re": re,
        "nx": nx,
        "ny": ny,
        "nu": nu,
        "mu": mu,
        "seconds": elapsed,
        "n_iterations": len(residuals),
        "final_max_abs_bP": float(residuals[-1]),
        "convection_scheme": CONVECTION_SCHEME,
        "u_min": float(fields.u.min()),
        "u_max": float(fields.u.max()),
        "vortex_centre_xy": [vx, vy],
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if do_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import post.plot_results as pr

        prev_rd = pr.RESULTS_DIR
        try:
            pr.RESULTS_DIR = out_dir
            pr.plot_all(
                fields.u,
                fields.v,
                fields.p,
                grid.x,
                grid.y,
                residuals,
                re=float(re),
            )
        finally:
            pr.RESULTS_DIR = prev_rd
            plt.close("all")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Re × grid sweep for lid-driven cavity.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation (only .npy and meta.json).",
    )
    args = parser.parse_args()

    os.makedirs(SWEEP_ROOT, exist_ok=True)

    print("Sweep: Re ∈", REYNOLDS_NUMBERS, ", grids", GRID_SIZES)
    print(f"Convection: {convection_scheme_label(CONVECTION_SCHEME)}")
    print(f"Outputs: {os.path.abspath(SWEEP_ROOT)}/Re*_n*/")
    print("=" * 60)

    manifest: List[Dict[str, Any]] = []
    for re in REYNOLDS_NUMBERS:
        for nx in GRID_SIZES:
            label = f"Re={re}, n={nx}"
            print(f"\n>>> {label}")
            meta = run_one_case(re, nx, do_plots=not args.no_plots)
            manifest.append(meta)
            print(
                f"    done in {meta['seconds']:.1f} s, "
                f"{meta['n_iterations']} iters, "
                f"max|bP|={meta['final_max_abs_bP']:.3e}"
            )
            print(f"    → {meta['output_dir']}")

    manifest_path = os.path.join(SWEEP_ROOT, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\n" + "=" * 60)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
