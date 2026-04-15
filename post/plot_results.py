"""
post/plot_results.py
====================
Visualise the solver output and validate against Ghia et al. (1982).

Plots produced:
  1. Pressure contours with velocity quiver arrows
  2. Streamlines over the pressure field
  3. u-velocity centreline profile at x = 0.5 (vs Ghia CSV, Re=100 and Re=400)
  4. v-velocity centreline profile at y = 0.5 (vs Ghia CSV, Re=100 and Re=400)

Usage
-----
  Called from run_simulation.py after the SIMPLE loop converges.
  Can also be run standalone to reload saved .npy arrays:

    python post/plot_results.py

Ghia et al. (1982) benchmark tables are read from
  data/ghia_et_al_1982_tables.csv
(non-dimensional velocities, 129×129 uniform grid; see data/ghia_et_al_1982_tables.txt).
"""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Directory for saved figures (PNGs are committed to the repo)
RESULTS_DIR = "results"

# Default Ghia reference Reynolds numbers to plot from the CSV (columns Re_100, Re_400, …)
DEFAULT_GHIA_REYNOLDS: Tuple[int, ...] = (100, 400)

# CSV column name for each Reynolds number
_RE_COLUMN: Dict[int, str] = {
    100: "Re_100",
    400: "Re_400",
    1000: "Re_1000",
    3200: "Re_3200",
    5000: "Re_5000",
    7500: "Re_7500",
    10000: "Re_10000",
}

_SERIES_U = "u_vertical_centre"
_SERIES_V = "v_horizontal_centre"


def _ghia_csv_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "ghia_et_al_1982_tables.csv")
    )


def load_ghia_centreline_profiles(
    csv_path: Optional[str] = None,
    reynolds_numbers: Iterable[int] = DEFAULT_GHIA_REYNOLDS,
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load Ghia u- and v-centreline benchmark data for the requested Reynolds numbers.

    Returns
    -------
    dict[Re] -> {"u": (y, u_velocity), "v": (x, v_velocity)}
    """
    path = csv_path or _ghia_csv_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ghia benchmark CSV not found: {path}")

    res_list: List[int] = list(reynolds_numbers)
    for re in res_list:
        if re not in _RE_COLUMN:
            raise ValueError(f"No CSV column mapping for Re={re}; known: {sorted(_RE_COLUMN)}")

    col_u = [_RE_COLUMN[re] for re in res_list]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        missing = [c for c in col_u if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns {missing} in {path}")

        rows_u: List[dict] = []
        rows_v: List[dict] = []
        for row in reader:
            s = row.get("series", "").strip()
            if s == _SERIES_U:
                rows_u.append(row)
            elif s == _SERIES_V:
                rows_v.append(row)

    out: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for re, col in zip(res_list, col_u):
        y_u = np.array([float(r["y"]) for r in rows_u])
        u_v = np.array([float(r[col]) for r in rows_u])
        x_v = np.array([float(r["x"]) for r in rows_v])
        v_v = np.array([float(r[col]) for r in rows_v])
        out[re] = {"u": (y_u, u_v), "v": (x_v, v_v)}

    return out


def plot_pressure_and_velocity(u, v, p, grid_x, grid_y):
    """
    Plot pressure contours with velocity quiver arrows.

    Parameters
    ----------
    u, v    : 2D velocity arrays  [i,j] indexing
    p       : 2D pressure array
    grid_x  : 1D x-coordinate array
    grid_y  : 1D y-coordinate array
    """
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(X, Y, p, levels=30, cmap="RdBu_r", alpha=0.8)
    plt.colorbar(cf, ax=ax, label="Pressure [Pa]")
    ax.contour(X, Y, p, levels=15, colors="k", linewidths=0.5, alpha=0.5)

    # Quiver: skip every other point for readability
    skip = max(1, len(grid_x) // 20)
    ax.quiver(
        X[::skip, ::skip],
        Y[::skip, ::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        scale=10,
        color="k",
        alpha=0.8,
    )

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Pressure Field and Velocity Vectors", fontsize=13)
    ax.set_aspect("equal")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "pressure_velocity.png"), dpi=150)
    plt.show()


def plot_streamlines(u, v, p, grid_x, grid_y):
    """
    Plot streamlines overlaid on pressure contours.

    Parameters
    ----------
    u, v    : 2D velocity arrays  [i,j] indexing
    p       : 2D pressure array
    grid_x  : 1D x-coordinate array
    grid_y  : 1D y-coordinate array
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(grid_x, grid_y, p.T, levels=50, cmap="coolwarm", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Pressure [Pa]")

    # streamplot needs (x, y, u.T, v.T) because it expects [y,x] ordering
    ax.streamplot(grid_x, grid_y, u.T, v.T, density=1.5, color="k", linewidth=0.8)

    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Streamlines", fontsize=13)
    ax.set_aspect("equal")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "streamlines.png"), dpi=150)
    plt.show()


def _ghia_scatter_style(re: int) -> Mapping[str, object]:
    """Distinct marker/colour per Reynolds number for Ghia reference points."""
    if re == 100:
        return {"color": "0.15", "marker": "o", "s": 48, "edgecolors": "0.15", "linewidths": 0.6}
    if re == 400:
        return {
            "color": "tab:orange",
            "marker": "^",
            "s": 52,
            "edgecolors": "darkorange",
            "linewidths": 0.6,
        }
    return {"color": None, "marker": "D", "s": 44, "edgecolors": "k", "linewidths": 0.5}


def plot_u_centreline(
    u,
    grid_x,
    grid_y,
    ghia_profiles: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    ghia_reynolds: Iterable[int] = DEFAULT_GHIA_REYNOLDS,
    solver_re: Optional[float] = None,
):
    """
    Plot u-velocity profile at x = 0.5 (vertical centreline).
    Overlays Ghia et al. (1982) data from CSV for each requested Re.
    """
    if ghia_profiles is None:
        ghia_profiles = load_ghia_centreline_profiles(reynolds_numbers=ghia_reynolds)

    i_mid = int(np.argmin(np.abs(grid_x - 0.5)))
    u_profile = u[i_mid, :]  # u at x closest to 0.5, varying y

    fig, ax = plt.subplots(figsize=(7, 7))
    sol_label = "SIMPLE solver"
    if solver_re is not None:
        sol_label = f"SIMPLE solver (Re ≈ {solver_re:.0f})"
    ax.plot(u_profile, grid_y, "b-", linewidth=2, label=sol_label, zorder=3)

    for re in ghia_reynolds:
        if re not in ghia_profiles:
            continue
        y_g, u_g = ghia_profiles[re]["u"]
        kw = dict(_ghia_scatter_style(re))
        fc = kw.pop("color")
        ax.scatter(
            u_g,
            y_g,
            facecolors="none" if re == 100 else fc,
            zorder=5,
            label=f"Ghia et al. (1982) Re={re}",
            **kw,
        )

    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("u [m/s]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title(
        f'Centreline u-velocity  (x = {grid_x[i_mid]:.4f} ≈ 0.5)',
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "u_centreline.png"), dpi=150)
    plt.show()


def plot_v_centreline(
    v,
    grid_x,
    grid_y,
    ghia_profiles: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
    ghia_reynolds: Iterable[int] = DEFAULT_GHIA_REYNOLDS,
    solver_re: Optional[float] = None,
):
    """
    Plot v-velocity profile at y = 0.5 (horizontal centreline).
    Overlays Ghia et al. (1982) data from CSV for each requested Re.
    """
    if ghia_profiles is None:
        ghia_profiles = load_ghia_centreline_profiles(reynolds_numbers=ghia_reynolds)

    j_mid = int(np.argmin(np.abs(grid_y - 0.5)))
    v_profile = v[:, j_mid]  # v at y closest to 0.5, varying x

    fig, ax = plt.subplots(figsize=(7, 7))
    sol_label = "SIMPLE solver"
    if solver_re is not None:
        sol_label = f"SIMPLE solver (Re ≈ {solver_re:.0f})"
    ax.plot(grid_x, v_profile, "r-", linewidth=2, label=sol_label, zorder=3)

    for re in ghia_reynolds:
        if re not in ghia_profiles:
            continue
        x_g, v_g = ghia_profiles[re]["v"]
        kw = dict(_ghia_scatter_style(re))
        fc = kw.pop("color")
        ax.scatter(
            x_g,
            v_g,
            facecolors="none" if re == 100 else fc,
            zorder=5,
            label=f"Ghia et al. (1982) Re={re}",
            **kw,
        )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("v [m/s]", fontsize=12)
    ax.set_title(
        f'Centreline v-velocity  (y = {grid_y[j_mid]:.4f} ≈ 0.5)',
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "v_centreline.png"), dpi=150)
    plt.show()


def plot_convergence(residuals):
    """
    Plot continuity residual convergence history.

    Parameters
    ----------
    residuals : list of float, max|bP| at each SIMPLE iteration
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(residuals, "k-", linewidth=1.5)
    ax.set_xlabel("SIMPLE Iteration", fontsize=12)
    ax.set_ylabel("max|bP|  (continuity residual)", fontsize=12)
    ax.set_title("SIMPLE Convergence History", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "convergence.png"), dpi=150)
    plt.show()


def plot_all(
    u,
    v,
    p,
    grid_x,
    grid_y,
    residuals=None,
    re: Optional[float] = None,
    ghia_reynolds: Iterable[int] = DEFAULT_GHIA_REYNOLDS,
    ghia_csv_path: Optional[str] = None,
):
    """
    Convenience function: produce all standard plots.

    Parameters
    ----------
    u, v, p   : 2D arrays from solver
    grid_x, grid_y : 1D coordinate arrays
    residuals : list of convergence residuals (optional)
    re        : optional solver Reynolds number (for legend labels)
    ghia_reynolds : which Re columns from the Ghia CSV to overlay (default 100 and 400)
    ghia_csv_path : optional path to CSV; default is data/ghia_et_al_1982_tables.csv
    """
    ghia_profiles = load_ghia_centreline_profiles(
        csv_path=ghia_csv_path,
        reynolds_numbers=ghia_reynolds,
    )

    print("Plotting pressure and velocity field...")
    plot_pressure_and_velocity(u, v, p, grid_x, grid_y)

    print("Plotting streamlines...")
    plot_streamlines(u, v, p, grid_x, grid_y)

    print("Plotting u centreline vs Ghia...")
    plot_u_centreline(
        u,
        grid_x,
        grid_y,
        ghia_profiles=ghia_profiles,
        ghia_reynolds=ghia_reynolds,
        solver_re=re,
    )

    print("Plotting v centreline vs Ghia...")
    plot_v_centreline(
        v,
        grid_x,
        grid_y,
        ghia_profiles=ghia_profiles,
        ghia_reynolds=ghia_reynolds,
        solver_re=re,
    )

    if residuals is not None:
        print("Plotting convergence history...")
        plot_convergence(residuals)


# =============================================================================
# Standalone usage: load saved numpy arrays and plot
# =============================================================================
if __name__ == "__main__":
    output_dir = "results"
    try:
        u = np.load(os.path.join(output_dir, "u.npy"))
        v = np.load(os.path.join(output_dir, "v.npy"))
        p = np.load(os.path.join(output_dir, "p.npy"))
        grid_x = np.load(os.path.join(output_dir, "grid_x.npy"))
        grid_y = np.load(os.path.join(output_dir, "grid_y.npy"))
        try:
            residuals = list(np.load(os.path.join(output_dir, "residuals.npy")))
        except FileNotFoundError:
            residuals = None
        plot_all(u, v, p, grid_x, grid_y, residuals)
    except FileNotFoundError:
        print("No saved results found. Run run_simulation.py first.")
