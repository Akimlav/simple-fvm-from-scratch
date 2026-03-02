"""
post/plot_results.py
====================
Visualise the solver output and validate against Ghia et al. (1982).

Plots produced:
  1. Pressure contours with velocity quiver arrows
  2. Streamlines over the pressure field
  3. u-velocity centreline profile at x = 0.5 (vs Ghia Re=100 data)
  4. v-velocity centreline profile at y = 0.5 (vs Ghia Re=100 data)

Usage
-----
  Called from run_simulation.py after the SIMPLE loop converges.
  Can also be run standalone to reload saved .npy arrays:

    python post/plot_results.py

Ghia et al. (1982) benchmark data for Re = 100:
  Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid method.
  Journal of Computational Physics, 48(3), 387-411.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# =============================================================================
# GHIA et al. (1982) BENCHMARK DATA — Re = 100
# =============================================================================
# u-velocity profile along the vertical centreline (x = 0.5)
# y_ghia: y-coordinate,  u_ghia: u-velocity
Y_GHIA_100 = np.array([1.0000, 0.9766, 0.8594, 0.7344, 0.6172,
                        0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0000])
U_GHIA_100 = np.array([1.0000,  0.8412,  0.3330,  0.0626, -0.0608,
                       -0.2109, -0.2058, -0.1566, -0.1034, -0.0643, 0.0000])

# v-velocity profile along the horizontal centreline (y = 0.5)
# x_ghia: x-coordinate,  v_ghia: v-velocity
X_GHIA_100 = np.array([0.0000, 0.0625, 0.0938, 0.1406, 0.5000,
                        0.7734, 0.9063, 1.0000])
V_GHIA_100 = np.array([0.0000, 0.0923, 0.1009, 0.1065, 0.0000,
                       -0.0853, -0.0982, 0.0000])


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
    X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(X, Y, p, levels=30, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(cf, ax=ax, label='Pressure [Pa]')
    cs = ax.contour(X, Y, p, levels=15, colors='k', linewidths=0.5, alpha=0.5)

    # Quiver: skip every other point for readability
    skip = max(1, len(grid_x) // 20)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              scale=10, color='k', alpha=0.8)

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Pressure Field and Velocity Vectors', fontsize=13)
    ax.set_aspect('equal')
    plt.tight_layout()
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
    cf = ax.contourf(grid_x, grid_y, p.T, levels=50, cmap='coolwarm', alpha=0.7)
    plt.colorbar(cf, ax=ax, label='Pressure [Pa]')

    # streamplot needs (x, y, u.T, v.T) because it expects [y,x] ordering
    ax.streamplot(grid_x, grid_y, u.T, v.T,
                  density=1.5, color='k', linewidth=0.8)

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Streamlines', fontsize=13)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_u_centreline(u, grid_x, grid_y):
    """
    Plot u-velocity profile at x = 0.5 (vertical centreline).
    Compares solver output to Ghia et al. (1982) Re=100 benchmark.
    """
    nx = len(grid_x)
    i_mid = nx // 2   # index closest to x = 0.5

    u_profile = u[i_mid, :]   # u at x ≈ 0.5, varying y

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(u_profile, grid_y, 'b-', linewidth=2, label='SIMPLE solver')
    ax.scatter(U_GHIA_100, Y_GHIA_100, color='k', s=50, zorder=5,
               label='Ghia et al. (1982) Re=100')

    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('u [m/s]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Centreline u-velocity  (x = 0.5)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_v_centreline(v, grid_x, grid_y):
    """
    Plot v-velocity profile at y = 0.5 (horizontal centreline).
    Compares solver output to Ghia et al. (1982) Re=100 benchmark.
    """
    ny = len(grid_y)
    j_mid = ny // 2   # index closest to y = 0.5

    v_profile = v[:, j_mid]   # v at y ≈ 0.5, varying x

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid_x, v_profile, 'r-', linewidth=2, label='SIMPLE solver')
    ax.scatter(X_GHIA_100, V_GHIA_100, color='k', s=50, zorder=5,
               label='Ghia et al. (1982) Re=100')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('v [m/s]', fontsize=12)
    ax.set_title('Centreline v-velocity  (y = 0.5)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence(residuals):
    """
    Plot continuity residual convergence history.

    Parameters
    ----------
    residuals : list of float, max|bP| at each SIMPLE iteration
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(residuals, 'k-', linewidth=1.5)
    ax.set_xlabel('SIMPLE Iteration', fontsize=12)
    ax.set_ylabel('max|bP|  (continuity residual)', fontsize=12)
    ax.set_title('SIMPLE Convergence History', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all(u, v, p, grid_x, grid_y, residuals=None):
    """
    Convenience function: produce all standard plots.

    Parameters
    ----------
    u, v, p   : 2D arrays from solver
    grid_x, grid_y : 1D coordinate arrays
    residuals : list of convergence residuals (optional)
    """
    print("Plotting pressure and velocity field...")
    plot_pressure_and_velocity(u, v, p, grid_x, grid_y)

    print("Plotting streamlines...")
    plot_streamlines(u, v, p, grid_x, grid_y)

    print("Plotting u centreline vs Ghia...")
    plot_u_centreline(u, grid_x, grid_y)

    print("Plotting v centreline vs Ghia...")
    plot_v_centreline(v, grid_x, grid_y)

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
