#!/usr/bin/env python3
"""
run_simulation.py
=================
Entry point for the lid-driven cavity SIMPLE solver.

Implements the full workflow:
  1. Create uniform Cartesian grid
  2. Initialise field arrays (u, v, p = 0)
  3. Apply lid boundary condition
  4. Run SIMPLE iterations
  5. Print continuity residual per iteration
  6. Save results as numpy arrays
  7. Plot results and compare to Ghia et al. (1982)

Theory references:
  - theory/08_lid_driven_cavity_setup.md  (problem setup)
  - theory/05_simple_algorithm.md         (algorithm overview)
  - README.md                             (full description)

Run:
  python run_simulation.py

Expected runtime:
  ~2–5 minutes for 41×41 grid, 500 iterations (pure Python loops)
"""

import os
import time
import numpy as np

from solver.grid import Grid
from solver.fields import Fields
from solver.boundary_conditions import apply_velocity_bcs
from solver.simple import run_simple
from post.plot_results import plot_all

# =============================================================================
# PHYSICAL PARAMETERS
# =============================================================================
rho   = 1.0     # Density              [kg/m³]
nu    = 1e-2    # Kinematic viscosity  [m²/s]  → Re = U*L/nu = 1/0.01 = 100
# nu  = 2.5e-3  # Uncomment for Re = 400
# nu  = 1e-3    # Uncomment for Re = 1000 (use nx=ny=81 or larger)
mu    = nu * rho  # Dynamic viscosity  [Pa·s]

U_lid = 1.0     # Lid velocity         [m/s]
L     = 1.0     # Domain side length   [m]

Re = U_lid * L / nu
print(f"Reynolds number: Re = {Re:.0f}")

# =============================================================================
# GRID
# =============================================================================
# Increase nx/ny for higher accuracy (at the cost of speed).
# 41×41  : fast, good for Re=100 validation
# 61×61  : better for Re=400
# 81×81  : recommended for Re=1000
nx = 41
ny = 41

grid = Grid(nx=nx, ny=ny, L=L)
print(f"Grid: {grid}")

# =============================================================================
# FIELD INITIALISATION
# =============================================================================
fields = Fields(grid)

# Fluid at rest, zero pressure — SIMPLE will develop the flow
# apply_velocity_bcs sets the moving lid and no-slip walls
apply_velocity_bcs(fields.u, fields.v, U_lid)
apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)

print(f"Fields initialised. Lid BC: u[:, -1] = {U_lid}")

# =============================================================================
# SOLVER SETTINGS
# =============================================================================
# Under-relaxation factors
#   Momentum: 0.5 is a safe starting value for Re=100
#   Pressure: 0.2 is conservative; increase to 0.3 if converging well
urf_u = 0.5   # u-momentum under-relaxation
urf_v = 0.5   # v-momentum under-relaxation
urf_p = 0.2   # pressure under-relaxation

# Gauss-Seidel sweeps per SIMPLE iteration
gs_mom = 10   # sweeps for each momentum equation
gs_p   = 30   # sweeps for pressure-correction equation

# SIMPLE outer iterations
n_iter = 500

# Convergence tolerance on max continuity residual
tol = 1e-5

print(f"\nSolver settings:")
print(f"  Under-relaxation: α_u={urf_u}, α_v={urf_v}, α_p={urf_p}")
print(f"  Gauss-Seidel sweeps: momentum={gs_mom}, pressure={gs_p}")
print(f"  Max SIMPLE iterations: {n_iter}")
print(f"  Convergence tolerance: {tol:.0e}")

# =============================================================================
# RUN SIMPLE
# =============================================================================
print("\n" + "="*60)
print("Starting SIMPLE iterations...")
print("="*60)

t0 = time.time()

residuals = run_simple(
    fields   = fields,
    grid     = grid,
    rho      = rho,
    mu       = mu,
    urf_u    = urf_u,
    urf_v    = urf_v,
    urf_p    = urf_p,
    gs_mom   = gs_mom,
    gs_p     = gs_p,
    n_iter   = n_iter,
    U_lid    = U_lid,
    print_every = 25,
    tol      = tol,
)

elapsed = time.time() - t0
print(f"\nCompleted {len(residuals)} iterations in {elapsed:.1f} s")
print(f"Final max|bP| = {residuals[-1]:.3e}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "u.npy"),         fields.u)
np.save(os.path.join(output_dir, "v.npy"),         fields.v)
np.save(os.path.join(output_dir, "p.npy"),         fields.p)
np.save(os.path.join(output_dir, "grid_x.npy"),    grid.x)
np.save(os.path.join(output_dir, "grid_y.npy"),    grid.y)
np.save(os.path.join(output_dir, "residuals.npy"), np.array(residuals))

print(f"\nResults saved to '{output_dir}/':")
print(f"  u.npy, v.npy, p.npy  — velocity and pressure fields")
print(f"  grid_x.npy, grid_y.npy — coordinate arrays")
print(f"  residuals.npy — convergence history")

# =============================================================================
# DIAGNOSTICS
# =============================================================================
print("\n--- Flow diagnostics ---")
print(f"  u range: [{fields.u.min():.4f}, {fields.u.max():.4f}]  (lid = {U_lid})")
print(f"  v range: [{fields.v.min():.4f}, {fields.v.max():.4f}]")
print(f"  p range: [{fields.p.min():.4f}, {fields.p.max():.4f}]")

# Vortex centre: approximate location of max |ω|
# Vorticity ω = ∂v/∂x - ∂u/∂y
omega = (np.gradient(fields.v, grid.x, axis=0)
       - np.gradient(fields.u, grid.y, axis=1))
i_vort, j_vort = np.unravel_index(np.argmax(np.abs(omega)), omega.shape)
print(f"  Primary vortex centre ≈ ({grid.x[i_vort]:.3f}, {grid.y[j_vort]:.3f})")
print(f"  (Expected near (0.5, 0.7) for Re=100)")

# =============================================================================
# PLOT
# =============================================================================
print("\nGenerating plots...")
plot_all(fields.u, fields.v, fields.p, grid.x, grid.y, residuals)
