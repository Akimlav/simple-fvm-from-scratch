"""
solver/simple.py
================
The SIMPLE algorithm outer loop.

Theory reference: theory/05_simple_algorithm.md

This file orchestrates all the other solver modules into the SIMPLE loop:

  LOOP:
    1. Solve u* from u-momentum  (momentum.py)
    2. Solve v* from v-momentum  (momentum.py)
    3. Compute mass imbalance bP (rhie_chow.py)
    4. Build p' equation coeffs  (pressure.py)
    5. Solve p'                  (pressure.py)
    6. Correct pressure p        (pressure.py)
    7. Correct velocities u, v   (this file)
    8. Check convergence

VELOCITY CORRECTION (Step 7)
-----------------------------
Derived from the momentum equation after subtracting the u* equation:

  a_P * (u - u*) = -dy * (p'_E - p'_P)
  =>  u[i,j] = u*[i,j] - (dy / a_P[i,j]) * (p'[i+1,j] - p'[i,j])

For v:
  v[i,j] = v*[i,j] - (dx / a_P[i,j]) * (p'[i,j+1] - p'[i,j])

Key: uses COMPACT gradient (adjacent cells), consistent with Rhie-Chow.
The coefficient dy/a_P is what we called d_u in theory/05.
"""

import numpy as np
from solver.grid import Grid
from solver.fields import Fields
from solver.momentum import solve_u_star, solve_v_star
from solver.rhie_chow import compute_face_velocity_rhie_chow
from solver.pressure import (build_pressure_correction_coeffs,
                              solve_pressure_correction,
                              correct_pressure)
from solver.boundary_conditions import apply_velocity_bcs


def correct_velocities(fields: Fields, grid: Grid, U_lid: float) -> None:
    """
    Apply velocity correction from pressure field p'.

    Equation (from theory/05_simple_algorithm.md, Step 6):
      u[i,j] = u*[i,j] - (dy/a_P[i,j]) * (p'[i+1,j] - p'[i,j])
      v[i,j] = v*[i,j] - (dx/a_P[i,j]) * (p'[i,j+1] - p'[i,j])

    The factor (dy/a_P) is d_u — velocity sensitivity to pressure change.
    The compact gradient (p'[i+1,j] - p'[i,j]) is CONSISTENT with:
      - The Rhie-Chow face velocity formula (same D_f factor)
      - The p' equation coefficients (same a_E' factor)

    Parameters
    ----------
    fields : Fields (reads u_star, v_star, p_prime, au_P_arr, av_P_arr;
                     updates u, v in-place)
    grid   : Grid
    U_lid  : float, lid velocity for BC re-application
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # x-velocity correction:
            #   d_u = dy / a_P   (velocity sensitivity to pressure)
            #   u = u* - d_u * (p'_E - p'_P)
            d_u = dy / fields.au_P_arr[i, j]
            fields.u[i, j] = (fields.u_star[i, j]
                               - d_u * (fields.p_prime[i+1, j]
                                        - fields.p_prime[i, j]))

            # y-velocity correction:
            #   d_v = dx / a_P
            #   v = v* - d_v * (p'_N - p'_P)
            d_v = dx / fields.av_P_arr[i, j]
            fields.v[i, j] = (fields.v_star[i, j]
                               - d_v * (fields.p_prime[i, j+1]
                                        - fields.p_prime[i, j]))

    # Re-enforce Dirichlet BCs on corrected velocity
    apply_velocity_bcs(fields.u, fields.v, U_lid)


def run_simple(fields: Fields, grid: Grid,
               rho: float, mu: float,
               urf_u: float, urf_v: float, urf_p: float,
               gs_mom: int, gs_p: int,
               n_iter: int, U_lid: float,
               print_every: int = 10,
               tol: float = 1e-5) -> list:
    """
    Run the SIMPLE algorithm for n_iter outer iterations.

    Parameters
    ----------
    fields     : Fields object (modified in-place)
    grid       : Grid object
    rho        : density [kg/m³]
    mu         : dynamic viscosity [Pa·s]
    urf_u      : under-relaxation factor for u-momentum (0 < urf_u <= 1)
    urf_v      : under-relaxation factor for v-momentum (0 < urf_v <= 1)
    urf_p      : under-relaxation factor for pressure (0 < urf_p <= 1)
    gs_mom     : Gauss-Seidel sweeps per momentum solve
    gs_p       : Gauss-Seidel sweeps for pressure-correction solve
    n_iter     : maximum number of SIMPLE outer iterations
    U_lid      : lid velocity [m/s]
    print_every: print residual every N iterations
    tol        : convergence tolerance on max continuity residual

    Returns
    -------
    residuals : list of float, max(|bP|) at each iteration
    """
    residuals = []

    for n in range(n_iter):

        # Reset p_prime and bP each iteration
        fields.reset_per_iteration()

        # ------------------------------------------------------------------
        # STEP 1: Solve u-momentum → u*
        # ------------------------------------------------------------------
        # Solves: a_P u* = Σ a_nb u*_nb - dy*(p[i+1,j]-p[i-1,j])/2
        # Stores au_P_arr for use in Steps 3, 4, 7
        solve_u_star(fields, grid, rho, mu, urf_u, gs_mom, U_lid)

        # ------------------------------------------------------------------
        # STEP 2: Solve v-momentum → v*
        # ------------------------------------------------------------------
        # Same structure as u-momentum, y-direction
        solve_v_star(fields, grid, rho, mu, urf_v, gs_mom, U_lid)

        # ------------------------------------------------------------------
        # STEP 3: Compute mass imbalance via Rhie-Chow face velocities
        # ------------------------------------------------------------------
        # bP[i,j] = F_e - F_w + F_n - F_s  (Rhie-Chow corrected fluxes)
        compute_face_velocity_rhie_chow(fields, grid, rho)

        # Track continuity residual
        max_residual = np.max(np.abs(fields.bP))
        residuals.append(max_residual)

        if n % print_every == 0 or n == n_iter - 1:
            print(f"  Iteration {n:4d}:  max|bP| = {max_residual:.6e}")

        # ------------------------------------------------------------------
        # STEP 4: Build pressure-correction equation coefficients
        # ------------------------------------------------------------------
        # a_E' = ρ dy² * 0.5 * (1/aP[i,j] + 1/aP[i+1,j])   etc.
        aE_p, aW_p, aN_p, aS_p, aP_p = build_pressure_correction_coeffs(
            fields, grid, rho)

        # ------------------------------------------------------------------
        # STEP 5: Solve pressure-correction equation → p'
        # ------------------------------------------------------------------
        # a_P' p' = a_E' p'_E + ... - bP   (Gauss-Seidel)
        solve_pressure_correction(fields, grid,
                                   aE_p, aW_p, aN_p, aS_p, aP_p,
                                   gs_p)

        # ------------------------------------------------------------------
        # STEP 6: Correct pressure
        # ------------------------------------------------------------------
        # p ← p + α_p * p'
        correct_pressure(fields, urf_p)

        # ------------------------------------------------------------------
        # STEP 7: Correct velocities
        # ------------------------------------------------------------------
        # u ← u* - (dy/a_P) * (p'_E - p'_P)
        # v ← v* - (dx/a_P) * (p'_N - p'_P)
        correct_velocities(fields, grid, U_lid)

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        if max_residual < tol:
            print(f"\n  Converged at iteration {n} with max|bP| = {max_residual:.2e}")
            break

    return residuals
