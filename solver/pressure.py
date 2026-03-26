"""
solver/pressure.py
==================
Build and solve the pressure-correction (p') equation.

Theory reference: theory/05_simple_algorithm.md (Steps 4 and 5)

DERIVATION SUMMARY
------------------
After solving momentum, velocities u*, v* do NOT satisfy continuity.
We seek a correction p' such that if we apply:

  u = u*  -  (dy / a_P) * (p'_E - p'_P)    ← velocity correction
  v = v*  -  (dx / a_P) * (p'_N - p'_P)

the corrected velocities DO satisfy continuity.

Substituting corrected face velocities into continuity gives a Poisson eq:

  a_E' p'_E + a_W' p'_W + a_N' p'_N + a_S' p'_S - a_P' p'_P = bP

where the coefficients are:

  a_E'[i,j] = ρ dy² * 0.5 * (1/a_P[i,j]   + 1/a_P[i+1,j])
  a_W'[i,j] = ρ dy² * 0.5 * (1/a_P[i-1,j] + 1/a_P[i,j])
  a_N'[i,j] = ρ dx² * 0.5 * (1/a_P[i,j]   + 1/a_P[i,j+1])
  a_S'[i,j] = ρ dx² * 0.5 * (1/a_P[i,j-1] + 1/a_P[i,j])
  a_P'[i,j] = a_E' + a_W' + a_N' + a_S'

The Gauss-Seidel inner loop is handled by linear_solvers.gauss_seidel().

PRESSURE REFERENCE
------------------
The p' equation with only Neumann BCs has a non-unique solution (any
constant added to p' still satisfies the equations). We fix this by
subtracting the mean of p' after solving, keeping pressure centered at 0.
"""

import numpy as np
from solver.grid import Grid
from solver.fields import Fields
from solver.boundary_conditions import apply_pressure_neumann_bcs
from solver.linear_solvers import gauss_seidel


def build_pressure_correction_coeffs(fields: Fields, grid: Grid,
                                      rho: float) -> tuple:
    """
    Assemble coefficient arrays for the pressure-correction (p') equation.

    a_E' p'_E + a_W' p'_W + a_N' p'_N + a_S' p'_S - a_P' p'_P = bP

    Derived from substituting Rhie-Chow velocity corrections into continuity.
    See theory/05_simple_algorithm.md for full derivation.

    Parameters
    ----------
    fields : Fields (reads au_P_arr, av_P_arr)
    grid   : Grid
    rho    : float, density

    Returns
    -------
    aE_p, aW_p, aN_p, aS_p, aP_p : 2D numpy arrays, p' equation coefficients
    """
    nx, ny   = grid.nx, grid.ny
    dx, dy   = grid.dx, grid.dy
    au_P_arr = fields.au_P_arr
    av_P_arr = fields.av_P_arr

    aE_p = np.zeros((nx, ny))
    aW_p = np.zeros((nx, ny))
    aN_p = np.zeros((nx, ny))
    aS_p = np.zeros((nx, ny))
    aP_p = np.ones( (nx, ny))  # 1 on boundaries to avoid /0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # a_E' = ρ * dy² * 0.5 * (1/aP[i,j] + 1/aP[i+1,j])
            aE_p[i, j] = (rho * dy**2 * 0.5
                          * (1.0 / (au_P_arr[i,   j] + 1e-20)
                           + 1.0 / (au_P_arr[i+1, j] + 1e-20)))

            # a_W' = ρ * dy² * 0.5 * (1/aP[i-1,j] + 1/aP[i,j])
            aW_p[i, j] = (rho * dy**2 * 0.5
                          * (1.0 / (au_P_arr[i-1, j] + 1e-20)
                           + 1.0 / (au_P_arr[i,   j] + 1e-20)))

            # a_N' = ρ * dx² * 0.5 * (1/aP[i,j] + 1/aP[i,j+1])
            aN_p[i, j] = (rho * dx**2 * 0.5
                          * (1.0 / (av_P_arr[i, j]   + 1e-20)
                           + 1.0 / (av_P_arr[i, j+1] + 1e-20)))

            # a_S' = ρ * dx² * 0.5 * (1/aP[i,j-1] + 1/aP[i,j])
            aS_p[i, j] = (rho * dx**2 * 0.5
                          * (1.0 / (av_P_arr[i, j-1] + 1e-20)
                           + 1.0 / (av_P_arr[i, j]   + 1e-20)))

            aP_p[i, j] = aE_p[i, j] + aW_p[i, j] + aN_p[i, j] + aS_p[i, j] + 1e-20

    return aE_p, aW_p, aN_p, aS_p, aP_p


def solve_pressure_correction(fields: Fields, grid: Grid,
                               aE_p: np.ndarray, aW_p: np.ndarray,
                               aN_p: np.ndarray, aS_p: np.ndarray,
                               aP_p: np.ndarray,
                               gs_sweeps: int) -> None:
    """
    Solve the p' equation using Gauss-Seidel iteration.

    Equation: a_P' p'_P = a_E' p'_E + a_W' p'_W + a_N' p'_N + a_S' p'_S - bP

    Note the sign convention: bP is on the RHS with a MINUS sign.
    This is because bP is the mass IN excess — a positive mass imbalance
    requires a positive p' to slow the flow down and satisfy continuity.

    The p' equation has the same algebraic structure as the momentum equations
    (a_P φ_P = Σ a_nb φ_nb + source), so we use the same Gauss-Seidel solver.
    Here the "source" term is -bP.

    Boundary condition: Neumann dp'/dn = 0 on all walls, applied by copying
    the first interior value to the boundary after each sweep.

    Pressure reference: subtract mean(p') after solving to pin the level.

    Parameters
    ----------
    fields    : Fields (reads bP; writes p_prime in-place)
    grid      : Grid
    aE_p..aP_p: p' equation coefficients from build_pressure_correction_coeffs
    gs_sweeps : number of Gauss-Seidel sweeps
    """
    # The source term for p' is -bP (see sign convention above)
    b_p = -fields.bP

    # Gauss-Seidel solve (same function used for momentum equations)
    gauss_seidel(fields.p_prime, aP_p, aE_p, aW_p, aN_p, aS_p,
                 b_p, n_sweeps=gs_sweeps)

    # Neumann BCs for p': dp'/dn = 0 at all walls
    apply_pressure_neumann_bcs(fields.p_prime)

    # Fix pressure reference level: subtract mean so pressure stays near 0
    # Required because the p' equation with pure Neumann BCs is singular
    fields.p_prime -= np.mean(fields.p_prime)


def correct_pressure(fields: Fields, urf_p: float) -> None:
    """
    Update pressure with under-relaxed correction:
      p  ←  p + α_p * p'

    Parameters
    ----------
    fields : Fields (reads p_prime; updates p in-place)
    urf_p  : pressure under-relaxation factor (typically 0.1–0.3)
    """
    fields.p += urf_p * fields.p_prime
