"""
solver/momentum.py
==================
Solve the u and v momentum equations to get predicted fields u*, v*.

Theory reference: theory/03_momentum_equations.md
                  theory/05_simple_algorithm.md (Step 1 and 2)

WHAT THIS FILE DOES
-------------------
Given:
  - Current pressure field p (may not be correct yet)
  - Current velocity fields u, v (from previous SIMPLE iteration)

Compute:
  - u* : x-velocity that satisfies x-momentum (but NOT continuity)
  - v* : y-velocity that satisfies y-momentum (but NOT continuity)

The predicted velocities are used to compute the mass imbalance in
rhie_chow.py, which drives the pressure correction in pressure.py.

DISCRETISED EQUATION (per cell, for u*)
-----------------------------------------
  a_P u*_P = a_E u*_E + a_W u*_W + a_N u*_N + a_S u*_S + b

where:
  b = - dy * (p[i+1,j] - p[i-1,j]) / 2    ← pressure gradient source term

Coefficients:
  a_E = D_E + max(-F_E, 0)    (upwind + diffusion, east neighbour)
  a_W = D_W + max( F_W, 0)    (west neighbour)
  a_N = D_N + max(-F_N, 0)    (north neighbour)
  a_S = D_S + max( F_S, 0)    (south neighbour)
  a_P = a_E + a_W + a_N + a_S + (F_E - F_W + F_N - F_S)

UNDER-RELAXATION
-----------------
  u*[i,j] = u[i,j] + α_u * (u_new - u[i,j])

Applied inside the Gauss-Seidel loop. Prevents large step changes.

IMPORTANT: au_P_arr and av_P_arr are stored in fields and used later by:
  1. rhie_chow.py: needs 1/a_P to compute Rhie-Chow face velocity
  2. simple.py: needs a_P to compute velocity correction d_u = dy/a_P
"""

import numpy as np
from solver.grid import Grid
from solver.fields import Fields
from solver.discretization import (diffusion_coeffs, convective_mass_fluxes,
                                    neighbour_coeffs, central_coeff)
from solver.boundary_conditions import apply_velocity_bcs


def solve_u_star(fields: Fields, grid: Grid,
                 rho: float, mu: float,
                 urf_u: float, gs_sweeps: int,
                 U_lid: float) -> None:
    """
    Solve u-momentum equation → update fields.u_star and fields.au_P_arr.

    Parameters
    ----------
    fields    : Fields object (u, v, p read; u_star, au_P_arr written)
    grid      : Grid object
    rho       : density [kg/m³]
    mu        : dynamic viscosity [Pa·s]
    urf_u     : under-relaxation factor for u (0 < urf_u <= 1)
    gs_sweeps : number of Gauss-Seidel sweeps
    U_lid     : lid velocity (needed for BCs after each sweep)
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Diffusion coefficients (constant for uniform grid)
    # D = μ * face_area / cell_centre_distance
    D_E, D_W, D_N, D_S = diffusion_coeffs(grid, mu)

    # Initialise u_star from previous u
    # (Gauss-Seidel will iterate from this starting point)
    np.copyto(fields.u_star, fields.u)

    # Build coefficient arrays for Gauss-Seidel
    # We need these as 2D arrays to pass to the generic G-S solver
    aE_arr = np.zeros((nx, ny))
    aW_arr = np.zeros((nx, ny))
    aN_arr = np.zeros((nx, ny))
    aS_arr = np.zeros((nx, ny))
    aP_arr = np.ones((nx, ny))   # 1 on boundary (prevents /0)
    b_arr  = np.zeros((nx, ny))

    # Assemble coefficients for all interior cells
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # Convective mass fluxes through each face [kg/s per unit depth]
            # Face velocity = arithmetic mean of adjacent cell centres
            #   u_e = (u[i,j] + u[i+1,j]) / 2
            #   F_e = ρ * u_e * dy
            F_E, F_W, F_N, F_S = convective_mass_fluxes(
                fields.u, fields.v, i, j, rho, dx, dy)

            # Neighbour coefficients (upwind differencing)
            #   a_E = D_E + max(-F_E, 0)   etc.
            aE, aW, aN, aS = neighbour_coeffs(F_E, F_W, F_N, F_S,
                                               D_E, D_W, D_N, D_S)

            # Central coefficient
            #   a_P = a_E + a_W + a_N + a_S + (F_E - F_W + F_N - F_S)
            aP = central_coeff(aE, aW, aN, aS, F_E, F_W, F_N, F_S)

            # Under-relaxation modifies a_P:
            #   a_P / α_u * u_P = Σ a_nb u_nb + b + (1-α_u)/α_u * a_P * u_old
            # Here we fold the under-relaxation into b and aP:
            aP_relax = aP / urf_u
            b_relax  = (1.0 - urf_u) / urf_u * aP * fields.u[i, j]

            # Pressure gradient source term:
            # -∂p/∂x * Volume ≈ -(p[i+1,j] - p[i-1,j]) / (2dx) * dx*dy
            #                  = -(p[i+1,j] - p[i-1,j]) * dy / 2
            # (the dx cancels with the 2dx denominator)
            b_pressure = -dy * (fields.p[i+1, j] - fields.p[i-1, j]) / 2.0

            aE_arr[i, j] = aE
            aW_arr[i, j] = aW
            aN_arr[i, j] = aN
            aS_arr[i, j] = aS
            aP_arr[i, j] = aP_relax
            b_arr[i, j]  = b_relax + b_pressure

            # Store UNRELAXED a_P for Rhie-Chow interpolation and velocity correction
            # (Rhie-Chow needs the physical a_P, not the relaxed one)
            fields.au_P_arr[i, j] = aP

    # Gauss-Seidel sweep
    for _ in range(gs_sweeps):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                fields.u_star[i, j] = (
                    aE_arr[i, j] * fields.u_star[i+1, j]
                  + aW_arr[i, j] * fields.u_star[i-1, j]
                  + aN_arr[i, j] * fields.u_star[i,  j+1]
                  + aS_arr[i, j] * fields.u_star[i,  j-1]
                  + b_arr[i, j]
                ) / aP_arr[i, j]

        # Enforce velocity BCs after each sweep
        apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)

    # Extrapolate a_P to boundary cells to avoid division by zero in Rhie-Chow
    _extrapolate_aP_to_boundaries(fields.au_P_arr)


def solve_v_star(fields: Fields, grid: Grid,
                 rho: float, mu: float,
                 urf_v: float, gs_sweeps: int,
                 U_lid: float) -> None:
    """
    Solve v-momentum equation → update fields.v_star and fields.av_P_arr.

    Identical structure to solve_u_star. Differences:
      - Solves for v instead of u
      - Pressure gradient in y: -(p[i,j+1] - p[i,j-1]) * dx / 2
      - Uses dx instead of dy in the pressure source term

    Parameters
    ----------
    Same as solve_u_star.
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    D_E, D_W, D_N, D_S = diffusion_coeffs(grid, mu)

    np.copyto(fields.v_star, fields.v)

    aE_arr = np.zeros((nx, ny))
    aW_arr = np.zeros((nx, ny))
    aN_arr = np.zeros((nx, ny))
    aS_arr = np.zeros((nx, ny))
    aP_arr = np.ones((nx, ny))
    b_arr  = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            F_E, F_W, F_N, F_S = convective_mass_fluxes(
                fields.u, fields.v, i, j, rho, dx, dy)

            aE, aW, aN, aS = neighbour_coeffs(F_E, F_W, F_N, F_S,
                                               D_E, D_W, D_N, D_S)

            aP = central_coeff(aE, aW, aN, aS, F_E, F_W, F_N, F_S)

            aP_relax = aP / urf_v
            b_relax  = (1.0 - urf_v) / urf_v * aP * fields.v[i, j]

            # y-direction pressure gradient:
            # -∂p/∂y * Volume ≈ -(p[i,j+1] - p[i,j-1]) * dx / 2
            b_pressure = -dx * (fields.p[i, j+1] - fields.p[i, j-1]) / 2.0

            aE_arr[i, j] = aE
            aW_arr[i, j] = aW
            aN_arr[i, j] = aN
            aS_arr[i, j] = aS
            aP_arr[i, j] = aP_relax
            b_arr[i, j]  = b_relax + b_pressure

            fields.av_P_arr[i, j] = aP

    for _ in range(gs_sweeps):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                fields.v_star[i, j] = (
                    aE_arr[i, j] * fields.v_star[i+1, j]
                  + aW_arr[i, j] * fields.v_star[i-1, j]
                  + aN_arr[i, j] * fields.v_star[i,  j+1]
                  + aS_arr[i, j] * fields.v_star[i,  j-1]
                  + b_arr[i, j]
                ) / aP_arr[i, j]

        apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)

    _extrapolate_aP_to_boundaries(fields.av_P_arr)


def _extrapolate_aP_to_boundaries(aP_arr: np.ndarray) -> None:
    """
    Copy nearest interior a_P value to boundary cells.

    Boundary cells are not solved by Gauss-Seidel, so au_P_arr[0,:] etc.
    would remain at their initialised value (1.0). Rhie-Chow computes
    1/a_P for faces adjacent to boundaries — we need physically reasonable
    values there to avoid large spurious corrections.

    First-order extrapolation (copy nearest interior neighbour):
      aP[0,:]  = aP[1,:]    (left ghost: use first interior column)
      aP[-1,:] = aP[-2,:]   etc.
    """
    aP_arr[0,  :] = aP_arr[1,  :]
    aP_arr[-1, :] = aP_arr[-2, :]
    aP_arr[:,  0] = aP_arr[:,  1]
    aP_arr[:, -1] = aP_arr[:, -2]
