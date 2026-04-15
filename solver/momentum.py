"""
solver/momentum.py
==================
Solve the u and v momentum equations to get predicted fields u*, v*.

Theory reference: theory/03_momentum_equations.md
                  theory/05_simple_algorithm.md (Steps 1 and 2)

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

The Gauss-Seidel inner loop is handled by linear_solvers.gauss_seidel().

IMPORTANT: au_P_arr and av_P_arr are stored in fields and used later by:
  1. rhie_chow.py: needs 1/a_P to compute Rhie-Chow face velocity
  2. simple.py: needs a_P to compute velocity correction d_u = dy/a_P
"""

import numpy as np
from solver.grid import Grid
from solver.fields import Fields
from solver.discretization import (diffusion_coeffs, convective_mass_fluxes,
                                    central_coeff)
from solver.convection_schemes import (
    ConvectionSchemeName,
    convection_coeffs_u,
    convection_coeffs_v,
)
from solver.boundary_conditions import apply_velocity_bcs
from solver.linear_solvers import gauss_seidel


def solve_u_star(fields: Fields, grid: Grid,
                 rho: float, mu: float,
                 urf_u: float, gs_sweeps: int,
                 U_lid: float,
                 convection_scheme: ConvectionSchemeName = "sou") -> None:
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
    U_lid     : lid velocity (needed for BCs after solving)
    convection_scheme : 'uds', 'cds' (central differencing), or 'sou' (second-order upwind)
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    # Diffusion coefficients (constant for uniform grid)
    D_E, D_W, D_N, D_S = diffusion_coeffs(grid, mu)

    # Build coefficient arrays for Gauss-Seidel
    aE_arr = np.zeros((nx, ny))
    aW_arr = np.zeros((nx, ny))
    aN_arr = np.zeros((nx, ny))
    aS_arr = np.zeros((nx, ny))
    aP_arr = np.ones((nx, ny))   # 1 on boundary nodes (prevents /0)
    b_arr  = np.zeros((nx, ny))

    # Assemble coefficients for all interior nodes
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # Convective mass fluxes through each face [kg/s per unit depth]
            # F_e = ρ * u_e * dy,  u_e = (u[i,j] + u[i+1,j]) / 2
            F_E, F_W, F_N, F_S = convective_mass_fluxes(
                fields.u, fields.v, i, j, rho, dx, dy)

            # Neighbour coefficients (UDS or SOU + deferred correction)
            aE, aW, aN, aS, b_corr = convection_coeffs_u(
                fields.u, i, j,
                F_E, F_W, F_N, F_S,
                D_E, D_W, D_N, D_S,
                convection_scheme,
            )

            # Central coefficient
            # a_P = a_E + a_W + a_N + a_S + (F_E - F_W + F_N - F_S)
            aP = central_coeff(aE, aW, aN, aS, F_E, F_W, F_N, F_S)

            # Under-relaxation: modify a_P and add extra source term
            # a_P/α * u_P = Σ a_nb u_nb + b + (1-α)/α * a_P * u_old
            aP_relax = aP / urf_u
            b_relax  = (1.0 - urf_u) / urf_u * aP * fields.u[i, j]

            # Pressure gradient source term (x-direction):
            # -∂p/∂x * Volume ≈ -(p[i+1,j] - p[i-1,j]) / (2dx) * dx*dy
            #                  = -(p[i+1,j] - p[i-1,j]) * dy / 2
            b_pressure = -dy * (fields.p[i+1, j] - fields.p[i-1, j]) / 2.0

            aE_arr[i, j] = aE
            aW_arr[i, j] = aW
            aN_arr[i, j] = aN
            aS_arr[i, j] = aS
            aP_arr[i, j] = aP_relax
            b_arr[i, j]  = b_relax + b_pressure + b_corr

            # Store UNRELAXED a_P for Rhie-Chow and velocity correction
            fields.au_P_arr[i, j] = aP

    # Initialise u_star from previous u (Gauss-Seidel iterates from here)
    np.copyto(fields.u_star, fields.u)

    # Inner Gauss-Seidel loop (see theory/06, solver/linear_solvers.py)
    # After each sweep, re-apply velocity BCs to keep boundaries correct
    for _ in range(gs_sweeps):
        gauss_seidel(fields.u_star, aP_arr, aE_arr, aW_arr, aN_arr, aS_arr,
                     b_arr, n_sweeps=1)
        apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)

    # Extrapolate a_P to boundary nodes (needed for Rhie-Chow on boundary faces)
    _extrapolate_aP_to_boundaries(fields.au_P_arr)


def solve_v_star(fields: Fields, grid: Grid,
                 rho: float, mu: float,
                 urf_v: float, gs_sweeps: int,
                 U_lid: float,
                 convection_scheme: ConvectionSchemeName = "sou") -> None:
    """
    Solve v-momentum equation → update fields.v_star and fields.av_P_arr.

    Identical structure to solve_u_star. Differences:
      - Solves for v instead of u
      - Pressure gradient in y: -(p[i,j+1] - p[i,j-1]) * dx / 2

    convection_scheme : same as solve_u_star ('uds', 'cds', or 'sou').
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    D_E, D_W, D_N, D_S = diffusion_coeffs(grid, mu)

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

            aE, aW, aN, aS, b_corr = convection_coeffs_v(
                fields.v, i, j,
                F_E, F_W, F_N, F_S,
                D_E, D_W, D_N, D_S,
                convection_scheme,
            )

            aP = central_coeff(aE, aW, aN, aS, F_E, F_W, F_N, F_S)

            aP_relax = aP / urf_v
            b_relax  = (1.0 - urf_v) / urf_v * aP * fields.v[i, j]

            # Pressure gradient source term (y-direction):
            # -(p[i,j+1] - p[i,j-1]) * dx / 2
            b_pressure = -dx * (fields.p[i, j+1] - fields.p[i, j-1]) / 2.0

            aE_arr[i, j] = aE
            aW_arr[i, j] = aW
            aN_arr[i, j] = aN
            aS_arr[i, j] = aS
            aP_arr[i, j] = aP_relax
            b_arr[i, j] = b_relax + b_pressure + b_corr

            fields.av_P_arr[i, j] = aP

    np.copyto(fields.v_star, fields.v)

    for _ in range(gs_sweeps):
        gauss_seidel(fields.v_star, aP_arr, aE_arr, aW_arr, aN_arr, aS_arr,
                     b_arr, n_sweeps=1)
        apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)

    _extrapolate_aP_to_boundaries(fields.av_P_arr)


def _extrapolate_aP_to_boundaries(aP_arr: np.ndarray) -> None:
    """
    Copy nearest interior a_P value to boundary nodes.

    Boundary nodes are not solved by Gauss-Seidel, so au_P_arr[0,:] etc.
    would remain at their initialised value (1.0). Rhie-Chow computes
    1/a_P for faces adjacent to boundaries — we need physically reasonable
    values there to avoid large spurious corrections.
    """
    aP_arr[0,  :] = aP_arr[1,  :]
    aP_arr[-1, :] = aP_arr[-2, :]
    aP_arr[:,  0] = aP_arr[:,  1]
    aP_arr[:, -1] = aP_arr[:, -2]
