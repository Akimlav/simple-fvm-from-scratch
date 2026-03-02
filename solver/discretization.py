"""
solver/discretization.py
=========================
FVM diffusion coefficients and convective flux utilities.

Theory reference: theory/02_finite_volume_discretization.md
                  theory/03_momentum_equations.md

DIFFUSION COEFFICIENTS
-----------------------
From integrating μ ∇²u over cell (i,j) and applying Gauss's theorem:

  ∫_V μ ∂²u/∂x² dV ≈ D_E (u_E - u_P) - D_W (u_P - u_W)

where the east-face diffusion coefficient is:

  D_E = μ * A_e / d_PE = μ * dy / dx

For a uniform grid dx = dy, so D_E = D_W = D_N = D_S = μ (constant).
But we keep them explicit to be clear about the physics.

CONVECTIVE MASS FLUXES
-----------------------
Mass flux through each face [kg/s per unit depth]:
  F_e = ρ * u_e * A_e = ρ * u_e * dy
  F_w = ρ * u_w * A_w = ρ * u_w * dy
  F_n = ρ * v_n * A_n = ρ * v_n * dx
  F_s = ρ * v_s * A_s = ρ * v_s * dx

Face velocities are linearly interpolated (arithmetic mean of two neighbours)
for use in the convection coefficients:
  u_e ≈ (u_P + u_E) / 2

Note: Rhie-Chow (solver/rhie_chow.py) uses a CORRECTED interpolation for
the continuity equation. The momentum equation uses this simple interpolation
only to compute which direction the flow is going (for upwind selection).
"""

import numpy as np
from solver.grid import Grid


def diffusion_coeffs(grid: Grid, mu: float) -> tuple:
    """
    Compute constant FVM diffusion coefficients for a uniform grid.

    From: D = μ * (face_area) / (distance_between_cell_centres)
    For uniform grid:
      East/West: area = dy, distance = dx  →  D_EW = μ * dy / dx
      North/South: area = dx, distance = dy  →  D_NS = μ * dx / dy

    Parameters
    ----------
    grid : Grid
    mu   : float, dynamic viscosity [Pa·s]

    Returns
    -------
    D_E, D_W, D_N, D_S : float (all equal on uniform grid, kept separate
                          for physical clarity and easy generalisation)
    """
    dx, dy = grid.dx, grid.dy

    # D_E = μ * dy / dx  (east face diffusion coefficient)
    D_E = mu * dy / dx

    # D_W = μ * dy / dx  (west face, same geometry)
    D_W = mu * dy / dx

    # D_N = μ * dx / dy  (north face)
    D_N = mu * dx / dy

    # D_S = μ * dx / dy  (south face)
    D_S = mu * dx / dy

    return D_E, D_W, D_N, D_S


def convective_mass_fluxes(u: np.ndarray, v: np.ndarray,
                            i: int, j: int,
                            rho: float, dx: float, dy: float) -> tuple:
    """
    Compute convective mass fluxes through the four faces of cell (i,j).

    Face velocity = arithmetic mean of two adjacent cell-centre values.
    (Linear interpolation on a uniform grid.)

    F_e = ρ * u_e * dy  where  u_e = (u[i,j] + u[i+1,j]) / 2

    Parameters
    ----------
    u, v : 2D velocity arrays
    i, j : cell indices
    rho  : density [kg/m³]
    dx, dy : grid spacing [m]

    Returns
    -------
    F_E, F_W, F_N, F_S : float, mass fluxes [kg/s per unit depth]
    """
    # East face: between (i,j) and (i+1,j), normal in x
    # u_e = average of cell-centre values on each side
    u_e = 0.5 * (u[i, j] + u[i+1, j])
    F_E = rho * u_e * dy

    # West face: between (i-1,j) and (i,j)
    u_w = 0.5 * (u[i-1, j] + u[i, j])
    F_W = rho * u_w * dy

    # North face: between (i,j) and (i,j+1), normal in y
    # v_n = average of cell-centre values above and below
    v_n = 0.5 * (v[i, j] + v[i, j+1])
    F_N = rho * v_n * dx

    # South face: between (i,j-1) and (i,j)
    v_s = 0.5 * (v[i, j-1] + v[i, j])
    F_S = rho * v_s * dx

    return F_E, F_W, F_N, F_S


def neighbour_coeffs(F_E: float, F_W: float, F_N: float, F_S: float,
                     D_E: float, D_W: float, D_N: float, D_S: float) -> tuple:
    """
    Compute upwind neighbour coefficients for the FVM momentum equation.

    From theory/03_momentum_equations.md:

      a_E = D_E + max(-F_E, 0)
      a_W = D_W + max( F_W, 0)
      a_N = D_N + max(-F_N, 0)
      a_S = D_S + max( F_S, 0)

    The max(·,0) terms implement the upwind differencing scheme:
      max(-F_E, 0) is nonzero only when F_E < 0
        (flow goes west through east face → upwind value is u_E)
      max(F_W, 0) is nonzero only when F_W > 0
        (flow goes east through west face → upwind value is u_W)

    Combined with diffusion coefficient D, this gives the hybrid/upwind
    interpolation for the convection-diffusion problem.

    Parameters
    ----------
    F_E, F_W, F_N, F_S : float, convective mass fluxes
    D_E, D_W, D_N, D_S : float, diffusion coefficients

    Returns
    -------
    aE, aW, aN, aS : float, neighbour coefficients
    """
    # aE: east neighbour coefficient
    # Diffusion always present; upwind adds max(-F_E,0) when flow is westward
    aE = D_E + max(-F_E, 0.0)

    # aW: west neighbour coefficient
    # Upwind adds max(F_W,0) when flow is eastward through west face
    aW = D_W + max(F_W, 0.0)

    # aN: north neighbour coefficient
    aN = D_N + max(-F_N, 0.0)

    # aS: south neighbour coefficient
    aS = D_S + max(F_S, 0.0)

    return aE, aW, aN, aS


def central_coeff(aE: float, aW: float, aN: float, aS: float,
                  F_E: float, F_W: float, F_N: float, F_S: float) -> float:
    """
    Compute the central coefficient a_P.

    a_P = a_E + a_W + a_N + a_S + (F_E - F_W + F_N - F_S)

    The extra term is the net outward mass flux from the cell.
    At convergence (continuity satisfied) it equals zero.
    Including it ensures diagonal dominance during iterations.

    A small value (1e-20) is added to prevent division by zero
    at the very first iteration when velocities are zero.
    """
    net_flux = F_E - F_W + F_N - F_S
    aP = aE + aW + aN + aS + net_flux + 1e-20
    return aP
