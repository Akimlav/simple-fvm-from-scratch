"""
solver/rhie_chow.py
===================
Rhie-Chow interpolation for face velocities and mass imbalance.

Theory reference: theory/07_rhie_chow_interpolation.md

WHY THIS IS NEEDED
------------------
On a collocated grid, the naive face velocity interpolation:
  u_e = (u_P + u_E) / 2
leads to a wide-stencil continuity equation that cannot detect
checkerboard pressure modes. See theory/04 and theory/07 for details.

THE RHIE-CHOW CORRECTION
------------------------
The corrected face velocity adds a pressure-smoothing term:

  u_e = avg(u*_P, u*_E)  -  D_f * (p_E - p_P)

where the correction coefficient:

  D_f = 0.5 * dy * (1/a_P[i,j] + 1/a_P[i+1,j])

is derived from the momentum equation:  u' = -(dy/a_P) * p'

The compact gradient (p_E - p_P) couples ADJACENT cells, detecting
the alternating pressure pattern that the momentum equation misses.

MASS IMBALANCE
--------------
Using Rhie-Chow face velocities, compute for each interior cell:

  bP[i,j] = ρ*(u_e - u_w)*dy  +  ρ*(v_n - v_s)*dx

This is the continuity residual. At convergence bP → 0.
bP also serves as the RHS of the pressure-correction equation.
"""

import numpy as np
from solver.grid import Grid
from solver.fields import Fields


def compute_face_velocity_rhie_chow(fields: Fields, grid: Grid,
                                     rho: float) -> None:
    """
    Compute Rhie-Chow face velocities and mass imbalance bP.

    Writes to fields.bP[i,j] = net outward mass flux for each interior cell.

    Rhie-Chow east face velocity (u_e):
    -----------------------------------
    Standard interpolation:
      u_bar_e = 0.5 * (u*[i,j] + u*[i+1,j])

    Pressure correction term:
      Δu_e = -0.5 * dy * (1/aP[i,j] + 1/aP[i+1,j]) * (p[i+1,j] - p[i,j])

    Combined:
      u_e = u_bar_e + Δu_e

    Mass flux through east face:
      F_e = ρ * u_e * dy

    Parameters
    ----------
    fields : Fields (reads u_star, v_star, p, au_P_arr, av_P_arr; writes bP)
    grid   : Grid
    rho    : float, density [kg/m³]
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    u_star   = fields.u_star
    v_star   = fields.v_star
    p        = fields.p
    au_P_arr = fields.au_P_arr
    av_P_arr = fields.av_P_arr

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # ------------------------------------------------------------------
            # EAST FACE  (between cell P=(i,j) and cell E=(i+1,j))
            # ------------------------------------------------------------------
            # Standard interpolation of u*
            u_bar_e = 0.5 * (u_star[i, j] + u_star[i+1, j])

            # Rhie-Chow correction:
            #   D_f_e = 0.5 * dy * (1/aP[i,j] + 1/aP[i+1,j])
            #   Δu_e  = -D_f_e * (p[i+1,j] - p[i,j])
            # The negative sign: higher pressure on the east pushes flow westward
            D_f_e = 0.5 * dy * (1.0 / au_P_arr[i, j] + 1.0 / au_P_arr[i+1, j])
            u_e   = u_bar_e - D_f_e * (p[i+1, j] - p[i, j])
            F_E   = rho * u_e * dy

            # ------------------------------------------------------------------
            # WEST FACE  (between cell W=(i-1,j) and cell P=(i,j))
            # ------------------------------------------------------------------
            u_bar_w = 0.5 * (u_star[i-1, j] + u_star[i, j])
            D_f_w   = 0.5 * dy * (1.0 / au_P_arr[i-1, j] + 1.0 / au_P_arr[i, j])
            u_w     = u_bar_w - D_f_w * (p[i, j] - p[i-1, j])
            F_W     = rho * u_w * dy

            # ------------------------------------------------------------------
            # NORTH FACE  (between cell P=(i,j) and cell N=(i,j+1))
            # ------------------------------------------------------------------
            v_bar_n = 0.5 * (v_star[i, j] + v_star[i, j+1])
            D_f_n   = 0.5 * dx * (1.0 / av_P_arr[i, j] + 1.0 / av_P_arr[i, j+1])
            v_n     = v_bar_n - D_f_n * (p[i, j+1] - p[i, j])
            F_N     = rho * v_n * dx

            # ------------------------------------------------------------------
            # SOUTH FACE  (between cell S=(i,j-1) and cell P=(i,j))
            # ------------------------------------------------------------------
            v_bar_s = 0.5 * (v_star[i, j-1] + v_star[i, j])
            D_f_s   = 0.5 * dx * (1.0 / av_P_arr[i, j-1] + 1.0 / av_P_arr[i, j])
            v_s     = v_bar_s - D_f_s * (p[i, j] - p[i, j-1])
            F_S     = rho * v_s * dx

            # ------------------------------------------------------------------
            # MASS IMBALANCE = continuity residual for cell (i,j)
            # bP = (F_e - F_w) + (F_n - F_s)
            # = net outward mass flux through all four faces
            # At convergence: bP → 0  (continuity satisfied)
            # ------------------------------------------------------------------
            fields.bP[i, j] = F_E - F_W + F_N - F_S
