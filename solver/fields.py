"""
solver/fields.py
================
Velocity and pressure field arrays for the collocated FVM solver.

Theory reference: theory/01_what_problem_are_we_solving.md
                  theory/08_lid_driven_cavity_setup.md

All variables (u, v, p) are stored at cell centres — this is the
"collocated" arrangement. See Chapter 4 for why this causes the
checkerboard problem and how Rhie-Chow fixes it.

Variable naming:
  u[i,j]  : x-velocity at cell (i,j)   [m/s]
  v[i,j]  : y-velocity at cell (i,j)   [m/s]
  p[i,j]  : pressure at cell (i,j)     [Pa]

  au_P_arr[i,j] : central coefficient a_P from u-momentum equation
  av_P_arr[i,j] : central coefficient a_P from v-momentum equation
  p_prime[i,j]  : pressure correction field p'
  bP[i,j]       : mass imbalance (continuity residual)
"""

import numpy as np
from solver.grid import Grid


class Fields:
    """
    Container for all field arrays.

    Initialised to zero. Lid boundary condition is applied after creation
    by calling solver.boundary_conditions.apply_velocity_bcs().

    Parameters
    ----------
    grid : Grid
        The computational grid.
    """

    def __init__(self, grid: Grid):
        nx, ny = grid.nx, grid.ny

        # Primary solution variables (cell-centred, collocated)
        self.u = np.zeros((nx, ny))   # x-velocity [m/s]
        self.v = np.zeros((nx, ny))   # y-velocity [m/s]
        self.p = np.zeros((nx, ny))   # pressure   [Pa]

        # Predicted velocity fields (u*, v*) from momentum solve
        # These satisfy momentum but NOT continuity
        self.u_star = np.zeros((nx, ny))
        self.v_star = np.zeros((nx, ny))

        # Central coefficients from the discretised momentum equations
        # a_P for u-momentum:  a_P u_P = a_E u_E + a_W u_W + a_N u_N + a_S u_S + b
        # Stored because they are re-used in Rhie-Chow and velocity correction
        self.au_P_arr = np.ones((nx, ny))   # initialised to 1 to avoid /0
        self.av_P_arr = np.ones((nx, ny))

        # Pressure correction field p'
        # Satisfies: a_P' p'_P = a_E' p'_E + ... - bP
        self.p_prime = np.zeros((nx, ny))

        # Mass imbalance (RHS of pressure-correction equation)
        # bP[i,j] = F_e - F_w + F_n - F_s  (net outward mass flux)
        # At convergence: bP → 0 everywhere
        self.bP = np.zeros((nx, ny))

    def reset_per_iteration(self):
        """
        Reset p_prime and bP at the start of each SIMPLE iteration.
        u_star and v_star will be overwritten by the momentum solver,
        so they don't need explicit resetting here.
        """
        self.p_prime[:] = 0.0
        self.bP[:]      = 0.0
