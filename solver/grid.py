"""
solver/grid.py
==============
Uniform Cartesian grid for the 2D lid-driven cavity.

Theory reference: theory/02_finite_volume_discretization.md
                  theory/08_lid_driven_cavity_setup.md

The grid stores:
  - Physical dimensions (L)
  - Number of cells (nx, ny)
  - Spacing (dx, dy)
  - Cell-centre coordinate arrays (x, y)

Convention:
  x → horizontal (i-index)
  y → vertical   (j-index)
  Cell (i,j): east face at i+0.5, west at i-0.5, north at j+0.5, south at j-0.5
"""

import numpy as np


class Grid:
    """
    Uniform Cartesian grid.

    Parameters
    ----------
    nx : int
        Number of grid points (cell centres) in x-direction.
    ny : int
        Number of grid points (cell centres) in y-direction.
    L  : float
        Side length of the square domain [m].
    """

    def __init__(self, nx: int, ny: int, L: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.L  = L

        # Grid spacing
        # For N points on [0, L]: spacing = L / (N-1)
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)

        # Cell-centre coordinates
        # x[i] = i * dx,  i = 0 ... nx-1
        self.x = np.linspace(0.0, L, nx)
        self.y = np.linspace(0.0, L, ny)

        # 2D meshgrid (useful for plotting)
        # X[i,j] = x-coordinate of cell (i,j)
        # Y[i,j] = y-coordinate of cell (i,j)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def __repr__(self):
        return (f"Grid(nx={self.nx}, ny={self.ny}, "
                f"L={self.L}, dx={self.dx:.4f}, dy={self.dy:.4f})")
