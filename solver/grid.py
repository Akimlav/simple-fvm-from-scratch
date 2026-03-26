"""
solver/grid.py
==============
Uniform Cartesian grid for the 2D lid-driven cavity.

Theory reference: theory/02_finite_volume_discretization.md
                  theory/08_lid_driven_cavity_setup.md

GRID ARRANGEMENT
-----------------
This solver uses a node-based collocated grid: u, v, p are all stored at
the same grid points, and the grid points include the physical boundaries.

  x[i] = i * dx,   i = 0 ... nx-1
  y[j] = j * dy,   j = 0 ... ny-1

So x[0] = 0 and x[nx-1] = L — these are the actual boundary nodes.
Interior nodes (where Gauss-Seidel iterates) run i = 1..nx-2, j = 1..ny-2.

Convention:
  x → horizontal (i-index)
  y → vertical   (j-index)
  Cell (i,j): east face at i+0.5, west at i-0.5, north at j+0.5, south at j-0.5
"""

import numpy as np


class Grid:
    """
    Uniform Cartesian grid (node-based, boundaries included).

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction (includes both boundary nodes).
    ny : int
        Number of grid points in y-direction (includes both boundary nodes).
    L  : float
        Side length of the square domain [m].
    """

    def __init__(self, nx: int, ny: int, L: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.L  = L

        # Grid spacing between nodes
        self.dx = L / (nx - 1)
        self.dy = L / (ny - 1)

        # Node coordinates: x[0]=0, x[nx-1]=L (boundary nodes included)
        self.x = np.linspace(0.0, L, nx)
        self.y = np.linspace(0.0, L, ny)

        # 2D meshgrid (useful for plotting)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def __repr__(self):
        return (f"Grid(nx={self.nx}, ny={self.ny}, "
                f"L={self.L}, dx={self.dx:.4f}, dy={self.dy:.4f})")
