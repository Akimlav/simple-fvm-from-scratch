"""
solver/linear_solvers.py
=========================
Gauss-Seidel iterative solver for the FVM linear system.

Theory reference: theory/06_gauss_seidel_solver.md

EQUATION BEING SOLVED
----------------------
Each cell (i,j) contributes one algebraic equation:

  a_P * φ_P = a_E * φ_E + a_W * φ_W + a_N * φ_N + a_S * φ_S + b

Rearranged to isolate φ_P:

  φ_P = (a_E φ_E + a_W φ_W + a_N φ_N + a_S φ_S + b) / a_P

Gauss-Seidel applies this update in-place: cells already updated in the
current sweep are used immediately (unlike Jacobi which uses all old values).

This function is used for BOTH the momentum equations (φ = u or v)
AND the pressure-correction equation (φ = p').
"""

import numpy as np


def gauss_seidel(phi: np.ndarray,
                 aP:  np.ndarray,
                 aE:  np.ndarray,
                 aW:  np.ndarray,
                 aN:  np.ndarray,
                 aS:  np.ndarray,
                 b:   np.ndarray,
                 n_sweeps: int) -> np.ndarray:
    """
    Solve the FVM linear system using Gauss-Seidel iteration.

    Update formula (derived in theory/06_gauss_seidel_solver.md):

      phi[i,j] = (aE[i,j]*phi[i+1,j] + aW[i,j]*phi[i-1,j]
               +  aN[i,j]*phi[i,j+1] + aS[i,j]*phi[i,j-1]
               +  b[i,j]) / aP[i,j]

    Sweep order: i = 1..nx-2, j = 1..ny-2  (interior cells only)
    Boundary cells (i=0, i=nx-1, j=0, j=ny-1) are set by BC functions,
    NOT updated here.

    Parameters
    ----------
    phi      : 2D array, solution field (modified IN-PLACE)
    aP       : 2D array, central coefficient (diagonal of the linear system)
    aE,aW,aN,aS : 2D arrays, off-diagonal (neighbour) coefficients
    b        : 2D array, source/RHS term
    n_sweeps : int, number of complete Gauss-Seidel sweeps

    Returns
    -------
    phi : 2D array (same array, modified in-place, also returned for convenience)

    Notes
    -----
    - No matrix is ever assembled. The stencil is applied cell by cell.
    - Convergence requires diagonal dominance: |a_P| >= sum|a_nb|
      This is guaranteed by how a_P is constructed in discretization.py.
    - phi is updated in-place. When computing phi[i,j], the values
      phi[i-1,j] and phi[i,j-1] are already from THIS sweep (new values),
      while phi[i+1,j] and phi[i,j+1] are still from the PREVIOUS sweep.
    """
    nx, ny = phi.shape

    for _ in range(n_sweeps):
        for i in range(1, nx - 1):      # interior cells only
            for j in range(1, ny - 1):  # boundary cells handled by BCs

                # Gauss-Seidel update:
                # Numerator = weighted sum of neighbours + source
                # Denominator = central coefficient
                phi[i, j] = (
                    aE[i, j] * phi[i+1, j]   # east neighbour  (old value)
                  + aW[i, j] * phi[i-1, j]   # west neighbour  (new value — already updated)
                  + aN[i, j] * phi[i,  j+1]  # north neighbour (old value)
                  + aS[i, j] * phi[i,  j-1]  # south neighbour (new value — already updated)
                  + b[i, j]                  # source term
                ) / aP[i, j]                 # central coefficient (divide to isolate phi_P)

    return phi


def compute_residual(phi: np.ndarray,
                     aP:  np.ndarray,
                     aE:  np.ndarray,
                     aW:  np.ndarray,
                     aN:  np.ndarray,
                     aS:  np.ndarray,
                     b:   np.ndarray) -> float:
    """
    Compute the maximum residual of the linear system.

    Residual at cell (i,j):
      r[i,j] = |a_P φ_P - a_E φ_E - a_W φ_W - a_N φ_N - a_S φ_S - b|

    Returns the maximum residual over all interior cells.
    A small residual means the linear system is nearly satisfied.

    Parameters
    ----------
    Same as gauss_seidel.

    Returns
    -------
    float : maximum residual
    """
    nx, ny = phi.shape
    max_res = 0.0

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            res = abs(
                aP[i, j] * phi[i,   j]
              - aE[i, j] * phi[i+1, j]
              - aW[i, j] * phi[i-1, j]
              - aN[i, j] * phi[i,   j+1]
              - aS[i, j] * phi[i,   j-1]
              - b[i, j]
            )
            if res > max_res:
                max_res = res

    return max_res
