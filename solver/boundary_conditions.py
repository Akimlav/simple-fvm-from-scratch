"""
solver/boundary_conditions.py
==============================
Boundary condition application for the lid-driven cavity.

Theory reference: theory/08_lid_driven_cavity_setup.md

VELOCITY (Dirichlet conditions)
---------------------------------
  Top wall (j = -1, y = L): u = U_lid, v = 0   ← moving lid
  Bottom wall (j = 0):       u = 0,    v = 0   ← no-slip
  Left wall  (i = 0):        u = 0,    v = 0   ← no-slip
  Right wall (i = -1):       u = 0,    v = 0   ← no-slip

PRESSURE CORRECTION (Neumann conditions)
------------------------------------------
All walls are impermeable → dp'/dn = 0
Enforced by copying first interior value to boundary:
  p'[0,:]  = p'[1,:]    (left wall:   dp'/dx = 0)
  p'[-1,:] = p'[-2,:]   (right wall:  dp'/dx = 0)
  p'[:,0]  = p'[:,1]    (bottom wall: dp'/dy = 0)
  p'[:,-1] = p'[:,-2]   (top wall:    dp'/dy = 0)

WHY NEUMANN FOR PRESSURE?
  There is no prescribed pressure at any wall — all walls are solid.
  The condition dp'/dn = 0 means "no pressure-driven flux through the wall",
  consistent with zero normal velocity there.
"""

import numpy as np


def apply_velocity_bcs(u: np.ndarray, v: np.ndarray, U_lid: float) -> None:
    """
    Apply Dirichlet velocity BCs.

    Parameters
    ----------
    u, v    : 2D velocity arrays (modified in-place)
    U_lid   : lid velocity [m/s]
    """
    # Bottom wall (y = 0, j = 0): no-slip
    u[:, 0] = 0.0
    v[:, 0] = 0.0

    # Top wall (y = L, j = -1): MOVING LID
    u[:, -1] = U_lid   # horizontal velocity = U_lid
    v[:, -1] = 0.0     # no penetration normal to wall

    # Left wall (x = 0, i = 0): no-slip
    u[0, :] = 0.0
    v[0, :] = 0.0

    # Right wall (x = L, i = -1): no-slip
    u[-1, :] = 0.0
    v[-1, :] = 0.0


def apply_pressure_neumann_bcs(p_prime: np.ndarray) -> None:
    """
    Apply zero-gradient (Neumann) BCs for the pressure-correction field.

    Implemented by copying the first interior value to the boundary:
      dp'/dx = 0 at x=0  →  p'[0,:] = p'[1,:]
    etc.

    Parameters
    ----------
    p_prime : 2D pressure correction array (modified in-place)
    """
    p_prime[0,  :] = p_prime[1,  :]   # left wall
    p_prime[-1, :] = p_prime[-2, :]   # right wall
    p_prime[:,  0] = p_prime[:,  1]   # bottom wall
    p_prime[:, -1] = p_prime[:, -2]   # top wall
