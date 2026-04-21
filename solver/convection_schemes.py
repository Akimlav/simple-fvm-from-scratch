"""
solver/convection_schemes.py
============================

Convection discretisation schemes for FVM momentum equations.

Provides:
  - First-order upwind (UDS)
  - Central differencing (CDS) for convection
  - Second-order upwind (SOU, deferred correction)

All return:
  - neighbour coefficients (aE, aW, aN, aS)
  - explicit correction term to RHS (b_corr; zero for UDS and CDS)

Switch schemes via `convection_coeffs_u` / `convection_coeffs_v`, or set
`convection_scheme` in run_simulation.py ("uds", "cds", or "sou").

CDS uses φ_face = average of adjacent nodes; it is not diagonally dominant at
high cell Peclet number and may oscillate or diverge — use fine grids or UDS/SOU
for strongly convective cases.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# Names accepted by convection_coeffs_* and run_simple(..., convection_scheme=...)
ConvectionSchemeName = Literal["uds", "cds", "sou"]

SCHEME_UDS = "uds"
SCHEME_CDS = "cds"
SCHEME_SOU = "sou"

_SCHEME_LABELS: dict[str, str] = {
    SCHEME_UDS: "first-order upwind (UDS)",
    SCHEME_CDS: "central differencing (CDS)",
    SCHEME_SOU: "second-order upwind (SOU, deferred correction)",
}


def convection_scheme_label(scheme: str) -> str:
    """Human-readable description for logging."""
    if scheme not in _SCHEME_LABELS:
        raise ValueError(
            f"Unknown convection scheme {scheme!r}; "
            f"use {SCHEME_UDS!r}, {SCHEME_CDS!r}, or {SCHEME_SOU!r}"
        )
    return _SCHEME_LABELS[scheme]


def convection_coeffs_u(
    u: np.ndarray,
    i: int,
    j: int,
    F_E: float,
    F_W: float,
    F_N: float,
    F_S: float,
    D_E: float,
    D_W: float,
    D_N: float,
    D_S: float,
    scheme: ConvectionSchemeName,
) -> tuple[float, float, float, float, float]:
    """u-momentum convection: UDS, CDS, or SOU (deferred correction for SOU only)."""
    if scheme == SCHEME_UDS:
        return upwind_scheme(F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S)
    if scheme == SCHEME_CDS:
        return central_difference_scheme(F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S)
    return sou_scheme_u(
        u, i, j, F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S
    )


def convection_coeffs_v(
    v: np.ndarray,
    i: int,
    j: int,
    F_E: float,
    F_W: float,
    F_N: float,
    F_S: float,
    D_E: float,
    D_W: float,
    D_N: float,
    D_S: float,
    scheme: ConvectionSchemeName,
) -> tuple[float, float, float, float, float]:
    """v-momentum convection: UDS, CDS, or SOU (deferred correction for SOU only)."""
    if scheme == SCHEME_UDS:
        return upwind_scheme(F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S)
    if scheme == SCHEME_CDS:
        return central_difference_scheme(F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S)
    return sou_scheme_v(
        v, i, j, F_E, F_W, F_N, F_S, D_E, D_W, D_N, D_S
    )


# ============================================================
# FIRST-ORDER UPWIND (UDS)
# ============================================================

def upwind_scheme(F_E, F_W, F_N, F_S,
                  D_E, D_W, D_N, D_S):
    """
    Standard first-order upwind scheme.

    Returns:
        aE, aW, aN, aS, b_corr (always zero)
    """

    aE = D_E + max(-F_E, 0.0)
    aW = D_W + max( F_W, 0.0)
    aN = D_N + max(-F_N, 0.0)
    aS = D_S + max( F_S, 0.0)

    b_corr = 0.0

    return aE, aW, aN, aS, b_corr


# ============================================================
# CENTRAL DIFFERENCING (CDS)
# ============================================================


def central_difference_scheme(F_E, F_W, F_N, F_S,
                              D_E, D_W, D_N, D_S):
    """
    Linear (central) differencing for convection, combined with diffusion.

    Face values φ_e = (φ_P + φ_E)/2 etc. With the code's F and D definitions
    and the same central_coeff() for a_P, this gives:

      aE = D_E - 0.5*F_E,  aW = D_W + 0.5*F_W,
      aN = D_N - 0.5*F_N,  aS = D_S + 0.5*F_S

    Returns:
        aE, aW, aN, aS, b_corr (always zero)
    """
    aE = D_E - 0.5 * F_E
    aW = D_W + 0.5 * F_W
    aN = D_N - 0.5 * F_N
    aS = D_S + 0.5 * F_S
    return aE, aW, aN, aS, 0.0


# ============================================================
# SECOND-ORDER UPWIND (SOU, deferred correction)
# ============================================================

def sou_scheme_u(u, i, j,
                 F_E, F_W, F_N, F_S,
                 D_E, D_W, D_N, D_S):
    """
    Second-order upwind for u-momentum.

    Matrix uses upwind (stable). Deferred correction on all four faces for the
    convected quantity u. Near boundaries, incomplete stencils fall back locally
    to first order.
    """

    nx = u.shape[0]

    # --- base upwind coefficients ---
    aE = D_E + max(-F_E, 0.0)
    aW = D_W + max( F_W, 0.0)
    aN = D_N + max(-F_N, 0.0)
    aS = D_S + max( F_S, 0.0)

    b_corr = 0.0

    # -------------------------------
    # EAST face correction
    # -------------------------------
    if F_E > 0:
        phi_up = u[i, j]
        phi_ww = u[i - 1, j]
        phi_face = phi_up + 0.5 * (phi_up - phi_ww)
        phi_upwind = phi_up
    else:
        phi_up = u[i + 1, j]
        if i + 2 < nx:
            phi_ee = u[i + 2, j]
            phi_face = phi_up + 0.5 * (phi_up - phi_ee)
        else:
            phi_face = phi_up
        phi_upwind = phi_up

    b_corr += F_E * (phi_face - phi_upwind)

    # -------------------------------
    # WEST face correction
    # -------------------------------
    if F_W > 0:
        phi_up = u[i - 1, j]
        if i - 2 >= 0:
            phi_ww = u[i - 2, j]
            phi_face = phi_up + 0.5 * (phi_up - phi_ww)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = u[i, j]
        phi_e = u[i + 1, j]
        phi_face = phi_up + 0.5 * (phi_up - phi_e)
        phi_upwind = phi_up

    b_corr -= F_W * (phi_face - phi_upwind)

    # -------------------------------
    # NORTH / SOUTH faces (u convected in y)
    # -------------------------------
    ny = u.shape[1]
    if F_N > 0:
        phi_up = u[i, j]
        if j - 1 >= 0:
            phi_ss = u[i, j - 1]
            phi_face = phi_up + 0.5 * (phi_up - phi_ss)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = u[i, j + 1]
        if j + 2 < ny:
            phi_nn = u[i, j + 2]
            phi_face = phi_up + 0.5 * (phi_up - phi_nn)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    b_corr += F_N * (phi_face - phi_upwind)

    if F_S > 0:
        phi_up = u[i, j - 1]
        if j - 2 >= 0:
            phi_ss = u[i, j - 2]
            phi_face = phi_up + 0.5 * (phi_up - phi_ss)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = u[i, j]
        phi_n = u[i, j + 1]
        phi_face = phi_up + 0.5 * (phi_up - phi_n)
        phi_upwind = phi_up
    b_corr -= F_S * (phi_face - phi_upwind)

    return aE, aW, aN, aS, b_corr


def sou_scheme_v(v, i, j,
                 F_E, F_W, F_N, F_S,
                 D_E, D_W, D_N, D_S):
    """
    Second-order upwind for v-momentum (y-direction).
    Near x/y boundaries, incomplete stencils fall back to first order (no correction).
    """

    nx = v.shape[0]
    ny = v.shape[1]

    aE = D_E + max(-F_E, 0.0)
    aW = D_W + max( F_W, 0.0)
    aN = D_N + max(-F_N, 0.0)
    aS = D_S + max( F_S, 0.0)

    b_corr = 0.0

    # -------------------------------
    # EAST / WEST faces (v convected in x)
    # -------------------------------
    if F_E > 0:
        phi_up = v[i, j]
        if i - 1 >= 0:
            phi_ww = v[i - 1, j]
            phi_face = phi_up + 0.5 * (phi_up - phi_ww)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = v[i + 1, j]
        if i + 2 < nx:
            phi_ee = v[i + 2, j]
            phi_face = phi_up + 0.5 * (phi_up - phi_ee)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    b_corr += F_E * (phi_face - phi_upwind)

    if F_W > 0:
        phi_up = v[i - 1, j]
        if i - 2 >= 0:
            phi_ww = v[i - 2, j]
            phi_face = phi_up + 0.5 * (phi_up - phi_ww)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = v[i, j]
        phi_e = v[i + 1, j]
        phi_face = phi_up + 0.5 * (phi_up - phi_e)
        phi_upwind = phi_up
    b_corr -= F_W * (phi_face - phi_upwind)

    # -------------------------------
    # NORTH face
    # -------------------------------
    if F_N > 0:
        phi_up = v[i, j]
        if j - 1 >= 0:
            phi_ss = v[i, j - 1]
            phi_face = phi_up + 0.5 * (phi_up - phi_ss)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = v[i, j + 1]
        if j + 2 < ny:
            phi_nn = v[i, j + 2]
            phi_face = phi_up + 0.5 * (phi_up - phi_nn)
        else:
            phi_face = phi_up
        phi_upwind = phi_up

    b_corr += F_N * (phi_face - phi_upwind)

    # -------------------------------
    # SOUTH face
    # -------------------------------
    if F_S > 0:
        phi_up = v[i, j - 1]
        if j - 2 >= 0:
            phi_ss = v[i, j - 2]
            phi_face = phi_up + 0.5 * (phi_up - phi_ss)
        else:
            phi_face = phi_up
        phi_upwind = phi_up
    else:
        phi_up = v[i, j]
        phi_n = v[i, j + 1]
        phi_face = phi_up + 0.5 * (phi_up - phi_n)
        phi_upwind = phi_up

    b_corr -= F_S * (phi_face - phi_upwind)

    return aE, aW, aN, aS, b_corr