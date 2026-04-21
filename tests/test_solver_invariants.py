"""
Regression / invariant checks for the FVM solver (no pytest required).

Run from repo root:
  python -m unittest tests.test_solver_invariants -v
"""

from __future__ import annotations

import unittest

import numpy as np

from solver.grid import Grid
from solver.fields import Fields
from solver.boundary_conditions import apply_velocity_bcs
from solver.discretization import diffusion_coeffs, neighbour_coeffs, central_coeff
from solver.convection_schemes import (
    SCHEME_UDS,
    central_difference_scheme,
    convection_scheme_label,
    upwind_scheme,
)
from solver.linear_solvers import gauss_seidel


class TestBoundaryConditions(unittest.TestCase):
    def test_lid_corners_keep_u_lid(self):
        u = np.zeros((7, 7))
        v = np.zeros((7, 7))
        apply_velocity_bcs(u, v, U_lid=1.0)
        self.assertEqual(u[0, -1], 1.0)
        self.assertEqual(u[-1, -1], 1.0)
        self.assertEqual(u[0, 0], 0.0)
        self.assertEqual(v[0, -1], 0.0)


class TestDiscretization(unittest.TestCase):
    def test_diffusion_uniform_square_cell(self):
        g = Grid(5, 5, L=1.0)
        mu = 0.01
        D_E, D_W, D_N, D_S = diffusion_coeffs(g, mu)
        expect = mu * g.dy / g.dx
        self.assertAlmostEqual(D_E, expect)
        self.assertAlmostEqual(D_W, expect)
        self.assertAlmostEqual(D_N, expect)
        self.assertAlmostEqual(D_S, expect)

    def test_upwind_matches_neighbour_coeffs(self):
        FE, FW, FN, FS = 0.3, -0.1, 0.2, -0.05
        DE, DW, DN, DS = 1.0, 1.0, 1.0, 1.0
        a1 = neighbour_coeffs(FE, FW, FN, FS, DE, DW, DN, DS)
        a2 = upwind_scheme(FE, FW, FN, FS, DE, DW, DN, DS)[:4]
        self.assertEqual(a1, a2)

    def test_central_coeff_zero_flux(self):
        aE = aW = aN = aS = 1.0
        aP = central_coeff(aE, aW, aN, aS, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(aP, 4.0 + 1e-20)


class TestConvectionSchemes(unittest.TestCase):
    def test_unknown_scheme_raises(self):
        with self.assertRaises(ValueError):
            convection_scheme_label("hybrid")

    def test_cds_zero_flux_matches_pure_diffusion(self):
        FE, FW, FN, FS = 0.0, 0.0, 0.0, 0.0
        DE, DW, DN, DS = 1.1, 1.2, 1.3, 1.4
        c = central_difference_scheme(FE, FW, FN, FS, DE, DW, DN, DS)
        u = upwind_scheme(FE, FW, FN, FS, DE, DW, DN, DS)
        self.assertEqual(c[:4], u[:4])
        self.assertEqual(c[4], 0.0)

    def test_cds_label_registered(self):
        self.assertIn("central", convection_scheme_label("cds").lower())


class TestGaussSeidel(unittest.TestCase):
    def test_solves_diagonal_system(self):
        n = 5
        phi = np.zeros((n, n))
        aP = np.ones((n, n)) * 2.0
        aE = aW = aN = aS = np.zeros((n, n))
        b = np.ones((n, n))
        gauss_seidel(phi, aP, aE, aW, aN, aS, b, n_sweeps=30)
        np.testing.assert_allclose(phi[1:-1, 1:-1], 0.5, rtol=1e-10, atol=1e-10)


class TestRhieChowBalance(unittest.TestCase):
    """Constant u*, v*, uniform p → zero divergence flux for interior."""

    def test_uniform_star_fields_zero_bP(self):
        from solver.rhie_chow import compute_face_velocity_rhie_chow

        g = Grid(11, 11, L=1.0)
        f = Fields(g)
        f.u_star[:, :] = 0.25
        f.v_star[:, :] = 0.1
        f.p[:, :] = 0.0
        f.au_P_arr[:, :] = 2.0
        f.av_P_arr[:, :] = 2.0
        compute_face_velocity_rhie_chow(f, g, rho=1.0)
        interior = f.bP[1:-1, 1:-1]
        self.assertLess(np.max(np.abs(interior)), 1e-14)


if __name__ == "__main__":
    unittest.main()
