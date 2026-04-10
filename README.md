# simple-fvm-from-scratch

A complete, from-scratch 2D incompressible Navier–Stokes solver built to teach how SIMPLE and the Finite Volume Method actually work — equation by equation, line by line.

---

## Quick Start

```bash
pip install numpy matplotlib
python run_simulation.py
```

This runs a lid-driven cavity simulation at Re = 100 on a 41×41 grid. You'll see convergence output and four plots: pressure contours, streamlines, and centreline velocity comparisons against the Ghia et al. (1982) benchmark.

---

## What's Inside

| Component | Method |
|---|---|
| Spatial discretisation | Finite Volume Method, uniform collocated Cartesian grid |
| Pressure–velocity coupling | SIMPLE algorithm (Patankar, 1980) |
| Checkerboard fix | Rhie–Chow interpolation |
| Convection scheme | Upwind differencing |
| Linear solver | Gauss–Seidel iteration |
| Dependencies | `numpy` only (`matplotlib` for plots) |

---

## Repository Layout

```
simple-fvm-from-scratch/
│
├── run_simulation.py          ← entry point
│
├── solver/                    ← one file per concept
│   ├── grid.py                ← mesh geometry
│   ├── fields.py              ← field arrays (u, v, p, p', bP)
│   ├── discretization.py      ← FVM coefficient assembly
│   ├── momentum.py            ← u* and v* prediction
│   ├── pressure.py            ← pressure-correction equation
│   ├── rhie_chow.py           ← face velocity interpolation
│   ├── simple.py              ← outer SIMPLE loop
│   ├── linear_solvers.py      ← Gauss–Seidel solver
│   └── boundary_conditions.py ← velocity and pressure BCs
│
├── theory/                    ← step-by-step derivations (start here)
│   ├── 01_what_problem_are_we_solving.md
│   ├── 02_finite_volume_discretization.md
│   ├── 03_momentum_equations.md
│   ├── 04_pressure_velocity_coupling.md
│   ├── 05_simple_algorithm.md
│   ├── 06_gauss_seidel_solver.md
│   ├── 07_rhie_chow_interpolation.md
│   └── 08_lid_driven_cavity_setup.md
│
└── post/
    └── plot_results.py        ← visualisation + Ghia comparison
```

Every equation in the `theory/` notes has a direct counterpart in the `solver/` code. Each theory chapter ends with a table mapping concepts to files and functions.

---

## Changing Parameters

Edit `run_simulation.py`:

```python
# Reynolds number (change nu)
nu = 1e-2     # Re = 100  (default, fast convergence)
nu = 2.5e-3   # Re = 400  (increase n_iter to ~2000)
nu = 1e-3     # Re = 1000 (use 61×61 or 81×81 grid)

# Grid resolution
nx, ny = 41, 41   # default
nx, ny = 81, 81   # finer, needed for Re >= 1000

# Under-relaxation (reduce if diverging)
urf_u, urf_v = 0.5, 0.5   # momentum
urf_p = 0.2                # pressure

# Inner solver sweeps
gs_mom = 10   # Gauss–Seidel sweeps for momentum
gs_p   = 30   # Gauss–Seidel sweeps for pressure correction
```

---

## Expected Output

```
  Iteration    0:  max|bP| = 3.241000e+00
  Iteration   10:  max|bP| = 4.512000e-02
  Iteration  100:  max|bP| = 6.710000e-04
  Iteration  300:  max|bP| = 2.183000e-06

  Converged at iteration 312 with max|bP| = 9.87e-06
```

Early iterations may be non-monotonic — this is normal while the pressure field develops from zero.

After convergence, plots are saved to `results/` and displayed:

| Streamlines | Centreline u vs Ghia |
|---|---|
| ![Streamlines](results/streamlines.png) | ![u-centreline](results/u_centreline.png) |

| Pressure + velocity | Convergence history |
|---|---|
| ![Pressure](results/pressure_velocity.png) | ![Convergence](results/convergence.png) |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Residual diverges to infinity | Under-relaxation too aggressive | Reduce `urf_u`, `urf_v` to 0.3 |
| Very slow convergence | Too few pressure sweeps | Increase `gs_p` to 50 |
| Checkerboard in pressure plot | Bug in Rhie–Chow or `au_P_arr` | Check `au_P_arr` is not all-ones |
| Vortex not forming | Lid BC not enforced during solve | Verify `apply_velocity_bcs` is called after each G-S sweep |
| Centreline profiles shifted from Ghia | Grid too coarse | Increase to 61×61 or 81×81 |

---

## Validation

Benchmark: Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* Journal of Computational Physics, **48**(3), 387–411.

At Re = 100, the primary vortex centre should be near (0.617, 0.737). Centreline velocity profiles match Ghia within 2–5% on a 41×41 grid.

---

## References

- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Ferziger, J.H. & Peric, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.
- Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. *AIAA Journal*, **21**(11), 1525–1532.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *Journal of Computational Physics*, **48**(3), 387–411.
