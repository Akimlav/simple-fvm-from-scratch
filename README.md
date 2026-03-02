# simple-fvm-from-scratch

**A complete, from-scratch 2D incompressible Navier–Stokes solver** built to teach you how SIMPLE and the Finite Volume Method actually work — equation by equation, line by line.


---

## What This Is

This repository implements a **lid-driven cavity** flow solver using:

| Component | Method |
|---|---|
| Spatial discretisation | Finite Volume Method (FVM), uniform collocated Cartesian grid |
| Pressure–velocity coupling | SIMPLE algorithm (Patankar, 1980) |
| Checkerboard fix | Rhie–Chow interpolation |
| Convection scheme | Upwind differencing |
| Diffusion | Central differencing |
| Linear solver | Gauss–Seidel iteration |
| Dependencies | `numpy` only |

Every equation in the `theory/` notes has a direct, named counterpart in the `solver/` code. If you see `a_P u_P = a_E u_E + ...` in a markdown file, you will find `aP`, `aE`, etc. in the corresponding Python file.

---

## What is FVM?

The **Finite Volume Method** discretises the governing PDEs by integrating them over small control volumes (cells) rather than evaluating them at points. The integral form naturally conserves mass, momentum, and energy. Fluxes through shared faces are computed once and used by both neighbouring cells.

For a scalar φ satisfying a general transport equation:

```
∂/∂t(ρφ) + ∇·(ρuφ) = ∇·(Γ∇φ) + S
```

integrating over a cell volume V and applying Gauss's theorem:

```
∑_faces (ρ u_f φ_f A_f) = ∑_faces (Γ (∂φ/∂n)_f A_f) + S·V
```

Each face flux becomes a coefficient in the linear system.

---

## What is SIMPLE?

**SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations) solves the incompressible Navier–Stokes equations by decoupling pressure and velocity through a predictor–corrector loop:

1. **Guess** a pressure field `p`
2. **Solve** momentum equations → get predicted velocities `u*`, `v*` (which violate continuity)
3. **Derive** a pressure-correction equation from the continuity constraint
4. **Solve** for `p'` (pressure correction)
5. **Correct** pressure: `p ← p + αp · p'`
6. **Correct** velocities: `u ← u* − (dy/aP) · ∂p'/∂x`
7. **Repeat** until the mass imbalance is negligible

The key insight: the momentum equation tells us how to turn a pressure gradient into a velocity correction. SIMPLE exploits this to build the pressure equation.

---

## What is Rhie–Chow?

On a collocated grid (u, v, p all at cell centres), naive interpolation of velocities to faces lets a **checkerboard pressure field** satisfy continuity exactly — a purely numerical artefact. Rhie–Chow interpolation adds a compact pressure-gradient term to the face velocity that eliminates this decoupling without moving to a staggered grid.

---

## Repository Layout

```
simple-fvm-from-scratch/
│
├── README.md                  ← you are here
├── requirements.txt
├── run_simulation.py          ← entry point: runs SIMPLE, saves results
│
├── theory/                    ← step-by-step derivations
│   ├── 01_what_problem_are_we_solving.md
│   ├── 02_finite_volume_discretization.md
│   ├── 03_momentum_equations.md
│   ├── 04_pressure_velocity_coupling.md
│   ├── 05_simple_algorithm.md
│   ├── 06_gauss_seidel_solver.md
│   ├── 07_rhie_chow_interpolation.md
│   └── 08_lid_driven_cavity_setup.md
│
├── solver/                    ← one file per concept
│   ├── grid.py                ← mesh geometry
│   ├── fields.py              ← field initialisation
│   ├── discretization.py      ← FVM coefficient assembly
│   ├── linear_solvers.py      ← Gauss–Seidel
│   ├── momentum.py            ← u* and v* prediction
│   ├── pressure.py            ← pressure-correction equation
│   ├── rhie_chow.py           ← face velocity interpolation
│   ├── simple.py              ← outer SIMPLE loop
│   └── boundary_conditions.py ← all BCs in one place
│
└── post/
    └── plot_results.py        ← matplotlib visualisation + Ghia comparison
```

---

## How to Run

```bash
pip install numpy matplotlib
python run_simulation.py
```

Results are saved to `results/` as numpy `.npy` files and four plots are displayed:

1. Pressure contour + velocity quiver
2. Streamlines
3. Centreline u-velocity vs Ghia et al. (1982) benchmark
4. Centreline v-velocity vs Ghia et al. (1982) benchmark

---

## Expected Output

```
Iteration   1 | mass residual: 3.241e+00
Iteration   5 | mass residual: 8.123e-01
Iteration  20 | mass residual: 4.512e-02
Iteration 100 | mass residual: 6.710e-04
Iteration 500 | mass residual: 2.183e-06
```

The continuity residual should decrease monotonically. At Re = 100 with a 41×41 grid and 500 SIMPLE iterations, the centreline profiles match Ghia et al. within a few percent.

---

## Validation: Ghia et al. (1982)

The standard benchmark for this problem is:

> Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, **48**(3), 387–411.

They solved the lid-driven cavity at Re = 100, 400, 1000, 3200, 5000, 7500, 10000 on a 129×129 grid using a vorticity–streamfunction multigrid solver. Their tabulated centreline velocities are the accepted reference values.

At Re = 100, the primary vortex centre is near (0.617, 0.737). Our solver should reproduce this vortex clearly and match the u-velocity profile along x = 0.5 to within 2–5% on a 41×41 grid.

---

## Changing Reynolds Number

In `run_simulation.py`, change the kinematic viscosity:

```python
nu = 1e-2    # Re = 100  (default, converges easily)
nu = 2.5e-3  # Re = 400  (increase iterations to ~2000)
nu = 1e-3    # Re = 1000 (needs finer grid, ~61x61)
```

---

## Reading Order

If you want to understand every line of code, read:

1. `theory/01_what_problem_are_we_solving.md`
2. `theory/02_finite_volume_discretization.md`
3. `theory/03_momentum_equations.md`
4. `theory/06_gauss_seidel_solver.md`
5. `theory/04_pressure_velocity_coupling.md`
6. `theory/05_simple_algorithm.md`
7. `theory/07_rhie_chow_interpolation.md`
8. `theory/08_lid_driven_cavity_setup.md`

Then open `run_simulation.py` and follow the code — every step references the theory files.

---

## References

- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Ferziger, J.H. & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.
- Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. *AIAA Journal*, 21(11), 1525–1532.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
