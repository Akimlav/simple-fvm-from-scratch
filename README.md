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
| Language | Python 3.8+ |
| Dependencies | `numpy`, `matplotlib` |

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

On a collocated grid (u, v, p all at the same locations), naive interpolation of velocities to faces lets a **checkerboard pressure field** satisfy continuity exactly — a purely numerical artefact. Rhie–Chow interpolation adds a compact pressure-gradient term to the face velocity that eliminates this decoupling without moving to a staggered grid.

---

## What This Is NOT

This is a **teaching solver**, not a production CFD code. Limitations you should know about:

- **Pure Python loops** — slow on grids larger than ~80x80. No vectorisation or compiled extensions.
- **2D only** — no 3D support.
- **Steady, incompressible, laminar** — no transient terms, no compressibility, no turbulence models.
- **Uniform Cartesian grid** — no unstructured meshes, no local refinement.
- **First-order upwind** — significant numerical diffusion at moderate Reynolds numbers.

For production CFD, see [OpenFOAM](https://www.openfoam.com/), [SU2](https://su2code.github.io/), or [code_saturne](https://www.code-saturne.org/).

---

## Repository Layout

```
simple-fvm-from-scratch/
│
├── README.md                  <- you are here
├── LICENSE                    <- MIT license
├── requirements.txt
├── run_simulation.py          <- entry point: runs SIMPLE, saves results
│
├── theory/                    <- step-by-step derivations (01–08)
│
├── solver/                    <- one file per concept
│   ├── grid.py                <- mesh geometry
│   ├── fields.py              <- field initialisation
│   ├── discretization.py      <- FVM coefficient assembly
│   ├── linear_solvers.py      <- Gauss-Seidel
│   ├── momentum.py            <- u* and v* prediction
│   ├── pressure.py            <- pressure-correction equation
│   ├── rhie_chow.py           <- face velocity interpolation
│   ├── simple.py              <- outer SIMPLE loop
│   └── boundary_conditions.py <- all BCs in one place
│
├── data/                      <- Ghia et al. (1982) benchmark tables
├── post/
│   └── plot_results.py        <- visualisation + Ghia comparison
└── results/                   <- generated plots (*.png)
```

---

## How to Run

```bash
pip install -r requirements.txt
python run_simulation.py
```

Results are saved to `results/` as `.png` plots:

1. Pressure contour + velocity quiver
2. Streamlines
3. Centreline u-velocity vs Ghia et al. (1982) benchmark
4. Centreline v-velocity vs Ghia et al. (1982) benchmark
5. Convergence history

---

## Expected Output

```
Iteration   1 | mass residual: 3.241e+00
Iteration   5 | mass residual: 8.123e-01
Iteration  20 | mass residual: 4.512e-02
Iteration 100 | mass residual: 6.710e-04
Iteration 500 | mass residual: 2.183e-06
```

The continuity residual should decrease overall, though **early iterations
may be non-monotonic** — while the pressure field is still developing from
zero, bP can temporarily increase before settling into a steady decrease.
This is normal. At Re = 100 with a 41×41 grid and 500 SIMPLE iterations,
the centreline profiles match Ghia et al. within a few percent.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| bP diverges to infinity | Under-relaxation too aggressive | Reduce `urf_u`, `urf_v` to 0.3 |
| Slow convergence, barely decreasing | Not enough inner sweeps | Increase `gs_p` to 50 |
| Checkerboard pattern in pressure plot | Bug in Rhie-Chow or `au_P_arr` | Check `au_P_arr` is not all-ones (initialisation bug) |
| Primary vortex not forming | Lid BC not applied to `u_star` | Check `apply_velocity_bcs` is called after each G-S sweep in momentum |
| Centreline profiles shifted from Ghia | Grid too coarse | Increase to 61×61 or 81×81 |

---

## Results

### Re = 100

![Pressure and velocity](results/pressure_velocity_Re100_CentralDifference.png)

![Streamlines](results/streamlines_Re100_CentralDifference.png)

![u-velocity centreline vs Ghia](results/u_centreline_Re100_CentralDifference.png)

![v-velocity centreline vs Ghia](results/v_centreline_Re100_CentralDifference.png)

![Convergence history](results/convergence_Re100_CentralDifference.png)

### Re = 400

![Pressure and velocity](results/pressure_velocity_Re400_CentralDifference.png)

![Streamlines](results/streamlines_Re400_CentralDifference.png)

![u-velocity centreline vs Ghia](results/u_centreline_Re400_CentralDifference.png)

![v-velocity centreline vs Ghia](results/v_centreline_Re400_CentralDifference.png)

---

## Validation: Ghia et al. (1982)

The standard benchmark for this problem is:

> Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, **48**(3), 387-411.

The full benchmark tables (Re = 100 through 10,000) are included in `data/`. At Re = 100, the primary vortex centre is near (0.617, 0.737). Our solver reproduces this vortex clearly and matches the centreline velocity profiles on a 41x41 grid.

---

## Changing Reynolds Number

In `run_simulation.py`, change the kinematic viscosity:

```python
nu = 1e-2    # Re = 100  (default, converges easily)
nu = 2.5e-3  # Re = 400  (increase iterations to ~2000, use 61x61 grid)
```

---

## Reading Order

Read the theory files in this order, then open `run_simulation.py` and follow
the code — every step references the theory files.

1. `theory/01_what_problem_are_we_solving.md` — physics and governing equations
2. `theory/02_finite_volume_discretization.md` — how PDEs become algebraic equations
3. `theory/03_momentum_equations.md` — assembling the u and v equations
4. `theory/04_pressure_velocity_coupling.md` — why pressure is hard, the checkerboard problem
5. `theory/05_simple_algorithm.md` — the full SIMPLE loop
6. `theory/06_gauss_seidel_solver.md` — the linear solver used inside SIMPLE
7. `theory/07_rhie_chow_interpolation.md` — fixing the checkerboard in continuity
8. `theory/08_lid_driven_cavity_setup.md` — boundary conditions and problem setup

---

## References

- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Ferziger, J.H. & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.
- Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. *AIAA Journal*, 21(11), 1525–1532.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

---

## License

MIT. See [LICENSE](LICENSE).
