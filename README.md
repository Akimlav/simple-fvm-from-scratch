# simple-fvm-from-scratch

A complete, from-scratch 2D incompressible Navier–Stokes solver built to teach how SIMPLE and the Finite Volume Method actually work — equation by equation, line by line.

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
## Quick Start

```bash
pip install numpy matplotlib
python run_simulation.py
```

This runs a lid-driven cavity simulation at Re = 100 on a 129×129 grid using central differencing. You'll see convergence output and plots saved to `results/`: pressure contours, streamlines, and centreline velocity comparisons against the Ghia et al. (1982) benchmark.

---

## What's Inside

| Component | Method |
|---|---|
| Spatial discretisation | Finite Volume Method, uniform collocated Cartesian grid |
| Pressure–velocity coupling | SIMPLE algorithm (Patankar, 1980) |
| Checkerboard fix | Rhie–Chow interpolation |
| Convection schemes | Upwind (UDS), Central Differencing (CDS), Second-Order Upwind (SOU) |
| Linear solver | Gauss–Seidel iteration |
| Dependencies | `numpy` only (`matplotlib` for plots) |

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
├── run_simulation.py          ← entry point (single case)
├── requirements.txt
│
├── solver/                    ← one file per concept
│   ├── grid.py                ← mesh geometry
│   ├── fields.py              ← field arrays (u, v, p, p', bP)
│   ├── discretization.py      ← FVM coefficient assembly
│   ├── convection_schemes.py  ← UDS, CDS, SOU schemes
│   ├── momentum.py            ← u* and v* prediction
│   ├── pressure.py            ← pressure-correction equation
│   ├── rhie_chow.py           ← face velocity interpolation
│   ├── simple.py              ← outer SIMPLE loop
│   ├── linear_solvers.py      ← Gauss–Seidel solver
│   └── boundary_conditions.py ← velocity and pressure BCs
│
├── theory/                    ← step-by-step derivations (start here)
│   ├── README.md              ← reading roadmap with dependency graph
│   ├── 01_what_problem_are_we_solving.md
│   ├── 02_finite_volume_discretization.md
│   ├── 03_momentum_equations.md
│   ├── 04_pressure_velocity_coupling.md
│   ├── 05_simple_algorithm.md
│   ├── 06_gauss_seidel_solver.md
│   ├── 07_rhie_chow_interpolation.md
│   └── 08_lid_driven_cavity_setup.md
│
├── post/
│   └── plot_results.py        ← visualisation + Ghia comparison
│
└── results/                   ← saved plots (PNGs committed)
```

Every equation in the `theory/` notes has a direct counterpart in the `solver/` code. Each theory chapter ends with a table mapping concepts to files and functions.

---

## Changing Parameters

Edit `run_simulation.py`:

```python
# Reynolds number (change nu)
nu = 1e-2     # Re = 100  (default, fast convergence)
nu = 2.5e-3   # Re = 400  (increase n_iter to ~2000)
nu = 1e-3     # Re = 1000 (use 81×81 or larger grid)

# Grid resolution
nx, ny = 129, 129  # default
nx, ny = 41, 41    # fast, good for Re=100 testing
nx, ny = 256, 256  # high resolution

# Under-relaxation (reduce if diverging)
urf_u, urf_v = 0.7, 0.7   # momentum
urf_p = 0.3                # pressure

# Inner solver sweeps
gs_mom = 20   # Gauss–Seidel sweeps for momentum
gs_p   = 100  # Gauss–Seidel sweeps for pressure correction

# Convection scheme
convection_scheme = SCHEME_CDS  # central differencing (default)
convection_scheme = SCHEME_UDS  # first-order upwind
convection_scheme = SCHEME_SOU  # second-order upwind (deferred correction)
```

---

## Results

### Re = 100, Central Differencing (129×129 grid)

| Streamlines | Pressure + velocity |
|---|---|
| ![Streamlines Re100](results/streamlines_Re100_CentralDifference.png) | ![Pressure Re100](results/pressure_velocity_Re100_CentralDifference.png) |

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
| u-centreline vs Ghia | v-centreline vs Ghia |
|---|---|
| ![u-centreline Re100](results/u_centreline_Re100_CentralDifference.png) | ![v-centreline Re100](results/v_centreline_Re100_CentralDifference.png) |

| Convergence history |
|---|
| ![Convergence Re100](results/convergence_Re100_CentralDifference.png) |

> Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, **48**(3), 387-411.

The full benchmark tables (Re = 100 through 10,000) are included in `data/`. At Re = 100, the primary vortex centre is near (0.617, 0.737). Our solver reproduces this vortex clearly and matches the centreline velocity profiles on a 41x41 grid.
### Re = 400, Central Differencing (129×129 grid)

| Streamlines | Pressure + velocity |
|---|---|
| ![Streamlines Re400](results/streamlines_Re400_CentralDifference.png) | ![Pressure Re400](results/pressure_velocity_Re400_CentralDifference.png) |

| u-centreline vs Ghia | v-centreline vs Ghia |
|---|---|
| ![u-centreline Re400](results/u_centreline_Re400_CentralDifference.png) | ![v-centreline Re400](results/v_centreline_Re400_CentralDifference.png) |

---

## Troubleshooting

```python
nu = 1e-2    # Re = 100  (default, converges easily)
nu = 2.5e-3  # Re = 400  (increase iterations to ~2000, use 61x61 grid)
```
| Symptom | Likely cause | Fix |
|---|---|---|
| Residual diverges to infinity | Under-relaxation too aggressive | Reduce `urf_u`, `urf_v` to 0.3 |
| Very slow convergence | Too few pressure sweeps | Increase `gs_p` to 150 |
| Checkerboard in pressure plot | Bug in Rhie–Chow or `au_P_arr` | Check `au_P_arr` is not all-ones |
| Vortex not forming | Lid BC not enforced during solve | Verify `apply_velocity_bcs` is called after each G-S sweep |
| Centreline profiles shifted from Ghia | Grid too coarse or first-order scheme | Increase to 129×129, try CDS or SOU scheme |

---

## Validation

Benchmark: Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* Journal of Computational Physics, **48**(3), 387–411.

At Re = 100, the primary vortex centre should be near (0.617, 0.737). Centreline velocity profiles match Ghia within 2–5% on a 129×129 grid with central difference scheme. With first-order upwind the discrepancy is larger due to numerical diffusion.

---

## References

- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
- Ferziger, J.H. & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.
- Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. *AIAA Journal*, 21(11), 1525–1532.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

---

## License

MIT. See [LICENSE](LICENSE).
- Ferziger, J.H. & Peric, M. (2002). *Computational Methods for Fluid Dynamics*. Springer.
- Rhie, C.M. & Chow, W.L. (1983). Numerical study of the turbulent flow past an airfoil with trailing edge separation. *AIAA Journal*, **21**(11), 1525–1532.
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *Journal of Computational Physics*, **48**(3), 387–411.
