# Theory — Reading Roadmap

These 8 chapters derive every equation in the solver from first principles. Start at Chapter 1 and work through in order — each chapter builds on the previous one.

```
                    ┌─────────────────────────┐
                    │  01  Governing Equations │   What PDEs do we solve?
                    │      (Navier–Stokes)     │   Continuous form.
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  02  Finite Volume       │   How to turn a PDE into
                    │      Method (FVM)        │   an algebraic equation.
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  03  Momentum Equations  │   Upwind scheme, coefficients,
                    │      (Discrete)          │   pressure source, under-relaxation.
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  04  Pressure–Velocity│   │  06  Gauss–Seidel     │
    │      Coupling Problem │   │      Linear Solver    │
    │      (Checkerboard)   │   │                       │
    └───────────┬───────────┘   └───────────────────────┘
                │                         ▲
                ▼                         │  (used by 05)
    ┌───────────────────────┐             │
    │  05  SIMPLE Algorithm │─────────────┘
    │      (Pressure        │
    │       Correction)     │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  07  Rhie–Chow        │   Fixes the checkerboard
    │      Interpolation    │   from Chapter 04.
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │  08  Boundary         │   Lid-driven cavity setup,
    │      Conditions &     │   BCs, initial conditions.
    │      Problem Setup    │
    └───────────────────────┘
```

## Chapter Summaries

| # | Chapter | Key Derivation |
|---|---------|----------------|
| 01 | [What Problem Are We Solving?](01_what_problem_are_we_solving.md) | Navier–Stokes from the general transport equation |
| 02 | [Finite Volume Discretization](02_finite_volume_discretization.md) | PDE → volume integral → divergence theorem → face fluxes → $a_P \phi_P = \sum a_{nb} \phi_{nb} + b$ |
| 03 | [Momentum Equations](03_momentum_equations.md) | Upwind coefficients, central coefficient with diagonal dominance, under-relaxation |
| 04 | [Pressure–Velocity Coupling](04_pressure_velocity_coupling.md) | Why the wide stencil can't see checkerboard pressure |
| 05 | [The SIMPLE Algorithm](05_simple_algorithm.md) | Pressure-correction equation derived from velocity correction + continuity |
| 06 | [Gauss–Seidel Solver](06_gauss_seidel_solver.md) | Matrix splitting, in-place updates, convergence condition |
| 07 | [Rhie–Chow Interpolation](07_rhie_chow_interpolation.md) | Compact pressure gradient at faces, consistency with $p'$ equation |
| 08 | [Lid-Driven Cavity Setup](08_lid_driven_cavity_setup.md) | Dirichlet/Neumann BCs, pressure reference level, Re number settings |

## How Theory Maps to Code

Every chapter ends with a table showing exactly which file and function implements each concept. The variable names in the code (`aE`, `aP`, `D_f`, `bP`, `p_prime`) match the symbols in the derivations.

| Theory concept | Solver file |
|---|---|
| Grid, coordinates | `solver/grid.py` |
| Field arrays ($u$, $v$, $p$, $p'$, $b_P$) | `solver/fields.py` |
| FVM coefficients ($a_E$, $a_P$, $D_E$, $F_e$) | `solver/discretization.py` |
| Momentum solve → $u^*$, $v^*$ | `solver/momentum.py` |
| Rhie–Chow face velocities, mass imbalance | `solver/rhie_chow.py` |
| Pressure-correction build & solve | `solver/pressure.py` |
| SIMPLE outer loop, velocity correction | `solver/simple.py` |
| Gauss–Seidel iteration | `solver/linear_solvers.py` |
| Boundary conditions | `solver/boundary_conditions.py` |
