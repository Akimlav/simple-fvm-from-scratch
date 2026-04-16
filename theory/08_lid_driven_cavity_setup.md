# 08 — Lid-Driven Cavity Setup

This chapter collects all the implementation-specific choices: domain, grid, boundary conditions, initial conditions, and convergence criteria.

---

## Domain and Grid

**Domain:** unit square $0 \leq x \leq 1$, $0 \leq y \leq 1$.

**Grid:** uniform Cartesian, $N \times N$ nodes. Grid spacing:

$$\Delta x = \Delta y = \frac{L}{N - 1} = \frac{1}{N - 1}$$

Node coordinates (boundary nodes included):

$$x_i = i\,\Delta x \quad \text{for } i = 0, \ldots, N-1$$

$$y_j = j\,\Delta y \quad \text{for } j = 0, \ldots, N-1$$

Nodes $i = 0$ and $i = N-1$ lie on the physical walls. The interior nodes where equations are solved run $i = 1, \ldots, N-2$.

---

## Velocity Boundary Conditions (Dirichlet)

| Wall | Location | $u$ | $v$ |
|---|---|---|---|
| Bottom | $y = 0$, $j = 0$ | $0$ | $0$ |
| Top (lid) | $y = L$, $j = N-1$ | $U_{\text{lid}} = 1$ | $0$ |
| Left | $x = 0$, $i = 0$ | $0$ | $0$ |
| Right | $x = L$, $i = N-1$ | $0$ | $0$ |

These are **Dirichlet** conditions — fixed values applied directly to boundary nodes after every Gauss–Seidel sweep.

```python
u[:, -1] = U_lid    # top wall: moving lid
u[:,  0] = 0.0      # bottom wall
u[0,  :] = 0.0      # left wall
u[-1, :] = 0.0      # right wall
# v = 0 on all boundaries
```

### Why Re-Apply After Each Sweep?

Gauss–Seidel updates interior nodes using neighbour values. Interior nodes adjacent to the lid ($j = N-2$) reference $u[i, N-1]$. If we only set the BC once at the start, the solver could overwrite it. Re-applying after each sweep guarantees the lid velocity is always $U_{\text{lid}}$.

---

## Pressure Boundary Conditions (Neumann)

At solid walls the flow is impermeable: no velocity correction normal to the wall. From the velocity correction $u' = -(d_u)\,(p'_E - p'_P)$, setting $u' = 0$ at a wall requires:

$$\frac{\partial p'}{\partial n}\bigg|_{\text{wall}} = 0$$

This **zero-gradient (Neumann)** condition is implemented by copying the first interior value to the boundary:

```python
p_prime[0,  :] = p_prime[1,  :]    # left wall
p_prime[-1, :] = p_prime[-2, :]    # right wall
p_prime[:,  0] = p_prime[:,  1]    # bottom wall
p_prime[:, -1] = p_prime[:, -2]    # top wall
```

This is applied after each pressure-correction solve.

---

## Pressure Reference Level

With Neumann conditions on all boundaries, the pressure-correction equation has infinitely many solutions — any constant can be added to $p'$ without changing any gradient. To make the solution unique, we **subtract the mean**:

$$p' \;\leftarrow\; p' - \overline{p'}$$

This keeps the pressure field centred around zero. An equivalent alternative is to pin one node ($p[0,0] = 0$) and skip it in the solver.

---

## Initial Conditions

```python
u[:, :] = 0.0      # fluid at rest
v[:, :] = 0.0
p[:, :] = 0.0      # zero pressure everywhere
u[:, -1] = U_lid   # apply lid BC immediately
```

SIMPLE develops the flow from this quiescent state. The initial pressure guess ($p = 0$) is far from the true pressure, but SIMPLE corrects it iteratively.

---

## Reynolds Number Settings

$$\text{Re} = \frac{U_{\text{lid}} \cdot L}{\nu} = \frac{1}{\nu}$$

| $\nu$ | $\text{Re}$ | Suggested grid | Notes |
|---|---|---|---|
| $0.01$ | $100$ | $41 \times 41$ | Fast convergence, easy Ghia validation |
| $0.0025$ | $400$ | $41 \times 41$ | Visible secondary corner vortices |
| $0.001$ | $1000$ | $81 \times 81$ | Needs finer grid, more iterations |

---

## Expected Flow at $\text{Re} = 100$

1. **Primary vortex** — large clockwise rotation filling most of the cavity, centre near $(0.617, 0.737)$
2. **Bottom-left corner vortex** — small, counter-clockwise
3. **Bottom-right corner vortex** — small, clockwise
4. **Smooth pressure field** — no checkerboard (Rhie–Chow is working)

---

## Convergence Criterion

We monitor the maximum absolute mass imbalance:

$$\text{residual} = \max_{i,j}\,|b_P[i,j]|$$

Convergence is declared when $\text{residual} < 10^{-5}$. Typical behaviour:
- First ~10 iterations: residual may be non-monotonic as the pressure field develops from zero
- Iterations 10–100: steady exponential decrease
- Total: ~200–400 iterations for $\text{Re} = 100$ on a 41×41 grid with $\alpha_u = 0.5$, $\alpha_p = 0.2$

---

## Where This Appears in the Code

| Concept | File | Function / Variable |
|---|---|---|
| Grid creation | `solver/grid.py` | `Grid` class |
| Field initialisation | `solver/fields.py` | `Fields` class |
| Velocity BCs | `solver/boundary_conditions.py` | `apply_velocity_bcs` |
| Pressure BCs | `solver/boundary_conditions.py` | `apply_pressure_neumann_bcs` |
| Pressure reference | `solver/pressure.py` | `p_prime -= np.mean(p_prime)` |
| Main simulation | `run_simulation.py` | top-level script |
