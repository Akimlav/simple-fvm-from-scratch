# 08 — Lid-Driven Cavity Setup

## Domain

Unit square: 0 ≤ x ≤ 1,  0 ≤ y ≤ 1

Uniform grid with `N×N` cells. Grid spacing: `dx = dy = 1/(N-1)`.

Cell centres at:
```
x[i] = i * dx,   i = 0 ... N-1
y[j] = j * dy,   j = 0 ... N-1
```

---

## Velocity Boundary Conditions

| Wall | Location | u | v |
|---|---|---|---|
| Bottom | y = 0 | 0 | 0 |
| Top (lid) | y = L | U_lid = 1 | 0 |
| Left | x = 0 | 0 | 0 |
| Right | x = L | 0 | 0 |

These are **Dirichlet** conditions — fixed values applied after each sweep.

In code: the boundary cells (index 0 and N-1) are NOT solved by Gauss-Seidel.
The inner loops run from `i = 1 ... N-2` and `j = 1 ... N-2`.
Boundary cells are set directly:

```python
u[:, -1] = U_lid   # top wall: moving lid
u[:,  0] = 0.0     # bottom wall
u[0,  :] = 0.0     # left wall
u[-1, :] = 0.0     # right wall
v = 0 everywhere on boundaries
```

---

## Pressure Boundary Conditions

**At walls:** zero normal gradient (Neumann condition)

The wall is impermeable — no flow through it — so the pressure-correction
normal gradient is zero at all walls:

```
dp'/dn = 0  →  p'_boundary = p'_first_interior
```

In code:
```python
p_prime[0,  :] = p_prime[1,  :]   # left wall
p_prime[-1, :] = p_prime[-2, :]   # right wall
p_prime[:,  0] = p_prime[:,  1]   # bottom wall
p_prime[:, -1] = p_prime[:, -2]   # top wall
```

---

## Why Pressure Needs a Reference

For an **enclosed** domain with only Neumann pressure BCs, the pressure
equation has infinitely many solutions — you can add any constant to p
and still satisfy both the momentum equations and continuity.

Mathematically: the pressure-correction equation is:
```
∇·(D_f ∇p') = bP
```
This is a Poisson equation with only Neumann BCs. The solution exists
only if ∫bP dV = 0 (which it is, by global mass conservation), but is
unique only up to an additive constant.

**Fix 1 (used here):** subtract the mean of p' after each solve:
```python
p_prime -= np.mean(p_prime)
```

**Fix 2 (alternative):** pin one cell: `p[0, 0] = 0` and skip it in the solver.

Both approaches are valid. Subtracting the mean is simpler and keeps the
pressure physically centered around zero.

---

## Initial Conditions

```python
u[:, :] = 0.0    # fluid at rest
v[:, :] = 0.0
p[:, :] = 0.0    # zero pressure everywhere

# Apply lid BC immediately
u[:, -1] = U_lid
```

The solver will develop the flow from this rest state.

---

## Reynolds Number

```
Re = U_lid * L / nu = 1.0 * 1.0 / nu
```

| nu | Re | Grid | Notes |
|---|---|---|---|
| 0.01 | 100 | 41×41 | Easy, fast, validates against Ghia |
| 0.0025 | 400 | 41×41 | Good secondary vortices |
| 0.001 | 1000 | 81×81 | Needs finer grid |

---

## Expected Flow Features (Re = 100)

1. **Primary vortex**: large clockwise rotation filling most of the cavity
2. **Bottom-left corner vortex**: small counter-clockwise vortex
3. **Bottom-right corner vortex**: small clockwise vortex
4. **Smooth pressure field**: no checkerboard (Rhie-Chow working correctly)

---

## Convergence Criterion

We monitor the maximum mass imbalance across all cells:

```python
residual = np.max(np.abs(bP))
```

Convergence when `residual < 1e-5` (or after fixed number of iterations).

---

## Where This Appears in the Code

| Concept | File | Function/Variable |
|---|---|---|
| Grid creation | `solver/grid.py` | `Grid` class |
| Field initialization | `solver/fields.py` | `Fields` class |
| Velocity BCs | `solver/boundary_conditions.py` | `apply_velocity_bcs` |
| Pressure BCs | `solver/boundary_conditions.py` | `apply_pressure_neumann_bcs` |
| Pressure reference | `solver/pressure.py` | `p_prime -= np.mean(p_prime)` |
| Main run | `run_simulation.py` | top-level script |
