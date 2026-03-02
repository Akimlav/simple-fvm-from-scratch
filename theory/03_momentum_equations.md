# 03 — Momentum Equations

## From PDE to Algebraic Equation

After FVM discretization (Chapter 2), each cell (i,j) produces one algebraic
equation for u_P. The standard form is:

```
a_P * u_P = a_E * u_E + a_W * u_W + a_N * u_N + a_S * u_S + b
```

where:
- `a_P` = central coefficient (diagonal)
- `a_E, a_W, a_N, a_S` = neighbour coefficients
- `b` = source term (includes pressure gradient)

---

## Upwind Differencing for Convection

We need the face value `u_e`. The **upwind scheme** picks based on flow direction:

```
u_e = u_P   if  F_e > 0  (flow goes east → upwind is P)
u_e = u_E   if  F_e < 0  (flow goes west → upwind is E)
```

This is equivalent to:

```
Convective contribution from east face:
  F_e * u_e = max(F_e, 0) * u_P + min(F_e, 0) * u_E
            = max(F_e, 0) * u_P - max(-F_e, 0) * u_E
```

Rearranging, the east neighbour coefficient picks up:

```
a_E = D_E + max(-F_e, 0)
```

The `D_E` term is diffusion, `max(-F_e, 0)` adds convection when flow goes
from E towards P (i.e., F_e < 0).

Similarly:

```
a_W = D_W + max(F_w, 0)
a_N = D_N + max(-F_n, 0)
a_S = D_S + max(F_s, 0)
```

---

## Central Coefficient

```
a_P = a_E + a_W + a_N + a_S + (F_e - F_w + F_n - F_s)
```

The extra term `(F_e - F_w + F_n - F_s)` is the net mass outflow.
At convergence this is zero (continuity satisfied), but keeping it here
ensures **diagonal dominance** during iterations.

---

## Pressure Gradient as Source Term

The pressure gradient in x-momentum:

```
- ∂p/∂x ≈ -(p_E - p_W) / (2 dx)   [central difference on collocated grid]
```

Multiplied by cell volume (dx * dy * 1):

```
b_pressure = -(p[i+1,j] - p[i-1,j]) / 2 * dy
```

The factor `dy` comes from `(dx * dy) / dx = dy` after simplification.

Note: on a collocated grid this wide stencil `(p[i+1] - p[i-1])` does NOT
couple adjacent cells in pressure — it's one source of the checkerboard
instability. Rhie-Chow (Chapter 7) fixes the continuity equation.

---

## Under-Relaxation

Direct substitution of the new u often overshoots. Under-relaxation blends:

```
u_new = u_old + α_u * (u_direct - u_old)
```

Equivalently, modify the equation:

```
a_P / α_u * u_P = Σ a_nb u_nb + b + (1 - α_u) / α_u * a_P * u_old
```

In code, we apply it after computing the direct solution:

```python
u_star[i,j] = u[i,j] + urf_u * (u_new - u[i,j])
```

---

## Complete u-Momentum for Cell (i,j)

```
Fu_E = rho * dy * 0.5 * (u[i,j]   + u[i+1,j])    # east face mass flux
Fu_W = rho * dy * 0.5 * (u[i-1,j] + u[i,j])      # west face mass flux
Fu_N = rho * dx * 0.5 * (v[i,j]   + v[i,j+1])    # north face mass flux
Fu_S = rho * dx * 0.5 * (v[i,j-1] + v[i,j])      # south face mass flux

au_E = D_E + max(-Fu_E, 0)
au_W = D_W + max( Fu_W, 0)
au_N = D_N + max(-Fu_N, 0)
au_S = D_S + max( Fu_S, 0)

au_P = au_E + au_W + au_N + au_S + (Fu_E - Fu_W + Fu_N - Fu_S)

rhs = au_E * u[i+1,j] + au_W * u[i-1,j]
    + au_N * u[i,j+1] + au_S * u[i,j-1]
    - dy * (p[i+1,j] - p[i-1,j]) / 2

u_new = rhs / au_P
u_star[i,j] = u[i,j] + urf_u * (u_new - u[i,j])
```

The v-momentum equation is **identical in structure**, replacing u→v and
swapping x/y roles.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| a_E, a_W, a_N, a_S, a_P | `solver/momentum.py` | `solve_u_star` |
| Pressure gradient source | `solver/momentum.py` | `rhs` computation |
| Under-relaxation | `solver/momentum.py` | `u_star[i,j] = ...` |
| D_E, D_W, D_N, D_S | `solver/discretization.py` | `diffusion_coeffs()` |
