# 02 — Finite Volume Discretization

## The Core Idea

Instead of approximating derivatives at a point (finite differences), FVM
**integrates the equations over a control volume** and converts volume
integrals to **surface fluxes**.

For any scalar φ, the divergence theorem says:

```
∫_V ∇·(ρ u φ) dV  =  ∮_S ρ u φ · n̂ dS
```

This becomes a **sum over faces**:

```
∮_S ρ u φ · n̂ dS  ≈  F_E φ_E  -  F_W φ_W  +  F_N φ_N  -  F_S φ_S
```

where `F` is the mass flux through each face.

---

## The Grid

We use a **uniform Cartesian collocated grid** with `nx × ny` nodes.
All variables (u, v, p) are stored **at the same grid points** (collocated).

The grid is **node-based**: nodes include the physical boundaries.

```
x[i] = i * dx,   i = 0 ... nx-1
y[j] = j * dy,   j = 0 ... ny-1
```

So `x[0] = 0` and `x[nx-1] = L` — the first and last points sit exactly on
the walls. Interior nodes where the equations are solved run `i = 1..nx-2`.

A worked example: for `nx = 5`, `L = 1.0`, `dx = 0.25`:
```
x[0]=0.0  x[1]=0.25  x[2]=0.5  x[3]=0.75  x[4]=1.0
 wall      interior                          wall
```

Each interior node P connects to its four neighbours E, W, N, S:

```
           N (i, j+1)
           |
    W ---- P ---- E
(i-1,j)  (i,j)  (i+1,j)
           |
           S (i, j-1)
```

Face areas (per unit depth in z):

```
A_EW = dy   (east/west faces, normal in x)
A_NS = dx   (north/south faces, normal in y)
```

---

## Convective Flux Derivation

Starting from the x-momentum convective term:

```
∫_V ∂(ρ u u)/∂x dV
```

Apply divergence theorem over control volume (i,j):

```
= ∫_S ρ u u · n̂_x dS
≈ (ρ u u)_e * A_e  -  (ρ u u)_w * A_w
= ρ u_e * u_e * dy  -  ρ u_w * u_w * dy
```

The **mass flux** through the east face:

```
F_E = ρ * u_e * dy     [kg/s per unit depth]
```

So the convective contribution becomes:

```
F_E * u_e  -  F_W * u_w  +  F_N * v_n  -  F_S * v_s
```

A concrete example with `ρ=1`, `dy=0.25`, `u_e=0.5`:
```
F_E = 1.0 * 0.5 * 0.25 = 0.125 kg/s
```

---

## Diffusive Flux Derivation

For the viscous (diffusion) term in x-momentum:

```
∫_V μ ∂²u/∂x² dV = ∫_S μ ∂u/∂x · n̂ dS
```

Approximating the gradient at the east face by central difference:

```
μ (∂u/∂x)_e ≈ μ * (u_E - u_P) / dx
```

Multiplied by face area `dy`:

```
Diffusion through east face = μ * dy/dx * (u_E - u_P) = D_E * (u_E - u_P)
```

where the **diffusion coefficient** is:

```
D_E = μ * dy / dx
D_W = μ * dy / dx
D_N = μ * dx / dy
D_S = μ * dx / dy
```

These are **constant** for a uniform grid — computed once before the loop.

A concrete example with `μ=0.01`, `dx=dy=0.25`:
```
D_E = 0.01 * 0.25 / 0.25 = 0.01
```

---

## Combined: The FVM Equation for One Node

Summing convection + diffusion for node (i,j):

```
[F_E u_e - F_W u_w + F_N u_n - F_S u_s]   ← convection
- [D_E(u_E - u_P) - D_W(u_P - u_W)
 + D_N(u_N - u_P) - D_S(u_P - u_S)]       ← diffusion
= Source terms (pressure gradient)
```

This is rearranged in Chapter 3 into the standard form:

```
a_P u_P = a_E u_E + a_W u_W + a_N u_N + a_S u_S + b
```

---

## Where This Appears in the Code

| Concept | File | Variable |
|---|---|---|
| Grid spacing dx, dy | `solver/grid.py` | `grid.dx`, `grid.dy` |
| Node coordinates | `solver/grid.py` | `grid.x`, `grid.y` |
| Face areas, D_E etc | `solver/discretization.py` | `D_E, D_W, D_N, D_S` |
| Mass fluxes F_E, F_W | `solver/discretization.py` | `convective_mass_fluxes()` → `F_E, F_W, F_N, F_S` |
| Convection coefficients | `solver/discretization.py` | `neighbour_coeffs()` |
