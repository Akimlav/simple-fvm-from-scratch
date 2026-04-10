# 04 — Pressure–Velocity Coupling

This chapter explains the central difficulty of incompressible flow: **there is no explicit equation for pressure**. It then shows how a collocated grid makes the problem worse (checkerboard instability) and previews the two techniques that fix it: the SIMPLE algorithm and Rhie–Chow interpolation.

---

## The Problem: No Equation for Pressure

After discretisation (Chapters 2–3) we have, for each interior cell $(i,j)$:

| Equation | Solves for | Contains |
|---|---|---|
| $x$-momentum | $u$ | $u$, $v$, $p$ |
| $y$-momentum | $v$ | $u$, $v$, $p$ |
| Continuity | — | $u$, $v$ only |

Three equations, three unknowns ($u, v, p$) — but look at the structure:

- Momentum equations contain $\nabla p$ as a forcing term, so if we **knew** $p$ we could solve for $u$ and $v$ directly.
- Continuity constrains $\nabla \cdot \mathbf{u} = 0$ but **contains no pressure at all**.

There is no equation of the form "$\text{something}(p) = \text{something else}$". For compressible flow the equation of state $\rho = \rho(p)$ provides the missing link. For **incompressible** flow, $\rho$ is constant regardless of $p$. Pressure acts as a **Lagrange multiplier** — whatever value it takes, it must be the one that makes the velocity field divergence-free.

---

## The Collocated Grid Problem: Checkerboarding

On our collocated grid, $u$, $v$, and $p$ are all stored at the same node locations. This is simple to code but introduces a numerical artefact.

### Checkerboard in the Momentum Equation

The pressure gradient in the $x$-momentum equation uses a **central difference** over two grid spacings:

$$\frac{\partial p}{\partial x}\bigg|_P \approx \frac{p_{i+1,j} - p_{i-1,j}}{2\,\Delta x}$$

Cell $P = (i,j)$ couples to cells $(i+1,j)$ and $(i-1,j)$, **skipping** cell $(i,j)$ itself. Now consider a pressure field that alternates:

```
index:     0     1     2     3     4
p:       100     0   100     0   100
```

At node 2:

$$\frac{\partial p}{\partial x}\bigg|_2 \approx \frac{p_3 - p_1}{2\,\Delta x} = \frac{0 - 0}{2\,\Delta x} = 0$$

The momentum equation sees **zero pressure gradient** even though the pressure is oscillating wildly. The checkerboard pattern is **invisible** to the discrete momentum operator.

### Checkerboard in the Continuity Equation

The problem extends to continuity. If we naively interpolate velocity to faces:

$$u_e = \tfrac{1}{2}(u_P + u_E)$$

then the discrete continuity $\frac{u_e - u_w}{\Delta x} + \frac{v_n - v_s}{\Delta y} = 0$ becomes:

$$\frac{u_{i+1,j} - u_{i-1,j}}{2\,\Delta x} + \frac{v_{i,j+1} - v_{i,j-1}}{2\,\Delta y} = 0$$

Again a **wide stencil** that couples only alternating cells. A checkerboard velocity field would satisfy discrete continuity exactly.

### The Root Cause

Both equations use **two-cell-wide** stencils for the terms that couple pressure and velocity. This decouples the "even" grid from the "odd" grid — two independent solutions coexist on interleaved sub-grids.

---

## Traditional Fix: The Staggered Grid

The classic remedy (Harlow & Welch, 1965) stores different variables at different locations:

- $p$ at cell centres
- $u$ at east/west face centres
- $v$ at north/south face centres

Then the pressure gradient at a $u$-location uses **directly adjacent** pressure values:

$$\frac{\partial p}{\partial x}\bigg|_e \approx \frac{p_E - p_P}{\Delta x}$$

This is a **compact stencil** — it sees every cell, no skipping. Checkerboarding is eliminated. However, staggered grids are significantly more complex to code: different variables live on different grids, and interpolation between them is needed everywhere.

---

## Modern Fix: Collocated Grid + Rhie–Chow

We keep everything at cell centres (**simpler code**, easier to extend to unstructured grids) and fix the checkerboard with a two-part strategy:

### Part 1: Rhie–Chow Interpolation (Chapter 7)

When computing face velocities for the continuity equation, we do **not** use naive interpolation $u_e = \frac{1}{2}(u_P + u_E)$. Instead, we add a pressure-smoothing correction:

$$u_e = \tfrac{1}{2}(u_P^* + u_E^*) - D_f\,(p_E - p_P)$$

The key term is $(p_E - p_P)$: a **compact** pressure gradient that couples **adjacent** cells. If a checkerboard is present, $(p_E - p_P)$ is large, the face velocity is modified, the continuity residual becomes large, and the pressure correction eliminates the oscillation.

### Part 2: SIMPLE Algorithm (Chapter 5)

SIMPLE handles the pressure–velocity coupling iteratively:

1. **Predict** velocities $u^*, v^*$ from momentum using current $p$
2. **Check** continuity via Rhie–Chow face velocities → mass imbalance $b_P$
3. **Solve** a pressure-correction equation to find $p'$ that drives $b_P \to 0$
4. **Correct** $p$ and velocities
5. **Repeat** until $\max|b_P| < \text{tolerance}$

Together, SIMPLE provides the iteration for pressure–velocity coupling, and Rhie–Chow ensures the continuity equation couples adjacent cells so that the checkerboard mode is damped.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Wide pressure gradient (momentum) | `solver/momentum.py` | source term in `solve_u_star` |
| Compact gradient (Rhie–Chow) | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow` |
| SIMPLE outer loop | `solver/simple.py` | `run_simple` |
