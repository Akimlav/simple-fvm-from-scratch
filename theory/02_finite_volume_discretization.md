# 02 — Finite Volume Discretization

## The Core Idea

The Finite Volume Method (FVM) discretises a PDE not by approximating derivatives at a point (finite differences) but by **integrating the equation over a control volume** and converting volume integrals into **surface fluxes** via the divergence theorem. This guarantees that what flows out of one cell flows into its neighbour — conservation is built into the method at the discrete level.

---

## Step 1: Start from the Continuous Equation

From Chapter 1, every equation we solve has the general transport form:

$$\frac{\partial(\rho u \phi)}{\partial x} + \frac{\partial(\rho v \phi)}{\partial y} = \frac{\partial}{\partial x}\left(\Gamma \frac{\partial \phi}{\partial x}\right) + \frac{\partial}{\partial y}\left(\Gamma \frac{\partial \phi}{\partial y}\right) + S_\phi$$

In compact vector notation:

$$\nabla \cdot (\rho \mathbf{u} \phi) = \nabla \cdot (\Gamma \nabla \phi) + S_\phi$$

---

## Step 2: Integrate over a Control Volume

Pick a control volume $V$ with closed surface $S$. Integrate both sides over $V$:

$$
\int_V \nabla \cdot (\rho \mathbf{u}\ \phi)\ dV = \int_V \nabla \cdot (\Gamma \nabla \phi)\ dV + \int_V S_\phi\ dV
$$

---

## Step 3: Apply the Divergence Theorem

The divergence theorem converts volume integrals of divergences into surface integrals:

$$\int_V \nabla \cdot \mathbf{F}\ dV = \oint_S \mathbf{F} \cdot \hat{\mathbf{n}} dS$$

Applying this to both the convective and diffusive terms:

$$\underbrace{\oint_S (\rho \mathbf{u}\ \phi) \cdot \hat{\mathbf{n}} dS}_{\text{net convective flux out}} = \underbrace{\oint_S (\Gamma \nabla \phi) \cdot \hat{\mathbf{n}}\ dS}_{\text{net diffusive flux in}} + \underbrace{\bar{S}_\phi \Delta V}_{\text{source}}$$

This is **exact** — no approximation has been made yet. The key insight: we only need fluxes through the cell faces, not the full field inside each cell.

---

## Step 4: Approximate Face Fluxes on a Cartesian Grid

### The Grid

We use a **uniform Cartesian collocated grid** with $n_x \times n_y$ nodes. All variables ($u$, $v$, $p$) are stored at the same grid points.

```
x[i] = i · Δx,    i = 0, ..., nx−1
y[j] = j · Δy,    j = 0, ..., ny−1
```

The grid is **node-based**: nodes 0 and $n_x - 1$ sit exactly on the physical boundaries. Interior nodes (where the equations are solved) run $i = 1, \ldots, n_x - 2$.

Each interior node $P = (i, j)$ has four neighbours:

```
            N (i, j+1)
            |
  W ─────── P ─────── E
(i−1,j)   (i,j)   (i+1,j)
            |
            S (i, j−1)
```

Between $P$ and each neighbour lies a **face** (e, w, n, s). Face areas per unit depth:

$$A_e = A_w = \Delta y \qquad A_n = A_s = \Delta x$$

### From Surface Integral to Four Face Fluxes

The surface integral becomes a sum over the four faces:

$$\oint_S (\rho \mathbf{u} \phi) \cdot \hat{\mathbf{n}} dS \approx  \underbrace{(\rho u \phi)_e \Delta y}_{J_e^c} - \underbrace{(\rho u \phi)_w \Delta y}_{J_w^c} + \underbrace{(\rho v \phi)_n \Delta x}_{J_n^c} - \underbrace{(\rho v \phi)_s \Delta x}_{J_s^c}$$

The signs follow from the outward normal: $+x$ at the east face, $-x$ at the west face, $+y$ at the north face, $-y$ at the south face.

Similarly for diffusion:

$$\oint_S (\Gamma \nabla \phi) \cdot \hat{\mathbf{n}}\;dS \;\approx\; \Gamma\frac{\phi_E - \phi_P}{\Delta x} \Delta y - \Gamma\frac{\phi_P - \phi_W}{\Delta x} \Delta y + \Gamma\frac{\phi_N - \phi_P}{\Delta y} \Delta x - \Gamma\frac{\phi_P - \phi_S}{\Delta y} \Delta x$$

---

## Step 5: Define Mass Fluxes and Diffusion Conductances

**Mass fluxes** through each face (positive in the $+x$ or $+y$ direction):

$$F_e = \rho u_e \Delta y \qquad F_w = \rho u_w \Delta y$$

$$F_n = \rho v_n \Delta x \qquad F_s = \rho v_s \Delta x$$

**Diffusion conductances** (face area divided by distance between adjacent nodes):

$$D_e = \Gamma\frac{\Delta y}{\Delta x} \qquad D_w = \Gamma\frac{\Delta y}{\Delta x}$$

$$D_n = \Gamma\frac{\Delta x}{\Delta y} \qquad D_s = \Gamma\frac{\Delta x}{\Delta y}$$

For a uniform grid with $\Gamma = \mu$, these are **constant** — computed once before the SIMPLE loop.

The integrated equation now reads:

$$\underbrace{F_e \phi_e - F_w \phi_w + F_n \phi_n - F_s\,\phi_s}_{\text{convection}} = \underbrace{D_e(\phi_E - \phi_P) - D_w(\phi_P - \phi_W) + D_n(\phi_N - \phi_P) - D_s(\phi_P - \phi_S)}_{\text{diffusion}} + S_\phi \Delta V$$

**Everything is exact except for two unknowns: the face values** $\phi_e, \phi_w, \phi_n, \phi_s$ in the convective terms. These are between the grid nodes, so we need an **interpolation scheme** to express them in terms of nodal values. That is the subject of Chapter 3.

---

## Step 6: Standard Algebraic Form

After choosing an interpolation scheme (Chapter 3 derives the upwind choice), all face values become linear combinations of $\phi_P$ and its neighbours. The equation for each cell collapses into:

$$\boxed{a_P \phi_P = a_E \phi_E + a_W \phi_W + a_N \phi_N + a_S \phi_S + b}$$

where:
- $a_E, a_W, a_N, a_S$ are **neighbour coefficients** (contain both convection and diffusion contributions)
- $a_P$ is the **central coefficient** (diagonal of the linear system)
- $b$ is the **source term** (includes the pressure gradient for momentum)

Writing this equation for every interior node $(i,j)$ produces a large, sparse linear system — one equation per node. The system is solved iteratively by Gauss–Seidel (Chapter 6).

---

## Why FVM?

1. **Conservation** — fluxes through shared faces are computed once and used by both neighbours, so mass/momentum is exactly conserved at the discrete level.
2. **Physical clarity** — every term in the algebraic equation corresponds to a flux through a specific face, making it easy to check the physics.
3. **Flexibility** — the same framework handles any PDE in transport form. Changing the equation means changing $\phi$, $\Gamma$, and $S_\phi$; the discretisation machinery stays the same.

---

## Where This Appears in the Code

| Concept | File | Variable / Function |
|---|---|---|
| Grid spacing $\Delta x$, $\Delta y$ | `solver/grid.py` | `grid.dx`, `grid.dy` |
| Node coordinates | `solver/grid.py` | `grid.x`, `grid.y` |
| Diffusion conductances $D_e, D_w, D_n, D_s$ | `solver/discretization.py` | `diffusion_coeffs()` |
| Mass fluxes $F_e, F_w, F_n, F_s$ | `solver/discretization.py` | `convective_mass_fluxes()` |
| Neighbour coefficients | `solver/discretization.py` | `neighbour_coeffs()` |
| Central coefficient $a_P$ | `solver/discretization.py` | `central_coeff()` |
