# 01 — What Problem Are We Solving?

## The Physical Setup

We solve for **steady, incompressible, viscous flow** inside a unit square cavity with a moving lid.

```
y
^
1 ── u = U_lid = 1,  v = 0 ──────   ← moving lid drags the fluid
|                              |
u=0, v=0               u=0, v=0     ← left and right walls: no-slip
|         FLUID                |
|                              |
0 ── u = 0,  v = 0 ────────────     ← stationary floor: no-slip
0                              1  → x
```

The lid drags the fluid rightward at the top, creating a **recirculating vortex** that fills most of the cavity. At $\text{Re} = 100$ the flow is smooth, steady, and laminar — an ideal first test case for a Navier–Stokes solver.

---

## From Conservation Laws to the Navier–Stokes Equations

All of fluid mechanics is built on conservation principles: mass cannot appear or disappear, and Newton's second law applies to every fluid element. In this section we start from those principles and arrive at the exact equations our solver discretises.

### The General Transport Equation

Every conservation law in this project — mass, $x$-momentum, $y$-momentum, and later the pressure correction — is a special case of a single **scalar transport equation**. For a conserved quantity $\phi$ per unit mass, the steady-state transport is:

$$\underbrace{\nabla \cdot (\rho \mathbf{u}  \phi)}_{\text{convection}} = \underbrace{\nabla \cdot (\Gamma \nabla \phi)}_{\text{diffusion}} + \underbrace{S_\phi}_{\text{source}}$$

In 2D component form:

$$\frac{\partial (\rho u \phi)}{\partial x} + \frac{\partial (\rho v \phi)}{\partial y} = \frac{\partial}{\partial x}\left(\Gamma \frac{\partial \phi}{\partial x}\right) + \frac{\partial}{\partial y}\left(\Gamma \frac{\partial \phi}{\partial y}\right) + S_\phi$$

**This is the equation the Finite Volume Method discretises** (Chapter 2). All three governing equations below are obtained by choosing $\phi$, $\Gamma$, and $S_\phi$.

### Conservation of Mass (Continuity)

Set $\phi = 1$, $\Gamma = 0$, $S_\phi = 0$ in the general transport equation:

$$\nabla \cdot (\rho \mathbf{u}) = 0$$

For an **incompressible** fluid ($\rho = \text{const}$), divide by $\rho$:

$$\boxed{\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0}$$

**Physical meaning:** whatever volume of fluid enters any region must also leave it. No accumulation, no voids — the velocity field is divergence-free everywhere.

### Conservation of $x$-Momentum

Set $\phi = u$ (the $x$-component of velocity), $\Gamma = \mu$ (dynamic viscosity), $S_\phi = -\partial p / \partial x$:

$$\boxed{\rho\left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)}$$

or equivalently in **conservative** (divergence) form, which is what FVM actually integrates:

$$\frac{\partial (\rho u u)}{\partial x} + \frac{\partial (\rho v u)}{\partial y} = -\frac{\partial p}{\partial x} + \frac{\partial}{\partial x}\left(\mu \frac{\partial u}{\partial x}\right) + \frac{\partial}{\partial y}\left(\mu \frac{\partial u}{\partial y}\right)$$

| Term | Role |
|---|---|
| $\rho(u \partial_x u + v \partial_y u)$ | **Convection** — fluid carries its own momentum |
| $-\partial_x p$ | **Pressure gradient** — pushes fluid from high to low pressure |
| $\mu(\partial_{xx}u + \partial_{yy}u)$ | **Diffusion** — viscosity smooths out velocity gradients |

### Conservation of $y$-Momentum

Identical structure with $u \to v$ and $x \leftrightarrow y$:

$$\boxed{\rho\left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) = -\frac{\partial p}{\partial y} + \mu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)}$$

### Summary: Three Equations, Three Unknowns

| Equation | Conservation of | $\phi$ | $\Gamma$ | $S_\phi$ |
|---|---|---|---|---|
| Continuity | mass | $1$ | $0$ | $0$ |
| $x$-momentum | horizontal momentum | $u$ | $\mu$ | $-\partial p/\partial x$ |
| $y$-momentum | vertical momentum | $v$ | $\mu$ | $-\partial p/\partial y$ |

---

## The Reynolds Number

The relative importance of convection vs. diffusion is captured by one dimensionless parameter:

$$\text{Re} = \frac{\rho  U  L}{\mu} = \frac{U  L}{\nu}$$

where $\nu = \mu / \rho$ is the kinematic viscosity.

With our settings $U = 1\;\text{m/s}$, $L = 1\;\text{m}$, $\nu = 0.01\;\text{m}^2/\text{s}$:

$$\text{Re} = \frac{1 \times 1}{0.01} = 100$$

| $\text{Re}$ range | Physics |
|---|---|
| $\text{Re} \ll 1$ | Viscosity dominates — creeping (Stokes) flow |
| $\text{Re} \sim 100$ | Moderate inertia — smooth recirculation, our target |
| $\text{Re} \gg 1000$ | Inertia dominates — turbulence, boundary layers |

At $\text{Re} = 100$ the flow develops one large **primary vortex** and two small **corner vortices** at the bottom, all smooth and steady.

---

## Why Is This Problem Hard to Solve Numerically?

Three unknowns ($u$, $v$, $p$) and three equations — seems straightforward. The difficulty is **pressure**.

**The coupling problem:** pressure appears as a source in both momentum equations, but there is **no independent equation for pressure**. Continuity constrains the velocity field ($\nabla \cdot \mathbf{u} = 0$) but says nothing directly about $p$.

For compressible flow an equation of state ($\rho = \rho(p)$) closes the system. For incompressible flow that link is severed: $\rho$ is constant regardless of $p$. Pressure becomes a **Lagrange multiplier** — the field that, whatever its value, enforces the divergence-free constraint.

This is the **pressure–velocity coupling problem**. The SIMPLE algorithm (Chapter 5) resolves it iteratively:

1. **Guess** a pressure field $p$
2. **Solve** momentum → approximate velocities $u^*$, $v^*$ (violate continuity)
3. **Derive** a pressure correction $p'$ from the continuity constraint
4. **Correct** $p$ and velocities
5. **Repeat** until continuity is satisfied within a tolerance

---

## Validation Benchmark: Ghia et al. (1982)

The standard reference for this problem is:

> Ghia, U., Ghia, K.N., & Shin, C.T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* Journal of Computational Physics, **48**(3), 387–411.

They solved the lid-driven cavity on a 129×129 grid and published tabulated centreline velocities at several Reynolds numbers. At $\text{Re} = 100$ the primary vortex centre is near $(0.617,\; 0.737)$. Our solver should reproduce this vortex clearly and match the $u$-velocity profile along $x = 0.5$ to within 2–5% on a 41×41 grid.

---

## Where This Appears in the Code

| Concept | File | Variable / Function |
|---|---|---|
| Domain size $L$ | `run_simulation.py` | `L = 1.0` |
| Kinematic viscosity $\nu$, $\text{Re}$ | `run_simulation.py` | `nu`, `Re` |
| Fields $u$, $v$, $p$ | `solver/fields.py` | `Fields.u`, `Fields.v`, `Fields.p` |
| $x$-momentum equation | `solver/momentum.py` | `solve_u_star()` |
| $y$-momentum equation | `solver/momentum.py` | `solve_v_star()` |
| Continuity residual | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow()` → `fields.bP` |
