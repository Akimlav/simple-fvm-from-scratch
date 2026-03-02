# 01 — What Problem Are We Solving?

## The Physical Setup

We solve for **steady incompressible viscous flow** inside a unit square cavity.
```
y
^
1 ── u = U_lid = 1,  v = 0 ──────   ← moving lid drives the flow
|                              |
u=0, v=0               u=0, v=0     ← left and right walls: no-slip
|         FLUID                |
|                              |
0 ── u = 0,  v = 0 ────────────     ← stationary floor: no-slip
0                              1  → x
```

The lid drags the fluid rightward at the top, creating a **recirculating
vortex** that fills most of the cavity. At Re = 100 the flow is smooth,
steady, and laminar — an ideal first solver test case.

---

## Governing Equations

We solve the **incompressible Navier–Stokes equations** in 2D steady form.

### Continuity (conservation of mass)

For an incompressible fluid (ρ = constant), mass conservation reduces to:
```
∂u/∂x  +  ∂v/∂y  =  0
```

This says: whatever mass enters a cell must also leave it. No accumulation anywhere.

### x-Momentum (Newton's second law in x)
```
ρ( u ∂u/∂x + v ∂u/∂y ) = -∂p/∂x + μ( ∂²u/∂x² + ∂²u/∂y² )
```

Left side:  inertia — the fluid carries its own momentum as it moves (convection)
Right side: pressure gradient pushes the fluid; viscosity resists velocity gradients (diffusion)

### y-Momentum (Newton's second law in y)
```
ρ( u ∂v/∂x + v ∂v/∂y ) = -∂p/∂y + μ( ∂²v/∂x² + ∂²v/∂y² )
```

Left side:  same inertia terms, now for the vertical velocity component v
Right side: same structure — y-pressure gradient and viscous diffusion

---

## Non-Dimensional Parameter: Reynolds Number
```
Re = ρ U L / μ = U L / ν
```

With U = 1 m/s, L = 1 m, ν = 0.01 m²/s → **Re = 100**.

Low Re means viscosity dominates over inertia. The flow develops one large
primary vortex and two small corner vortices at the bottom, all smooth and steady.

Ghia et al. (1982) solved this exact cavity problem on a fine 129×129 grid
and published tabulated velocity values at several Reynolds numbers. Their
Re = 100 data are the standard benchmark — we compare our centreline velocity
profiles against them at the end to validate the solver.

---

## Why is This Hard to Solve?

Three unknowns: **u**, **v**, **p**.
Three equations: continuity + x-momentum + y-momentum.

**The coupling problem**: pressure appears in the momentum equations, but
there is no standalone equation for pressure. Continuity constrains the
velocity field (∇·**u** = 0) but says nothing directly about p.

This is the **pressure–velocity coupling problem**. For compressible flow,
an equation of state (ρ = ρ(p)) closes the system. For incompressible flow,
that link is severed — pressure becomes a Lagrange multiplier that enforces
the divergence-free constraint.

SIMPLE (Chapter 5) resolves this by: guessing p, solving momentum to get
approximate velocities u* and v*, then *deriving* how much p must change to
make u* and v* satisfy continuity. The correction is applied and the loop
repeats until continuity is satisfied to within a tolerance.

---

## Where This Appears in the Code

| Concept | File | Variable / Function |
|---|---|---|
| Domain side length L | `run_simulation.py` | `L = 1.0` |
| Kinematic viscosity, Re | `run_simulation.py` | `nu`, `Re` |
| u, v, p field arrays | `solver/fields.py` | `Fields.u`, `Fields.v`, `Fields.p` |
| x-momentum equation | `solver/momentum.py` | `solve_u_star()` |
| y-momentum equation | `solver/momentum.py` | `solve_v_star()` |
| Continuity residual | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow()` → `fields.bP` |