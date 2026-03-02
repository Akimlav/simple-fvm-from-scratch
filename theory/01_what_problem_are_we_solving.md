# 01 — What Problem Are We Solving?

## The Physical Setup

We solve for **steady incompressible viscous flow** inside a unit square cavity.

```
y
^
1 ──────────────────── u = U_lid = 1,  v = 0   ← moving lid
|                    |
|    FLUID DOMAIN    |  u = v = 0 on left/right walls
|                    |
0 ──────────────────── u = v = 0                ← stationary floor
0                    1  → x
```

The lid motion drives a **recirculating vortex** inside the cavity. At low
Reynolds numbers the flow is steady and laminar — ideal for testing a solver.

---

## Governing Equations

We solve the **incompressible Navier–Stokes equations** in steady form.

### Continuity (conservation of mass)

For an incompressible fluid (ρ = const), mass conservation reduces to:

```
∂u/∂x  +  ∂v/∂y  =  0
```

This says: whatever fluid enters a volume must leave it — no accumulation.

### x-Momentum (Newton's second law in x)

```
ρ( u ∂u/∂x + v ∂u/∂y ) = -∂p/∂x + μ( ∂²u/∂x² + ∂²u/∂y² )
```

Left side:  inertia (convection of momentum)  
Right side: pressure force + viscous diffusion

### y-Momentum

```
ρ( u ∂v/∂x + v ∂v/∂y ) = -∂p/∂y + μ( ∂²v/∂x² + ∂²v/∂y² )
```

---

## Non-Dimensional Parameter: Reynolds Number

```
Re = ρ U L / μ = U L / ν
```

With U = 1 m/s, L = 1 m, ν = 0.01 m²/s → **Re = 100**.

Low Re means viscosity dominates. The flow has a smooth primary vortex and
two small corner vortices. This is our validation case against Ghia et al. (1982).

---

## Why is This Hard to Solve?

Three unknowns: u, v, p.  
Three equations: continuity + 2 momentum.

**The coupling problem**: pressure appears in momentum but there is no
explicit equation for pressure. Continuity gives a constraint on velocity,
not a direct equation for p.

This is the pressure–velocity coupling problem. The SIMPLE algorithm (Chapter 5)
resolves it by deriving a pressure equation from continuity.

---

## Where This Appears in the Code

| Concept | File | Function/Variable |
|---|---|---|
| Domain size, Re | `solver/grid.py` | `L`, `nu`, `Re` |
| u, v, p arrays | `solver/fields.py` | `Fields` class |
| Governing equations | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| Continuity residual | `solver/rhie_chow.py` | `compute_mass_imbalance` |
