# 04 — Pressure–Velocity Coupling

## The Problem

The incompressible Navier–Stokes system has **three unknowns** (u, v, p)
and **three equations** (continuity, x-momentum, y-momentum).

The issue: **pressure has no independent equation**.

- Momentum equations contain ∇p as a forcing term.
- Continuity constrains ∇·u = 0 but says nothing directly about p.

For compressible flow, pressure comes from the equation of state (ρ = ρ(p)).
For incompressible flow, that link is broken — pressure is a **Lagrange
multiplier** that enforces the divergence-free constraint.

---

## The Collocated Grid Problem: Checkerboarding

On a **collocated grid** (u, v, p at the same locations), the discrete
pressure gradient in the momentum equation couples **alternating** cells.

For the central-difference approximation:

```
∂p/∂x |_P ≈ (p[i+1] - p[i-1]) / (2 dx)
```

Cell (i) is coupled to cells (i-1) and (i+1) but **not** to cell (i).
This means an alternating pressure field like:

```
high - low - high - low - ...
```

produces **zero** pressure gradient in the momentum equation!

Such a "checkerboard" pressure field would satisfy the discrete equations
even though it is physically meaningless.

---

## Staggered Grid (Traditional Solution)

The classic fix is the **staggered grid**: store u at east/west face centres,
v at north/south face centres, and p at cell centres.

Then the pressure gradient uses directly adjacent pressure values:

```
∂p/∂x |_e ≈ (p_E - p_P) / dx    ← adjacent cells, no skipping!
```

This eliminates checkerboarding but complicates the code significantly.

---

## Collocated Grid + Rhie–Chow (Modern Approach)

We keep everything at cell centres (**simpler code**) but use a special
interpolation (Rhie-Chow, Chapter 7) when computing face velocities for
the continuity equation.

The Rhie-Chow face velocity uses a **compact** pressure gradient:

```
u_e = avg(u_P, u_E) - avg(dy/a_P) * (p_E - p_P)
```

The compact term `(p_E - p_P)` couples **adjacent** cells — it sees the
checkerboard where the momentum equation does not.

When the checkerboard appears, `(p_E - p_P)` is large → Rhie-Chow velocity
is modified → continuity is violated → pressure correction eliminates it.

---

## The SIMPLE Strategy

Given this coupling problem, SIMPLE (Chapter 5) uses:

1. **Predict** velocities from momentum (using current p, possibly wrong)
2. **Derive** how much p must change to satisfy continuity
3. **Correct** both p and velocities
4. **Repeat** until continuity residual → 0

The pressure correction p' satisfies a Poisson-type equation derived in
Chapter 5.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Wide pressure gradient (momentum) | `solver/momentum.py` | `rhs` term |
| Compact gradient (Rhie-Chow) | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow` |
| SIMPLE loop | `solver/simple.py` | `run_simple` |
