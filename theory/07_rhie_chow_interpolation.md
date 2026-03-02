# 07 — Rhie–Chow Interpolation

## The Checkerboard Problem

On a collocated grid, the pressure gradient in the u-momentum equation is:

```
∂p/∂x |_P ≈ (p[i+1,j] - p[i-1,j]) / (2 dx)
```

This is a **wide stencil** — cell P couples to cells i+1 and i-1, skipping
its immediate neighbour. Consider this pressure field:

```
index:   0    1    2    3    4
p:      100   0  100   0  100
```

The gradient at index 2: (p[3] - p[1]) / (2dx) = (0 - 0) / (2dx) = **0**

The momentum equation sees **zero pressure gradient** even though the pressure
field is wildly oscillating! This means the checkerboard pressure satisfies
the discrete momentum equations — physically wrong.

Now consider the continuity equation with naive face interpolation:

```
u_e = (u_P + u_E) / 2
```

Substituting into ∂u/∂x + ∂v/∂y = 0 for a 1D example:

```
(u_e - u_w) / dx = ((u[i+1] + u[i])/2 - (u[i] + u[i-1])/2) / dx
                 = (u[i+1] - u[i-1]) / (2 dx)
```

Again a wide stencil — coupling alternating cells. The checkerboard velocity
field also satisfies the discrete continuity equation. Both equations have
the same flaw.

---

## Rhie–Chow: The Fix

Rhie & Chow (1983) proposed correcting the face velocity by adding a term
that introduces a **compact** (adjacent-cell) pressure coupling:

```
u_e = avg(u*_P, u*_E)  -  D_f * (p_E - p_P)
```

Where:
```
D_f = 0.5 * (dy/a_P[i,j] + dy/a_P[i+1,j])
```

Let's understand each part:

**Term 1: `avg(u*_P, u*_E)`**
The standard linear interpolation — same as naive approach.

**Term 2: `-D_f * (p_E - p_P)`**
A pressure-smoothing correction.
- Uses `(p_E - p_P)` — **compact gradient**, adjacent cells only
- The coefficient `D_f = dy/a_P` comes from the momentum equation structure:
  `u' = -(dy/a_P) * p'` (see Chapter 5)

---

## Why Does D_f = dy/a_P?

From the momentum equation derivation:

```
a_P * u_P = Σ a_nb u_nb + b - dy * (p_E - p_W) / 2
```

If we imagine applying a small pressure correction δp, the velocity response is:

```
a_P * δu = -dy * δp  →  δu = -(dy/a_P) * δp
```

So `dy/a_P` is the **velocity sensitivity to pressure change** at cell P.
Rhie-Chow uses this physical quantity as the face interpolation correction.

At the east face (between P and E), we average the sensitivities:

```
D_f = 0.5 * (dy/a_P[i,j] + dy/a_P[i+1,j])
    = 0.5 * dy * (1/a_P[i,j] + 1/a_P[i+1,j])
```

---

## Full Rhie–Chow Face Velocity Formulas

**East face** (between cell P=(i,j) and E=(i+1,j)):
```
u_e = 0.5*(u*[i,j] + u*[i+1,j])
    - 0.5*dy * (1/aP[i,j] + 1/aP[i+1,j]) * (p[i+1,j] - p[i,j])
F_e = ρ * u_e * dy
```

**West face** (between W=(i-1,j) and P=(i,j)):
```
u_w = 0.5*(u*[i-1,j] + u*[i,j])
    - 0.5*dy * (1/aP[i-1,j] + 1/aP[i,j]) * (p[i,j] - p[i-1,j])
F_w = ρ * u_w * dy
```

**North face** (between P=(i,j) and N=(i,j+1)):
```
v_n = 0.5*(v*[i,j] + v*[i,j+1])
    - 0.5*dx * (1/aP[i,j] + 1/aP[i,j+1]) * (p[i,j+1] - p[i,j])
F_n = ρ * v_n * dx
```

**South face** (between S=(i,j-1) and P=(i,j)):
```
v_s = 0.5*(v*[i,j-1] + v*[i,j])
    - 0.5*dx * (1/aP[i,j-1] + 1/aP[i,j]) * (p[i,j] - p[i,j-1])
F_s = ρ * v_s * dx
```

**Mass imbalance (continuity residual):**
```
bP[i,j] = F_e - F_w + F_n - F_s
```

---

## Consistency Requirement

The pressure-correction equation coefficients (Chapter 5) are:

```
a_E' = ρ dy² * 0.5 * (1/a_P[i,j] + 1/a_P[i+1,j])
```

This is **exactly** the factor multiplying `(p_E - p_P)` in the Rhie-Chow
east face velocity. This consistency is critical:
- The p' equation says "this much p' change will produce this much velocity change"
- The Rhie-Chow formula uses the same coefficient to relate p to velocity
- If they were inconsistent, the pressure correction would not drive bP to zero

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Rhie-Chow face velocities | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow` |
| Mass imbalance bP | `solver/rhie_chow.py` | `compute_mass_imbalance` |
| p' coefficients (consistent) | `solver/pressure.py` | `build_pressure_correction_coeffs` |
| aP storage for Rhie-Chow | `solver/momentum.py` | `au_P_arr`, `av_P_arr` |
