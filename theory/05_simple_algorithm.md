# 05 — The SIMPLE Algorithm

## Why We Need SIMPLE

After discretization we have:
- 2 momentum equations (for u and v) containing pressure p
- 1 continuity equation containing u and v

If we knew p exactly, we could solve momentum directly.
If we knew u and v exactly, we could derive p from continuity.
We know neither. SIMPLE breaks this circular dependency iteratively.

---

## Algorithm Overview

```
Initialize: u = v = 0,  p = 0

LOOP until convergence:
  Step 1: Solve u-momentum  →  u*   (using current p)
  Step 2: Solve v-momentum  →  v*   (using current p)
  Step 3: Compute mass imbalance using Rhie-Chow face velocities
  Step 4: Solve pressure-correction equation  →  p'
  Step 5: Correct pressure:   p  ← p + α_p * p'
  Step 6: Correct velocities: u  ← u* + correction
                               v  ← v* + correction
  Check: max|bP| < tolerance?  → STOP
```

---

## Step 1 & 2: Predictor (Momentum Solve)

Solve with **current pressure** (which may be wrong):

```
a_P * u*_P = Σ a_nb * u*_nb  -  dy * (p[i+1,j] - p[i-1,j]) / 2
```

u* does NOT satisfy continuity — that's expected at this stage.
We store `a_P` values as `au_P_arr[i,j]` for use in Steps 4 and 6.

---

## Step 3: Mass Imbalance

Using Rhie-Chow face velocities (Chapter 7), compute for each cell:

```
bP[i,j] = ρ(u_e - u_w)*dy + ρ(v_n - v_s)*dx
```

This is the continuity residual. At convergence bP → 0 everywhere.

---

## Step 4: Pressure-Correction Equation

**Derivation from first principles:**

Define corrected fields:
```
u = u* + u'
p = p* + p'
```

The velocity correction comes from the momentum equation difference
(corrected minus predicted). For the u-correction:

```
a_P * u' = - dy * (p'[i+1,j] - p'[i,j])   ← compact gradient
```

Solving for u':
```
u'_P = -(dy / a_P) * (p'_E - p'_P)
```

We define:
```
d_u = dy / a_P     [m²·s/kg]
```

So:
```
u = u* - d_u * (p'_E - p'_P)
```

For v:
```
d_v = dx / a_P
v = v* - d_v * (p'_N - p'_P)
```

**Building the p' equation:**

Substitute the corrected face velocities into continuity:

```
ρ [u*_e - d_u_e(p'_E - p'_P)] dy
- ρ [u*_w - d_u_w(p'_P - p'_W)] dy
+ ρ [v*_n - d_v_n(p'_N - p'_P)] dx
- ρ [v*_s - d_v_s(p'_P - p'_S)] dx = 0
```

Collecting p' terms on left, u*/v* terms (= -bP) on right:

```
a_E' p'_E + a_W' p'_W + a_N' p'_N + a_S' p'_S - a_P' p'_P = bP
```

Where:
```
a_E' = ρ dy² * 0.5 * (1/a_P[i,j] + 1/a_P[i+1,j])
a_W' = ρ dy² * 0.5 * (1/a_P[i-1,j] + 1/a_P[i,j])
a_N' = ρ dx² * 0.5 * (1/a_P[i,j] + 1/a_P[i,j+1])
a_S' = ρ dx² * 0.5 * (1/a_P[i,j-1] + 1/a_P[i,j])
a_P' = a_E' + a_W' + a_N' + a_S'
```

This is a **Poisson equation** for p'. It is solved by Gauss-Seidel (Chapter 6).

---

## Step 5: Pressure Correction

```
p ← p + α_p * p'
```

α_p is typically 0.1–0.3. Smaller values are more stable but slower.
We subtract the mean of p' first to fix the pressure reference level.

---

## Step 6: Velocity Correction

```
u[i,j] = u*[i,j] - (dy / au_P_arr[i,j]) * (p'[i+1,j] - p'[i,j])
v[i,j] = v*[i,j] - (dx / av_P_arr[i,j]) * (p'[i,j+1] - p'[i,j])
```

Note: **compact gradient** (adjacent cells), NOT the wide stencil used in momentum.
This is consistent with the Rhie-Chow interpolation — consistency is critical.

---

## Under-Relaxation Summary

| Variable | Factor | Typical Value |
|---|---|---|
| u, v (momentum) | α_u, α_v | 0.3 – 0.7 |
| p (pressure) | α_p | 0.1 – 0.3 |

Without under-relaxation, SIMPLE diverges. The factors slow the update,
preventing large oscillations from one iteration to the next.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Full SIMPLE loop | `solver/simple.py` | `run_simple()` |
| Predictor step | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| p' equation build | `solver/pressure.py` | `build_pressure_correction_coeffs` |
| p' solve | `solver/pressure.py` | `solve_pressure_correction` |
| Velocity correction | `solver/simple.py` | `correct_velocities` |
| d_u, d_v | `solver/simple.py` | `dy/au_P_arr[i,j]` |
