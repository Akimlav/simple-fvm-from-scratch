# 05 — The SIMPLE Algorithm

## The Big Picture

Before the derivation, here is what SIMPLE does in plain language:

> **Guess a pressure field. Use it to compute velocities. Check how much mass is being created or destroyed in each cell. Adjust the pressure to fix it. Repeat until mass is conserved everywhere.**

The name stands for **Semi-Implicit Method for Pressure-Linked Equations** (Patankar & Spalding, 1972). "Semi-implicit" because the pressure–velocity coupling is handled iteratively rather than simultaneously.

---

## Why We Need SIMPLE

After discretisation we have:
- 2 momentum equations (for $u$ and $v$), each containing $\nabla p$
- 1 continuity equation, containing $u$ and $v$ but **not** $p$

If we knew $p$ exactly, we could solve momentum directly. If we knew $u$ and $v$ exactly, we could derive $p$ from continuity. We know neither. **SIMPLE breaks this circular dependency** through a predictor–corrector iteration.

---

## Algorithm Overview

```
Initialise:  u = v = p = 0

REPEAT until convergence:
  Step 1:  Solve u-momentum with current p     →  u*    (predicted)
  Step 2:  Solve v-momentum with current p     →  v*    (predicted)
  Step 3:  Compute mass imbalance bP via Rhie–Chow face velocities
  Step 4:  Build and solve pressure-correction equation  →  p'
  Step 5:  Correct pressure:    p  ←  p + α_p · p'
  Step 6:  Correct velocities:  u  ←  u* + velocity correction from p'
  Step 7:  Check convergence:   max|bP| < tolerance?  →  STOP
```

$u^*$ and $v^*$ are called **predicted** velocities — they satisfy momentum but **not** continuity. Steps 3–6 fix the continuity violation. $b_P$ is the mass imbalance: the measure of how far the velocity field is from being divergence-free.

---

## Steps 1–2: Momentum Prediction

Solve the momentum equations from Chapter 3 with the **current** (possibly wrong) pressure field:

$$a_P^u u_P^* = \sum a_{nb}^u u_{nb}^* + b^u(p)$$

This gives predicted velocities $u^*$, $v^*$ that satisfy momentum but generally violate continuity. We store the **unrelaxed** central coefficient $a_P^{u,0}$ (before under-relaxation) in `au_P_arr` — it will be needed in every subsequent step.

---

## Step 3: Mass Imbalance

Using Rhie–Chow face velocities (Chapter 7), compute the mass flux through each face of every cell and measure the continuity residual:

$$b_P = \rho u_e \Delta y - \rho u_w \Delta y + \rho v_n \Delta x - \rho v_s \Delta x$$

This is the **net mass outflow** from cell $(i,j)$. At convergence $b_P \to 0$ everywhere.

---

## Step 4: Derivation of the Pressure-Correction Equation

This is the key derivation. We need to find a pressure correction $p'$ such that the corrected velocities satisfy continuity.

### 4a. Define Correction Fields

Decompose the true (converged) fields into predicted + correction:

$$u = u^* + u' \qquad v = v^* + v' \qquad p = p^* + p'$$

where $p^*$ is the current pressure guess and $u^*$, $v^*$ are the momentum predictions.

### 4b. Velocity Correction from Momentum

The momentum equation for the **true** velocity:

$$a_P u_P = \sum a_{nb} u_{nb} + b - \frac{\Delta y}{2}(p_E - p_W)$$

The momentum equation for the **predicted** velocity (same coefficients, same old pressure $p^*$):

$$a_P u_P^* = \sum a_{nb} u_{nb}^* + b - \frac{\Delta y}{2}(p_E^* - p_W^*)$$

Subtract:

$$a_P u_P' = \sum a_{nb} u_{nb}' - \frac{\Delta y}{2}(p_E' - p_W')$$

### 4c. The SIMPLE Approximation

The term $\sum a_{nb} u_{nb}'$ contains the velocity corrections at all neighbouring cells. SIMPLE **drops this term**, assuming that neighbour corrections are small:

$$a_P u_P' \approx -\frac{\Delta y}{2}(p_E' - p_W')$$

This is what makes SIMPLE "semi-implicit" — the neighbour coupling is neglected in the correction step. The approximation is recovered through outer iterations.

> **Note:** This approximation is why SIMPLE converges iteratively rather than in one step. Methods like SIMPLEC modify this approximation for faster convergence.

### 4d. Cell-Centre Velocity Correction

For the velocity correction applied at the cell centre, SIMPLE uses the **compact** pressure gradient (adjacent cells, not the wide stencil):

$$\boxed{u_P' = -\frac{\Delta y}{a_P^{u,0}} (p_E' - p_P')}$$

$$\boxed{v_P' = -\frac{\Delta x}{a_P^{v,0}} (p_N' - p_P')}$$

where $a_P^{u,0}$ is the **unrelaxed** momentum central coefficient. The compact gradient is used (rather than the wide stencil from momentum) for **consistency with the Rhie–Chow face velocity** — this consistency is critical for convergence.

Define the **velocity-pressure sensitivity**:

$$d_P^u = \frac{\Delta y}{a_P^{u,0}} \qquad d_P^v = \frac{\Delta x}{a_P^{v,0}}$$

### 4e. Face Velocity Correction

The Rhie–Chow face velocity (Chapter 7) after the pressure correction becomes:

$$u_e^{\text{corrected}} = u_e^{RC} - D_f^{e} (p_E' - p_P')$$

where the face coupling coefficient is:

$$D_f^{e} = \tfrac{1}{2} \Delta y\left(\frac{1}{a_P^{P}} + \frac{1}{a_P^{E}}\right)$$

Similarly for all four faces (using $a_P$ from the appropriate momentum equation).

### 4f. Substitute into Continuity

The corrected velocity field must satisfy continuity:

$$\rho u_e^{\text{corr}} \Delta y - \rho u_w^{\text{corr}} \Delta y + \rho v_n^{\text{corr}} \Delta x - \rho v_s^{\text{corr}} \Delta x = 0$$

Substitute $u_e^{\text{corr}} = u_e^{RC} - D_f^{e}(p_E' - p_P')$ and similarly for $w$, $n$, $s$:

$$\underbrace{\rho u_e^{RC} \Delta y - \rho u_w^{RC} \Delta y + \rho v_n^{RC} \Delta x - \rho v_s^{RC} \Delta x}_{=  b_P \text{(mass imbalance from Step 3)}}$$

$$-  \rho \Delta y D_f^{e} (p_E' - p_P') + \rho \Delta y D_f^{w} (p_P' - p_W') - \rho \Delta x D_f^{n} (p_N' - p_P') + \rho \Delta x D_f^{s} (p_P' - p_S') = 0$$

### 4g. Collect $p'$ Terms

Define the pressure-correction coefficients:

$$a_E' = \rho \Delta y D_f^{e} = \tfrac{1}{2} \rho \Delta y^2 \left(\frac{1}{a_P^{P}} + \frac{1}{a_P^{E}}\right)$$

$$a_W' = \rho \Delta y D_f^{w} = \tfrac{1}{2} \rho \Delta y^2 \left(\frac{1}{a_P^{W}} + \frac{1}{a_P^{P}}\right)$$

$$a_N' = \rho \Delta x D_f^{n} = \tfrac{1}{2} \rho \Delta x^2 \left(\frac{1}{a_P^{P}} + \frac{1}{a_P^{N}}\right)$$

$$a_S' = \rho \Delta x D_f^{s} = \tfrac{1}{2} \rho \Delta x^2 \left(\frac{1}{a_P^{S}} + \frac{1}{a_P^{P}}\right)$$

$$a_P' = a_E' + a_W' + a_N' + a_S'$$

The **pressure-correction equation**:

$$\boxed{a_P' p_P' = a_E' p_E' + a_W' p_W' + a_N' p_N' + a_S' p_S' - b_P}$$

This is a **Poisson-type equation** for $p'$, driven by the mass imbalance $b_P$. Where $b_P$ is large (big continuity violation), $p'$ is large (big pressure correction needed). It is solved by Gauss–Seidel (Chapter 6).

> **Key observation:** The coefficients $a_E', a_W', \ldots$ involve $1/a_P$ from momentum. Large momentum coefficients (strong convection or diffusion) mean weak velocity response to pressure → small pressure-correction coefficients → pressure correction produces small velocity changes. This is physically correct: in regions of strong flow, it is harder to redirect the velocity.

---

## Step 5: Pressure Correction

$$p  \leftarrow  p + \alpha_p \cdot p'$$

The under-relaxation factor $\alpha_p$ (typically 0.1–0.3) prevents overshooting. We also **subtract the mean** of $p'$ before applying it, to fix the pressure reference level (Chapter 8 explains why).

---

## Step 6: Velocity Correction

$$u_{i,j} = u_{i,j}^* - \frac{\Delta y}{a_P^{u,0}[i,j]} (p'_{i+1,j} - p'_{i,j})$$

$$v_{i,j} = v_{i,j}^* - \frac{\Delta x}{a_P^{v,0}[i,j]} (p'_{i,j+1} - p'_{i,j})$$

Note: these use the **compact gradient** (adjacent cells $p'_E - p'_P$), consistent with the Rhie–Chow face velocity. Using the wide stencil here would break the consistency and prevent convergence.

---

## Under-Relaxation

| Variable | Factor | Typical range | Effect of reducing |
|---|---|---|---|
| $u$, $v$ | $\alpha_u$, $\alpha_v$ | 0.3–0.7 | More stable, slower convergence |
| $p$ | $\alpha_p$ | 0.1–0.3 | More stable, slower convergence |

A useful rule of thumb (Patankar): $\alpha_u + \alpha_p \approx 1$.

Without under-relaxation, SIMPLE diverges for almost all problems. The factors slow down the update at each iteration, preventing large oscillations between successive pressure guesses.

---

## Convergence Check

The natural convergence measure is the maximum mass imbalance:

$$\text{residual} = \max_{i,j} |b_P[i,j]|$$

When this falls below the tolerance (e.g. $10^{-5}$), continuity is satisfied everywhere and the iteration stops. Early iterations may show non-monotonic residuals — this is normal while the pressure field develops from zero.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Full SIMPLE loop | `solver/simple.py` | `run_simple()` |
| Momentum prediction | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| Mass imbalance $b_P$ | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow` |
| $p'$ equation coefficients | `solver/pressure.py` | `build_pressure_correction_coeffs` |
| $p'$ solve | `solver/pressure.py` | `solve_pressure_correction` |
| Pressure update | `solver/pressure.py` | `correct_pressure` |
| Velocity correction | `solver/simple.py` | `correct_velocities` |
| $d_u = \Delta y / a_P$ | `solver/simple.py` | `dy / au_P_arr[i,j]` |
