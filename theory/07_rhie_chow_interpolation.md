# 07 — Rhie–Chow Interpolation

## The Problem Recap

Chapter 4 showed that naive face interpolation $u_e = \frac{1}{2}(u_P + u_E)$ on a collocated grid produces a wide-stencil continuity equation that cannot detect checkerboard pressure. This chapter derives the Rhie–Chow correction that fixes it.

---

## Deriving the Rhie–Chow Face Velocity

### Starting Point: The Momentum Equation at Cell Centres

The discretised $x$-momentum at cell $P$ can be written as:

$$u_P = \frac{\sum a_{nb}\,u_{nb} + b^{u}}{a_P} - \frac{\Delta y}{a_P}\,\frac{p_E - p_W}{2}$$

Define the "momentum without pressure" part as:

$$\hat{u}_P = \frac{\sum a_{nb}\,u_{nb} + b^{u}}{a_P}$$

so that $u_P = \hat{u}_P - \frac{\Delta y}{a_P}\,\frac{p_E - p_W}{2}$.

### Step 1: Naive Interpolation

Linear interpolation to the east face gives:

$$u_e^{\text{naive}} = \tfrac{1}{2}(u_P + u_E) = \tfrac{1}{2}(\hat{u}_P + \hat{u}_E) - \tfrac{1}{2}\left(\frac{\Delta y}{a_P^{P}}\,\frac{p_E - p_W}{2} + \frac{\Delta y}{a_P^{E}}\,\frac{p_{EE} - p_P}{2}\right)$$

The pressure gradient terms here involve $p_W$, $p_P$, $p_E$, and $p_{EE}$ — a **wide stencil** that skips cells and allows checkerboarding.

### Step 2: What the Face Velocity *Should* Look Like

If the momentum equation were written directly at the east face (as on a staggered grid), the pressure gradient would use the **compact** stencil:

$$u_e^{\text{desired}} = \hat{u}_e - \overline{\left(\frac{\Delta y}{a_P}\right)}_e \frac{p_E - p_P}{\Delta x}\,\Delta x = \hat{u}_e - D_f^{e}\,(p_E - p_P)$$

where:

$$D_f^{e} = \tfrac{1}{2}\left(\frac{\Delta y}{a_P^{P}} + \frac{\Delta y}{a_P^{E}}\right) = \tfrac{1}{2}\,\Delta y\left(\frac{1}{a_P^{P}} + \frac{1}{a_P^{E}}\right)$$

This uses $(p_E - p_P)$: adjacent cells only, no skipping.

### Step 3: The Rhie–Chow Correction

We cannot compute $\hat{u}_e$ directly (we don't have the neighbour-sum at the face). Instead, **approximate** $\hat{u}_e$ by interpolating the full cell-centre velocities $u^{*}$ and separately adding back the compact pressure gradient:

$$\boxed{u_e = \tfrac{1}{2}(u_P^{*} + u_E^{*}) - D_f^{e}\,(p_E - p_P)}$$

The first term is the naive interpolation of **predicted** velocities (which already contain the wide-stencil pressure gradient). The second term replaces that implicit wide pressure coupling with a compact one. The net effect: the face velocity responds to $(p_E - p_P)$, which **sees every cell** and kills the checkerboard.

---

## Physical Meaning of $D_f$

From the momentum equation, a small pressure change $\delta p$ produces a velocity change:

$$a_P\,\delta u = -\Delta y\,\delta p \quad\Longrightarrow\quad \delta u = -\frac{\Delta y}{a_P}\,\delta p$$

So $\Delta y / a_P$ is the **velocity sensitivity to pressure** at a cell centre. At the face, we average the sensitivities from both sides:

$$D_f^{e} = \tfrac{1}{2}\,\Delta y\left(\frac{1}{a_P^{P}} + \frac{1}{a_P^{E}}\right)$$

Where $a_P$ is large (strong convection or diffusion), $D_f$ is small — the velocity barely responds to pressure changes. Where $a_P$ is small, $D_f$ is large — pressure has a strong effect. This is physically correct.

---

## All Four Face Velocities

**East face** (between $P = (i,j)$ and $E = (i+1,j)$):

$$u_e = \tfrac{1}{2}(u^{*}_{i,j} + u^{*}_{i+1,j}) - \tfrac{1}{2}\,\Delta y\left(\frac{1}{a_P^{i,j}} + \frac{1}{a_P^{i+1,j}}\right)(p_{i+1,j} - p_{i,j})$$

$$F_e = \rho\,u_e\,\Delta y$$

**West face** (between $W = (i-1,j)$ and $P = (i,j)$):

$$u_w = \tfrac{1}{2}(u^{*}_{i-1,j} + u^{*}_{i,j}) - \tfrac{1}{2}\,\Delta y\left(\frac{1}{a_P^{i-1,j}} + \frac{1}{a_P^{i,j}}\right)(p_{i,j} - p_{i-1,j})$$

$$F_w = \rho\,u_w\,\Delta y$$

**North face** (between $P = (i,j)$ and $N = (i,j+1)$):

$$v_n = \tfrac{1}{2}(v^{*}_{i,j} + v^{*}_{i,j+1}) - \tfrac{1}{2}\,\Delta x\left(\frac{1}{a_P^{i,j}} + \frac{1}{a_P^{i,j+1}}\right)(p_{i,j+1} - p_{i,j})$$

$$F_n = \rho\,v_n\,\Delta x$$

**South face** (between $S = (i,j-1)$ and $P = (i,j)$):

$$v_s = \tfrac{1}{2}(v^{*}_{i,j-1} + v^{*}_{i,j}) - \tfrac{1}{2}\,\Delta x\left(\frac{1}{a_P^{i,j-1}} + \frac{1}{a_P^{i,j}}\right)(p_{i,j} - p_{i,j-1})$$

$$F_s = \rho\,v_s\,\Delta x$$

The **mass imbalance** (continuity residual) for cell $(i,j)$:

$$b_P = F_e - F_w + F_n - F_s$$

---

## Consistency with the Pressure-Correction Equation

The $p'$ equation coefficients from Chapter 5 are:

$$a_E' = \rho\,\Delta y\,D_f^{e} = \tfrac{1}{2}\,\rho\,\Delta y^2\left(\frac{1}{a_P^{i,j}} + \frac{1}{a_P^{i+1,j}}\right)$$

This is **exactly** $\rho\,\Delta y$ times the $D_f^{e}$ from the Rhie–Chow formula. This consistency is not a coincidence — it is **required**:

- The $p'$ equation says: "this much $p'$ change will produce this much face velocity change, which will reduce $b_P$ by this much."
- The Rhie–Chow formula uses the same $D_f$ to compute how pressure affects face velocity.
- If the two were inconsistent, the pressure correction would over- or under-correct, and SIMPLE would not converge.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Rhie–Chow face velocities | `solver/rhie_chow.py` | `compute_face_velocity_rhie_chow` |
| Mass imbalance $b_P$ | `solver/rhie_chow.py` | computed inside the same function |
| $p'$ coefficients (consistent) | `solver/pressure.py` | `build_pressure_correction_coeffs` |
| Stored $a_P$ for $D_f$ | `solver/momentum.py` | `au_P_arr`, `av_P_arr` |
