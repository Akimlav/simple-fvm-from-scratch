# 03 — Momentum Equations

Chapter 2 derived the general FVM equation and left one question open: how to express the face values $\phi_e, \phi_w, \phi_n, \phi_s$ in terms of nodal values. This chapter answers that question using the **upwind scheme**, assembles the complete momentum coefficients, and adds the pressure source term.

---

## Starting Point

From Chapter 2, the integrated transport equation for cell $(i,j)$ is:

$$F_e\,\phi_e - F_w\,\phi_w + F_n\,\phi_n - F_s\,\phi_s = D_e(\phi_E - \phi_P) - D_w(\phi_P - \phi_W) + D_n(\phi_N - \phi_P) - D_s(\phi_P - \phi_S) + S\,\Delta V$$

We need to express each face value $\phi_f$ in terms of the values at nodes $P$ and its neighbour across that face.

---

## The Upwind Differencing Scheme

### Physical Motivation

Convection transports a quantity **in the direction of flow**. At a face between cells $P$ and $E$, if the flow is going from $P$ to $E$ (i.e. $F_e > 0$), the fluid arriving at the face carries the value from $P$, not from $E$. The **upwind scheme** captures this:

$$\phi_e = \begin{cases} \phi_P & \text{if } F_e > 0 \quad\text{(flow goes P → E, upwind is P)} \\ \phi_E & \text{if } F_e < 0 \quad\text{(flow goes E → P, upwind is E)} \end{cases}$$

### Algebraic Form

This conditional can be written without branching:

$$F_e\,\phi_e = \max(F_e,\,0)\;\phi_P \;+\; \min(F_e,\,0)\;\phi_E$$

Equivalently (using $\min(F_e, 0) = -\max(-F_e, 0)$):

$$F_e\,\phi_e = \max(F_e,\,0)\;\phi_P - \max(-F_e,\,0)\;\phi_E$$

Applying the same logic to all four faces:

| Face | $F_f\,\phi_f$ |
|---|---|
| East | $\max(F_e, 0)\,\phi_P - \max(-F_e, 0)\,\phi_E$ |
| West | $\max(F_w, 0)\,\phi_W - \max(-F_w, 0)\,\phi_P$ |
| North | $\max(F_n, 0)\,\phi_P - \max(-F_n, 0)\,\phi_N$ |
| South | $\max(F_s, 0)\,\phi_S - \max(-F_s, 0)\,\phi_P$ |

---

## Assembling the Coefficients

### Deriving the Neighbour Coefficients

Substitute the upwind expressions into the integrated equation and **collect terms** by grouping all contributions to each nodal value.

**Coefficient of $\phi_E$** (from convection + diffusion through the east face):

$$a_E = D_e + \max(-F_e,\; 0)$$

- $D_e$: diffusion always couples $P$ and $E$
- $\max(-F_e, 0)$: convection only couples $E$ to $P$ when flow goes $E \to P$ (i.e. $F_e < 0$)

By identical reasoning for the other faces:

$$\boxed{a_E = D_e + \max(-F_e,\; 0)}$$

$$\boxed{a_W = D_w + \max(F_w,\; 0)}$$

$$\boxed{a_N = D_n + \max(-F_n,\; 0)}$$

$$\boxed{a_S = D_s + \max(F_s,\; 0)}$$

### Deriving the Central Coefficient

Collecting all contributions to $\phi_P$:

$$a_P = \underbrace{a_E + a_W + a_N + a_S}_{\text{sum of neighbours}} + \underbrace{(F_e - F_w + F_n - F_s)}_{\text{net mass outflow}}$$

**Why the extra term?** The net mass outflow $(F_e - F_w + F_n - F_s)$ is zero at convergence (continuity is satisfied), but during iteration it is generally non-zero. Including it guarantees **diagonal dominance** ($a_P \geq a_E + a_W + a_N + a_S$), which is essential for the Gauss–Seidel solver to converge (Chapter 6).

**Proof of diagonal dominance:** Each $a_{nb} = D_f + \max(\pm F_f, 0) \geq 0$ and the net outflow term is non-negative when continuity is not yet fully satisfied, so $a_P \geq \sum a_{nb} \geq 0$.

---

## Mass Fluxes at Cell Faces

The mass fluxes $F_e, F_w, F_n, F_s$ require the velocity **at the face**, which lies between two nodes. We use **linear interpolation** of the current velocity field:

$$u_e = \tfrac{1}{2}(u_P + u_E), \qquad v_n = \tfrac{1}{2}(v_P + v_N)$$

For the $x$-momentum equation, the mass fluxes through each face are:

$$F_e = \rho\,\Delta y \cdot \tfrac{1}{2}(u_{i,j} + u_{i+1,j})$$

$$F_w = \rho\,\Delta y \cdot \tfrac{1}{2}(u_{i-1,j} + u_{i,j})$$

$$F_n = \rho\,\Delta x \cdot \tfrac{1}{2}(v_{i,j} + v_{i,j+1})$$

$$F_s = \rho\,\Delta x \cdot \tfrac{1}{2}(v_{i,j-1} + v_{i,j})$$

These are computed from the **current** velocity field at the start of each SIMPLE iteration and are held fixed while solving the momentum equation (this is the "semi-implicit" in SIMPLE).

---

## Pressure Gradient Source Term

The pressure gradient $-\partial p / \partial x$ is a source term for the $x$-momentum equation. Integrating over the cell volume:

$$\int_V \left(-\frac{\partial p}{\partial x}\right) dV \approx -\frac{\partial p}{\partial x}\bigg|_P \cdot \Delta x\,\Delta y$$

On a **collocated grid**, the pressure gradient at node $P$ is approximated by a central difference across two neighbours:

$$\frac{\partial p}{\partial x}\bigg|_P \approx \frac{p_E - p_W}{2\,\Delta x} = \frac{p_{i+1,j} - p_{i-1,j}}{2\,\Delta x}$$

Substituting:

$$b_{\text{pressure}} = -\frac{p_{i+1,j} - p_{i-1,j}}{2\,\Delta x} \cdot \Delta x\,\Delta y = -\frac{\Delta y}{2}\,(p_{i+1,j} - p_{i-1,j})$$

> **Note:** this wide-stencil gradient $(p_{i+1} - p_{i-1})$ skips the immediate neighbour — it is one source of the **checkerboard instability** (Chapter 4). Rhie–Chow interpolation (Chapter 7) repairs continuity without changing the momentum equation itself.

---

## Under-Relaxation

### Why It Is Needed

SIMPLE is an iterative scheme. At each iteration, the momentum equations are solved with a pressure field that may be far from the true solution. Direct substitution of the newly computed $u$ often **overshoots**, causing divergence. Under-relaxation slows down the update:

$$u_P^{\text{new}} = u_P^{\text{old}} + \alpha_u\,(u_P^{\text{direct}} - u_P^{\text{old}})$$

where $\alpha_u \in (0, 1)$ is the under-relaxation factor ($\alpha_u = 1$ means no relaxation; $\alpha_u = 0.5$ means take half the step).

### Equivalent Modified Equation

Under-relaxation can be built directly into the coefficients. Starting from $a_P\,u_P^{\text{direct}} = \sum a_{nb}\,u_{nb} + b$ and substituting the relaxation formula:

$$\frac{a_P}{\alpha_u}\,u_P = \sum a_{nb}\,u_{nb} + b + \frac{1 - \alpha_u}{\alpha_u}\,a_P\,u_P^{\text{old}}$$

The modified central coefficient is $a_P / \alpha_u$ (larger → more diagonally dominant → more stable), and an extra source $\frac{1-\alpha_u}{\alpha_u}\,a_P\,u_P^{\text{old}}$ anchors the solution to the previous iterate.

---

## Complete $u$-Momentum for Cell $(i,j)$

Putting it all together:

$$\boxed{a_P^u\,u_P^* = a_E^u\,u_E^* + a_W^u\,u_W^* + a_N^u\,u_N^* + a_S^u\,u_S^* - \frac{\Delta y}{2}(p_E - p_W) + \frac{1-\alpha_u}{\alpha_u}\,a_P^{u,0}\,u_P^{\text{old}}}$$

where $a_P^{u,0}$ is the unrelaxed central coefficient and $a_P^u = a_P^{u,0} / \alpha_u$.

In code, this is equivalent to:

```python
# Compute direct solution
u_direct = rhs / au_P_unrelaxed

# Apply under-relaxation
u_star[i,j] = u[i,j] + urf_u * (u_direct - u[i,j])
```

The **unrelaxed** $a_P$ is stored in `au_P_arr[i,j]` — it is reused in the pressure-correction equation (Chapter 5) and Rhie–Chow interpolation (Chapter 7).

### The $v$-Momentum Equation

Identical structure with $u \to v$ and $x \leftrightarrow y$:

- Pressure source: $b_{\text{pressure}} = -\frac{\Delta x}{2}(p_{i,j+1} - p_{i,j-1})$
- Diffusion: $D_e = D_w = \mu\,\Delta y / \Delta x$, $D_n = D_s = \mu\,\Delta x / \Delta y$ (same values)
- Mass fluxes: same as for $u$

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Neighbour coefficients $a_E, a_W, a_N, a_S$ | `solver/discretization.py` | `neighbour_coeffs()` |
| Central coefficient $a_P$ | `solver/discretization.py` | `central_coeff()` |
| Diffusion conductances | `solver/discretization.py` | `diffusion_coeffs()` |
| Mass fluxes | `solver/discretization.py` | `convective_mass_fluxes()` |
| $u$-momentum solve | `solver/momentum.py` | `solve_u_star()` |
| $v$-momentum solve | `solver/momentum.py` | `solve_v_star()` |
| Under-relaxation | `solver/momentum.py` | `u_star[i,j] = u[i,j] + urf_u * (...)` |
| Stored $a_P$ for later use | `solver/momentum.py` | `au_P_arr`, `av_P_arr` |
