# 06 — Gauss–Seidel Linear Solver

## The Problem: Solving the FVM Linear System

After discretisation, every interior cell $(i,j)$ contributes one equation:

$$a_P\,\phi_P = a_E\,\phi_E + a_W\,\phi_W + a_N\,\phi_N + a_S\,\phi_S + b$$

For an $N \times N$ grid, this is a system of $(N-2)^2$ equations (interior nodes only). Writing it in matrix form:

$$\mathbf{A}\,\boldsymbol{\phi} = \mathbf{b}$$

where $\mathbf{A}$ is a sparse banded matrix. For a 41×41 grid, $\mathbf{A}$ is $1521 \times 1521$ — storing it explicitly as a dense matrix is wasteful since each row has at most 5 non-zero entries. A direct solver like Gaussian elimination would work but is overkill for this structure.

---

## The Gauss–Seidel Method

### The Idea

Instead of solving the full system simultaneously, **visit each cell one at a time** and update its value using the latest available values from its neighbours. No matrix is stored — just the five-point stencil coefficients.

### The Update Formula

Rearrange the FVM equation to isolate $\phi_P$:

$$\phi_P = \frac{a_E\,\phi_E + a_W\,\phi_W + a_N\,\phi_N + a_S\,\phi_S + b}{a_P}$$

Apply this update in-place as we sweep through the grid.

### Matrix Splitting Perspective

Decompose $\mathbf{A} = \mathbf{D} + \mathbf{L} + \mathbf{U}$ where $\mathbf{D}$ is diagonal, $\mathbf{L}$ is strictly lower triangular, and $\mathbf{U}$ is strictly upper triangular:

- **Jacobi** iteration: $\boldsymbol{\phi}^{(k+1)} = \mathbf{D}^{-1}(\mathbf{b} - (\mathbf{L} + \mathbf{U})\boldsymbol{\phi}^{(k)})$ — uses only old values
- **Gauss–Seidel**: $\boldsymbol{\phi}^{(k+1)} = (\mathbf{D} + \mathbf{L})^{-1}(\mathbf{b} - \mathbf{U}\,\boldsymbol{\phi}^{(k)})$ — uses already-updated values from earlier in the sweep

The difference is that Gauss–Seidel **immediately uses** any value it has already updated in the current sweep. This propagates information faster and roughly **doubles the convergence rate** compared to Jacobi.

---

## Sweep Ordering and In-Place Updates

We sweep in natural (lexicographic) order: $i = 1, \ldots, n_x-2$, $j = 1, \ldots, n_y-2$:

```python
for sweep in range(n_sweeps):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            phi[i,j] = (aE[i,j] * phi[i+1, j]    # ← still OLD (not visited yet)
                       + aW[i,j] * phi[i-1, j]    # ← already NEW (visited this sweep)
                       + aN[i,j] * phi[i, j+1]    # ← still OLD
                       + aS[i,j] * phi[i, j-1]    # ← already NEW
                       + b[i,j]) / aP[i,j]
```

When we reach $(i,j)$:
- $(i-1,j)$ and $(i,j-1)$ have already been updated → **new** values
- $(i+1,j)$ and $(i,j+1)$ have not been updated yet → **old** values

This asymmetry is what distinguishes Gauss–Seidel from Jacobi and accelerates convergence.

---

## Convergence

### Sufficient Condition: Diagonal Dominance

Gauss–Seidel converges if $\mathbf{A}$ is **diagonally dominant**:

$$|a_P| \;\geq\; |a_E| + |a_W| + |a_N| + |a_S|$$

with strict inequality for at least one row. Our momentum equations satisfy this by construction:

$$a_P = a_E + a_W + a_N + a_S + \underbrace{(F_e - F_w + F_n - F_s)}_{\geq\, 0}$$

The net outflow term is non-negative during iteration, ensuring $a_P \geq \sum a_{nb}$.

### Convergence Rate

Gauss–Seidel converges geometrically: the error after $k$ sweeps satisfies $\|\mathbf{e}^{(k)}\| \leq \rho^{k} \|\mathbf{e}^{(0)}\|$ where $\rho < 1$ is the spectral radius of the iteration matrix. For the five-point Laplacian on an $N \times N$ grid:

$$\rho \approx 1 - \frac{\pi^2}{N^2}$$

This means convergence slows on finer grids (more sweeps needed). Advanced methods like multigrid address this, but for our 41×41 grid, Gauss–Seidel is adequate.

---

## Usage Within SIMPLE

The same `gauss_seidel()` function is called in two contexts with very different numbers of sweeps:

### 1. Momentum Equations (5–20 sweeps)

```python
for _ in range(gs_sweeps):
    gauss_seidel(u_star, aP, aE, aW, aN, aS, b, n_sweeps=1)
    apply_velocity_bcs(u_star, v_star, U_lid)
```

Boundary conditions are re-applied **after each sweep** because the lid velocity must remain enforced throughout the iterative process. We do not need full convergence of the inner solve — the outer SIMPLE loop will reconverge anyway.

### 2. Pressure-Correction Equation (20–50 sweeps)

```python
gauss_seidel(p_prime, aP_p, aE_p, aW_p, aN_p, aS_p, b_p, n_sweeps=gs_p)
```

The pressure correction needs more sweeps because its accuracy directly controls how effectively continuity is enforced. Too few sweeps → the pressure correction is too rough → SIMPLE converges slowly. Too many sweeps → wasted work, since the outer SIMPLE iteration has not advanced.

### Why the Same Function Works for Both

Both the momentum equation and the pressure-correction equation have the **identical algebraic structure**: $a_P\,\phi_P = \sum a_{nb}\,\phi_{nb} + b$. Only the coefficient values differ. The solver does not need to know which equation it is solving.

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Gauss–Seidel solver | `solver/linear_solvers.py` | `gauss_seidel()` |
| Residual computation | `solver/linear_solvers.py` | `compute_residual()` |
| Momentum solve call | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| Pressure-correction solve | `solver/pressure.py` | `solve_pressure_correction` |
