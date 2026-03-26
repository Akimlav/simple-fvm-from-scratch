# 06 — Gauss–Seidel Linear Solver

## Why an Iterative Solver?

After FVM discretization, each node (i,j) gives one equation:

```
a_P φ_P = a_E φ_E + a_W φ_W + a_N φ_N + a_S φ_S + b
```

For an N×N grid we have N² equations. Storing the full matrix would require
N⁴ entries — 41⁴ ≈ 2.8 million entries for a 41×41 grid, mostly zeros.

Instead, we use the **Gauss-Seidel method**: iterate the update formula
directly from the five-point stencil, no matrix required.

---

## The Update Formula

Rearrange the FVM equation to isolate φ_P:

```
φ_P = (a_E φ_E + a_W φ_W + a_N φ_N + a_S φ_S + b) / a_P
```

Gauss-Seidel applies this update **in-place**: when computing φ at node (i,j),
it immediately uses the **already-updated** values from nodes (i-1,j) and (i,j-1).

This makes convergence roughly twice as fast as the Jacobi method, which uses
only values from the previous iteration.

---

## Sweep Ordering

We sweep in natural order: i = 1..nx-2, j = 1..ny-2 (interior nodes only).

```
for sweep in range(n_sweeps):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            phi[i,j] = (aE * phi[i+1,j]   # OLD value (not updated yet this sweep)
                      + aW * phi[i-1,j]    # NEW value (already updated this sweep!)
                      + aN * phi[i,j+1]    # OLD value
                      + aS * phi[i,j-1]    # NEW value (already updated this sweep!)
                      + b[i,j]) / aP[i,j]
```

The key: when we reach (i,j), nodes (i-1,j) and (i,j-1) were already updated
in this sweep. Nodes (i+1,j) and (i,j+1) have not been updated yet.

---

## Convergence

Gauss-Seidel converges if the coefficient matrix is **diagonally dominant**:

```
|a_P| >= |a_E| + |a_W| + |a_N| + |a_S|
```

Our momentum equations satisfy this because a_P is constructed as the sum
of its neighbours plus the net mass flux term (which is non-negative):

```
a_P = a_E + a_W + a_N + a_S + (F_E - F_W + F_N - F_S)
```

---

## Where This Is Used

`gauss_seidel()` in `solver/linear_solvers.py` is called in two places:

**1. Momentum equations** (`solver/momentum.py`):

```python
# solve for u* one sweep at a time, applying BCs after each
for _ in range(gs_sweeps):
    gauss_seidel(fields.u_star, aP_arr, aE_arr, aW_arr, aN_arr, aS_arr,
                 b_arr, n_sweeps=1)
    apply_velocity_bcs(fields.u_star, fields.v_star, U_lid)
```

**2. Pressure-correction equation** (`solver/pressure.py`):

```python
# solve for p' in one batch (BCs applied inside gauss_seidel already)
gauss_seidel(fields.p_prime, aP_p, aE_p, aW_p, aN_p, aS_p,
             b_p, n_sweeps=gs_sweeps)
```

The momentum solve applies BCs after each sweep because the lid velocity
must be enforced throughout. The pressure solve applies BCs by using
the Neumann extrapolation inside `apply_pressure_neumann_bcs()`.

The same function works for both because the momentum and p' equations
have identical algebraic structure — only the coefficients differ.

---

## Number of Sweeps

- Momentum equations: typically 5–20 sweeps per SIMPLE iteration
  (we don't need to fully converge each inner solve)
- Pressure-correction: typically 20–50 sweeps
  (p' must be accurate enough to actually reduce the mass imbalance)

Too few sweeps → SIMPLE converges slowly.
Too many sweeps → wasted work (the outer residual hasn't changed yet).

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Gauss-Seidel solver | `solver/linear_solvers.py` | `gauss_seidel()` |
| Momentum solve call | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| Pressure-correction solve | `solver/pressure.py` | `solve_pressure_correction` |
