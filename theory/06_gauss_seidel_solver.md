# 06 — Gauss–Seidel Linear Solver

## Why an Iterative Solver?

After FVM discretization, each cell (i,j) gives one equation:

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

Gauss-Seidel applies this update **in-place**: when computing φ at cell (i,j),
it immediately uses the **already-updated** values from cells (i-1,j) and (i,j-1).

This makes convergence roughly twice as fast as the Jacobi method, which uses
only values from the previous iteration.

---

## Sweep Ordering

We sweep in natural order: i = 1..nx-2, j = 1..ny-2 (interior cells only).

```
for sweep in range(n_sweeps):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            phi[i,j] = (aE * phi[i+1,j]   # uses OLD value (not yet updated)
                      + aW * phi[i-1,j]    # uses NEW value (already updated!)
                      + aN * phi[i,j+1]    # uses OLD value
                      + aS * phi[i,j-1]    # uses NEW value (already updated!)
                      + b[i,j]) / aP[i,j]
```

The key: when we reach (i,j), cells (i-1,j) and (i,j-1) were already updated
in this sweep. Cells (i+1,j) and (i,j+1) have not been updated yet — they
still hold previous-iteration values. This asymmetry is what makes G-S
converge faster than Jacobi.

---

## Convergence

Gauss-Seidel converges if the coefficient matrix is **diagonally dominant**:

```
|a_P| >= |a_E| + |a_W| + |a_N| + |a_S|
```

Our momentum equations satisfy this because a_P is constructed as the sum
of its neighbours plus the net mass flux term (which is non-negative):

```
a_P = a_E + a_W + a_N + a_S + (F_e - F_w + F_n - F_s)
```

---

## Residual Computation

The residual for cell (i,j) is how far the current iterate is from satisfying
the equation:

```
residual[i,j] = |a_P φ_P - a_E φ_E - a_W φ_W - a_N φ_N - a_S φ_S - b|
```

We track `max(residual)` or `sum(residual)` to decide when to stop.

In the SIMPLE loop, we don't check Gauss-Seidel residuals explicitly — we
use a fixed number of sweeps per SIMPLE iteration and check the **continuity
residual** (bP) for the outer convergence criterion.

---

## Implementation

```python
def gauss_seidel(phi, aP, aE, aW, aN, aS, b, n_sweeps):
    """
    Solve  aP * phi = aE*phi_E + aW*phi_W + aN*phi_N + aS*phi_S + b
    using Gauss-Seidel iteration. Updates phi in-place.

    Parameters
    ----------
    phi   : 2D array, solution field (modified in-place)
    aP    : 2D array, central coefficient
    aE,aW,aN,aS : 2D arrays, neighbour coefficients
    b     : 2D array, source term
    n_sweeps : int, number of G-S sweeps to perform
    """
    nx, ny = phi.shape
    for _ in range(n_sweeps):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi[i, j] = (
                    aE[i,j] * phi[i+1, j]
                  + aW[i,j] * phi[i-1, j]
                  + aN[i,j] * phi[i,  j+1]
                  + aS[i,j] * phi[i,  j-1]
                  + b[i,j]
                ) / aP[i, j]
    return phi
```

This exact function is called for both the momentum equations AND the
pressure-correction equation — the same solver for both problems.

---

## Number of Sweeps

- Momentum equations: typically 5–20 sweeps per SIMPLE iteration
  (we don't need to fully converge each inner solve)
- Pressure-correction: typically 20–50 sweeps
  (p' must be accurate enough to correct mass imbalance)

Too few sweeps → SIMPLE converges slowly.
Too many sweeps → computational waste (outer residual hasn't changed yet).

---

## Where This Appears in the Code

| Concept | File | Function |
|---|---|---|
| Gauss-Seidel solver | `solver/linear_solvers.py` | `gauss_seidel()` |
| Momentum solve call | `solver/momentum.py` | `solve_u_star`, `solve_v_star` |
| Pressure-correction solve | `solver/pressure.py` | `solve_pressure_correction` |
