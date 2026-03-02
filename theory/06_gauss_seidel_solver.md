---
math: true
---

## Gauss–Seidel Linear Solver

We solve:

$$
a_P \phi_P = a_E \phi_E + a_W \phi_W + a_N \phi_N + a_S \phi_S + b
$$

Update:

$$
\phi_P = (a_E\phi_E + a_W\phi_W + a_N\phi_N + a_S\phi_S + b)/a_P
$$

Residual:

$$
R = a_P\phi_P - (a_E\phi_E + a_W\phi_W + a_N\phi_N + a_S\phi_S + b)
$$

---

### Where this appears in the code

- `linear_solvers.py`
