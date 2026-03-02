---
math: true
---

## Discretized Momentum Equations

We integrate Navier–Stokes over each cell.

Diffusion:

$$
D_e(u_E-u_P)+D_w(u_W-u_P)+D_n(u_N-u_P)+D_s(u_S-u_P)
$$

Convection via face mass fluxes.

Pressure gradient:

$$
\int_V \partial p/\partial x \, dV \approx (p_E-p_W)A
$$

Source term:

$$
b_p = (p_W - p_E)A
$$

Final form:

$$
a_P u_P = a_E u_E + a_W u_W + a_N u_N + a_S u_S + b - (p_E-p_W)A
$$

Solve using guessed pressure to obtain predicted velocity \(u^*\).

---

### Where this appears in the code

- `momentum.py`
- `simple.py`
