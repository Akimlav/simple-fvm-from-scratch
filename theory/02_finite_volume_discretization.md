## Idea of the Finite Volume Method

We solve equations over a **control volume** and use the Gauss divergence theorem:

\[\int_V \nabla \cdot \mathbf{F} dV = \oint_S \mathbf{F}\cdot\mathbf{n} dS\]

Derivatives become fluxes.

Control volume neighbors: E, W, N, S.

---

## Diffusion Term

\[\nabla \cdot (\Gamma \nabla \phi)\]

At east face:

\[(\nabla \phi)_e \approx (\phi_E - \phi_P)/\Delta x\]

Flux coefficient:

\[D_e = \Gamma A_e/\Delta x\]

Contribution:

\[D_e(\phi_E - \phi_P)\]

---

## Convection Term

\[\nabla \cdot (\rho \mathbf{u}\phi)\]

Face mass flux:

\[F_e = \rho u_e A_e\]

Upwind scheme:

\[\phi_e = \phi_P (F_e>0) \text{ else } \phi_E\]

---

## Final Algebraic Equation

\[a_P \phi_P = a_E \phi_E + a_W \phi_W + a_N \phi_N + a_S \phi_S + b\]

---

### Where this appears in the code

- `discretization.py`
- `linear_solvers.py`
