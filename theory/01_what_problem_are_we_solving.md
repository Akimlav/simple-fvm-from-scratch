## The Physical Problem

We want to compute the steady 2-D flow of an incompressible Newtonian fluid inside a square cavity where the **top wall moves with constant velocity**.

This is the classical **lid-driven cavity** benchmark.

Because it contains *all* CFD difficulties simultaneously:

- pressure–velocity coupling
- wall boundary layers
- recirculation
- elliptic pressure equation
- no analytical solution

---

## Governing Equations

We solve the **steady incompressible Navier–Stokes equations**.

### Continuity (mass conservation)

$$
\nabla \cdot \mathbf{u} = 0
$$

In 2D:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

Pressure has no evolution equation.

---

### Momentum Equations

**x-momentum**

$$
\begin{aligned}
\rho \Big(
u \frac{\partial u}{\partial x}
+ v \frac{\partial u}{\partial y}
\Big)
&= -\frac{\partial p}{\partial x} \\
&\quad + \mu \Big(
\frac{\partial^2 u}{\partial x^2}
+ \frac{\partial^2 u}{\partial y^2}
\Big)
\end{aligned}
$$

**y-momentum**

$$
\begin{aligned}
\rho \Big(
u \frac{\partial v}{\partial x}
+ v \frac{\partial v}{\partial y}
\Big)
&= -\frac{\partial p}{\partial y} \\
&\quad + \mu \Big(
\frac{\partial^2 v}{\partial x^2}
+ \frac{\partial^2 v}{\partial y^2}
\Big)
\end{aligned}
$$

---

## Why Pressure Is Difficult

Momentum needs pressure. Continuity does not compute pressure.

Pressure acts as a **Lagrange multiplier** enforcing

$$
\nabla \cdot \mathbf{u} = 0
$$

---

### Where this appears in the code

- `momentum.py`
- `pressure.py`
- `simple.py`