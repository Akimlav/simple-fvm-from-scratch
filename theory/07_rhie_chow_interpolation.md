## Rhie–Chow Interpolation

Naive face velocity interpolation causes checkerboard pressure.

Corrected face velocity:

\[u_f = \bar{u}_f - D_f(p_E - p_P)\]

\[D_f = A_f/a_P\]

This links pressure to mass flux and stabilizes SIMPLE.

---

### Where this appears in the code

- `rhie_chow.py`
- `pressure.py`
