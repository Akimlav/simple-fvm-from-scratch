## Velocity Correction

Assume:

$$
u = u^* + u'
$$
$$
v = v^* + v'
$$
$$
p = p^* + p'
$$

From momentum:

$$
u_P = u_P^* + d_u(p_W' - p_P')
$$

$$
d_u = A/a_P
$$

Insert into continuity to obtain pressure correction equation:

$$
a_P p_P' = a_E p_E' + a_W p_W' + a_N p_N' + a_S p_S' + b
$$

---

### Where this appears in the code

- `pressure.py`
- `simple.py`
