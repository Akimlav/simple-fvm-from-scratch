## SIMPLE Algorithm

1. Guess pressure p*
2. Solve momentum → u*, v*
3. Compute mass imbalance
4. Solve pressure correction p'
5. Correct fields:
   p = p* + αp p'
   u = u* + d_u(pW' - pP')
   v = v* + d_v(pS' - pP')
6. Repeat until continuity residual small.

Pressure iteratively enforces continuity.

---

### Where this appears in the code

- `simple.py`
