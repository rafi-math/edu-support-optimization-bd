# src/_probe_solvers.py -- throwaway: find which PuLP 4.x solver runs in-process here.
import inspect, pulp
from pulp import LpProblem, LpMaximize

def micro():
    m = LpProblem("micro", LpMaximize)
    x = m.add_variable("x", cat="Binary")
    y = m.add_variable("y", cat="Binary")
    m += 3 * x + 2 * y          # known optimum = 5 at x=y=1
    m += x + y <= 4
    return m, x, y

def run(m, x, y, solver):
    m.solve() if solver is None else m.solve(solver)
    return 3 * (x.value() or 0) + 2 * (y.value() or 0)

rows = []
# candidate 0: bare default, no solver object at all (simplest if it works)
try:
    m, x, y = micro(); o = run(m, x, y, None)
    rows.append(("DEFAULT(no-arg)", "{}", o, "PASS" if abs(o - 5) < 1e-6 else f"obj={o}"))
except Exception as e:
    rows.append(("DEFAULT(no-arg)", "{}", None, f"ERR {type(e).__name__}: {str(e)[:90]}"))

# candidates 1..n: explicit classes, in-process preferred; try a few ctor arg sets
for nm in ["HiGHS", "HiGHS_CMD", "PULP_CBC_CMD", "COIN_CMD", "GLPK_CMD", "SCIP", "XPRESS"]:
    cls = getattr(pulp, nm, None)
    if cls is None:
        continue
    for args in ({}, {"msg": 0}, {"msg": False}):
        try:
            s = cls(**args)
            m, x, y = micro(); o = run(m, x, y, s)
            ok = abs(o - 5) < 1e-6
            rows.append((nm, str(args), o, "PASS" if ok else f"obj={o}"))
            if ok:
                break           # one working arg-set per class is enough
        except Exception as e:
            rows.append((nm, str(args), None, f"ERR {type(e).__name__}: {str(e)[:90]}"))

print(f"=== solver discovery (PuLP {pulp.__version__}) ===")
for nm, a, o, st in rows:
    print(f"{nm:16s} args={a:14s} obj={str(o):>6s} -> {st}")
print("\nUse the FIRST PASS line. If DEFAULT passes, prefer it (no class name to depend on).")