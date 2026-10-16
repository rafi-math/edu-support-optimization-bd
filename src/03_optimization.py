import sys
import numpy as np
import pandas as pd
import pulp
from pathlib import Path

DATA = Path("data/processed/chattogram_rural_phase2_input.csv")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def get_solver():
    return pulp.HiGHS()


SOLVER = get_solver()


def _solve_once(prob):
    return prob.solve(SOLVER) if SOLVER is not None else prob.solve()


def read_status(prob):
    try:
        return pulp.LpStatus[prob.status]
    except Exception:
        pass
    try:
        codes = {0: "Optimal", 1: "Infeasible", 2: "Unbounded",
                 3: "Infeasible or unbounded", 4: "Not solved"}
        if isinstance(prob.status, int):
            return codes.get(prob.status, f"status={prob.status}")
        return str(prob.status)
    except Exception:
        return "<unknown>"


def self_test():
    m = pulp.LpProblem("micro", pulp.LpMaximize)
    x = m.add_variable("x", cat="Binary")
    y = m.add_variable("y", cat="Binary")
    m += 3 * x + 2 * y
    m += x + y <= 4
    _solve_once(m)
    obj = 3 * (x.value() or 0) + 2 * (y.value() or 0)
    ok = abs(obj - 5.0) < 1e-6
    stat = read_status(m)
    print(f"[self-test] solver={type(SOLVER).__name__ if SOLVER else 'default'} "
          f"status={stat} objective={obj:.3f} -> {'PASS' if ok else 'FAIL'}")
    return ok


if not self_test():
    print("[self-test] FAILED: PuLP 4.x API differs from what this script assumes.")
    sys.exit(1)

df = pd.read_csv(DATA)
R = df["risk_score"].to_numpy(float)
age = df["age"].to_numpy(float)
fem = df["female"].to_numpy(float)
oos = df["Out_of_School"].to_numpy(int)
n = len(df)
yng = age < 13


def solve(n_serve, y_min=0, f_min=0):
    p = pulp.LpProblem("support_selection", pulp.LpMaximize)
    x = [p.add_variable(f"x_{i}", cat="Binary") for i in range(n)]
    p += pulp.lpSum(R[i] * x[i] for i in range(n))
    p += pulp.lpSum(x) <= n_serve
    p += pulp.lpSum(x[i] for i in range(n) if yng[i]) >= y_min
    p += pulp.lpSum(x[i] for i in range(n) if fem[i] == 1) >= f_min
    _solve_once(p)
    s = np.array([(v.value() is not None and v.value() > 0.5) for v in x])
    return dict(n_serve=n_serve, y_min=y_min, f_min=f_min,
                status=read_status(p),
                total_impact=float(np.dot(R, s)),
                oos_captured=int(oos[s].sum()),
                young_selected=int(yng[s].sum()),
                young_oos_captured=int((yng & (oos == 1))[s].sum()),
                girls_selected=int(fem[s].sum()),
                girl_share=float(fem[s].sum() / max(s.sum(), 1)),
                n_selected=int(s.sum()))


top = np.argsort(-R)[:734]
ref = float(R[top].sum())

rows = []
for y in (0, 20, 40):
    r = solve(734, y_min=y, f_min=0)
    rows.append(r)
    if y == 0:
        assert abs(r["total_impact"] - ref) < 1e-6, "MILP != top-734 at y_min=0"
        print(f"Sanity OK: y_min=0 reproduces top-734 (impact {ref:.2f}); "
              f"OOS captured = {r['oos_captured']} (compare to Phase 1's 332).")

res = pd.DataFrame(rows)
res["impact_drop_vs_y0"] = res.loc[res.y_min == 0, "total_impact"].iloc[0] - res["total_impact"]
res["drop_per_young"] = res["impact_drop_vs_y0"] / res["y_min"].where(res["y_min"] > 0)

print("\n--- Phase 2 step 1: young floor, N=734 (15% of sample) ---")
print(res[["y_min","status","total_impact","oos_captured","young_selected",
           "young_oos_captured","girls_selected","girl_share"]].to_string(index=False))
print("\nPrice of the young floor (preview of the fairness curve):")
print(res[["y_min","impact_drop_vs_y0","drop_per_young"]].to_string(index=False))

res.to_csv(RESULTS / "phase2_step1_young_floor.csv", index=False)
print(f"\nWrote aggregate results -> {RESULTS/'phase2_step1_young_floor.csv'}")