import numpy as np
import pandas as pd
import pulp
from pathlib import Path

DATA = Path("data/processed/chattogram_rural_phase2_input.csv")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

SOLVER = pulp.HiGHS()

df = pd.read_csv(DATA)
R = df["risk_score"].to_numpy(float)
age = df["age"].to_numpy(float)
fem = df["female"].to_numpy(float)
oos = df["Out_of_School"].to_numpy(int)
n = len(df)
yng = age < 13


def solve(n_serve, y_min, f_min):
    p = pulp.LpProblem("support_selection", pulp.LpMaximize)
    x = [p.add_variable(f"x_{i}", cat="Binary") for i in range(n)]
    p += pulp.lpSum(R[i] * x[i] for i in range(n))
    p += pulp.lpSum(x) <= n_serve
    p += pulp.lpSum(x[i] for i in range(n) if yng[i]) >= y_min
    p += pulp.lpSum(x[i] for i in range(n) if fem[i] == 1) >= f_min
    rc = p.solve(SOLVER)
    s = np.array([(v.value() is not None and v.value() > 0.5) for v in x])
    ns = int(s.sum())
    gs = int(fem[s].sum())
    return dict(solver_rc=rc,
                feasible=bool(ns >= y_min and gs >= f_min),
                total_impact=float(np.dot(R, s)),
                oos_captured=int(oos[s].sum()),
                young_selected=int(yng[s].sum()),
                young_oos_captured=int((yng & (oos == 1))[s].sum()),
                girls_selected=gs,
                girl_share=float(gs / max(ns, 1)),
                n_selected=ns)


N = 734
young_curve = [(y, 0) for y in (0, 20, 40, 60, 80, 100)]
girl_curve = [(0, f) for f in (0, 150, 240, 279, 320, 367)]
grid = [(y, f) for y in (0, 40, 80) for f in (0, 279, 367)]
pairs = sorted(set(young_curve + girl_curve + grid))

res = {}
for (y, f) in pairs:
    r = solve(N, y, f)
    r["y_min"] = y
    r["f_min"] = f
    res[(y, f)] = r

base = res[(0, 0)]
rows = []
for (y, f) in pairs:
    r = dict(res[(y, f)])
    r["impact_drop_vs_baseline"] = base["total_impact"] - r["total_impact"]
    rows.append(r)

full = pd.DataFrame(rows)[["y_min", "f_min", "feasible", "solver_rc",
                           "total_impact", "oos_captured", "young_selected",
                           "young_oos_captured", "girls_selected", "girl_share",
                           "n_selected", "impact_drop_vs_baseline"]]
full = full.round({"total_impact": 4, "girl_share": 4, "impact_drop_vs_baseline": 4})

print(f"Baseline (y=0,f=0): impact {base['total_impact']:.4f}, OOS captured {base['oos_captured']}")
print("\n--- Full sweep grid ---")
print(full.to_string(index=False))

yc = full[full["f_min"] == 0].sort_values("y_min").copy()
yc["drop_from_prev"] = -yc["total_impact"].diff()
yc["step"] = yc["y_min"].diff()
yc["drop_per_young"] = (yc["drop_from_prev"] / yc["step"]).round(5)
print("\n--- Young price curve (f=0) ---")
print(yc[["y_min", "total_impact", "young_oos_captured", "drop_from_prev", "drop_per_young"]].to_string(index=False))

gc = full[full["y_min"] == 0].sort_values("f_min").copy()
gc["drop_from_prev"] = -gc["total_impact"].diff()
gc["step"] = gc["f_min"].diff()
gc["drop_per_girl"] = (gc["drop_from_prev"] / gc["step"]).round(5)
print("\n--- Girl price curve (y=0); 279 = OOS-pool share, 367 = population parity ---")
print(gc[["f_min", "total_impact", "girl_share", "drop_from_prev", "drop_per_girl"]].to_string(index=False))

full.to_csv(RESULTS / "phase2_fairness_sweep.csv", index=False)
print(f"\nWrote aggregate results -> {RESULTS/'phase2_fairness_sweep.csv'}")