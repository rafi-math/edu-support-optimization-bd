from pathlib import Path
import numpy as np
import pandas as pd

GRID = Path("results/phase3_fairness_grid.csv")
OUT_CSV = Path("results/phase2_policy_table_from_verified_grid.csv")
OUT_REPORT = Path("reports/phase2_policy_table_report.txt")
FIG_DIR = Path("results/figures")

# Published constants from the verified Phase 3 grid / figure (do not edit).
K = 730
TOTAL_OOS = 641
OBJ_RAW_UW = 325.663745
OBJ_PROP_YOUNG_UW = 222.022851
OBJ_RAW_W = 835.475663
OBJ_PROP_YOUNG_W = 656.653056
CAP_RAW = 332
CAP_PROP_YOUNG = 219
INTERACTION_UW_EXPECTED = -5.74

lines = []
def p(m=""):
    m = str(m); print(m); lines.append(m)

if not GRID.exists():
    raise SystemExit(f"Missing verified grid: {GRID.resolve()}")

g = pd.read_csv(GRID, low_memory=False)
need = ["objective","young_floor","girls_floor","obj_unweighted","obj_weighted",
        "observed_oos_captured","young_selected","girls_selected","girls_share"]
miss = [c for c in need if c not in g.columns]
p(f"Grid rows={len(g)}; missing columns: {miss if miss else 'none'}")
if miss:
    raise SystemExit(f"Grid missing columns: {miss}")

for c in ["young_floor","girls_floor","obj_unweighted","obj_weighted",
          "observed_oos_captured","young_selected","girls_selected","girls_share"]:
    g[c] = pd.to_numeric(g[c], errors="coerce")

# Infer K from the grid and assert it equals the published budget.
ratio = (g["girls_selected"] / g["girls_share"]).replace([np.inf,-np.inf],np.nan).dropna()
K_inferred = int(round(float(ratio.median())))
p(f"K inferred from grid (median girls_selected/girls_share) = {K_inferred}")
if K_inferred != K:
    raise SystemExit(f"K mismatch: grid implies {K_inferred}, expected {K}. Stop.")

def cell(obj, y, gr):
    r = g[(g["objective"]==obj)&(g["young_floor"]==y)&(g["girls_floor"]==gr)]
    if len(r)!=1:
        raise SystemExit(f"Expected exactly one cell for {obj} y={y} g={gr}, found {len(r)}.")
    return r.iloc[0]

uw = lambda y,gr: cell("unweighted",y,gr)
wt = lambda y,gr: cell("weighted",y,gr)

# ---- cross-checks against published baselines ----
p(""); p("CROSS-CHECKS vs published figure/grid")
checks = []
def chk(name, got, exp, tol):
    ok = abs(got-exp) <= tol
    checks.append(ok)
    p(f"  {name}: got={got:.6f} expected={exp:.6f} -> {'OK' if ok else 'FAIL'}")

chk("raw unweighted obj", uw(0,0)["obj_unweighted"], OBJ_RAW_UW, 1e-3)
chk("prop-young unweighted obj", uw(423,0)["obj_unweighted"], OBJ_PROP_YOUNG_UW, 1e-3)
chk("raw weighted obj", wt(0,0)["obj_weighted"], OBJ_RAW_W, 1e-3)
chk("prop-young weighted obj", wt(423,0)["obj_weighted"], OBJ_PROP_YOUNG_W, 1e-3)
chk("raw captured OOS", uw(0,0)["observed_oos_captured"], CAP_RAW, 0.5)
chk("prop-young captured OOS", uw(423,0)["observed_oos_captured"], CAP_PROP_YOUNG, 0.5)

# interaction (subadditivity) at proportional corner, unweighted
c_raw   = OBJ_RAW_UW - uw(0,0)["obj_unweighted"]
c_young = OBJ_RAW_UW - uw(423,0)["obj_unweighted"]
c_girls = OBJ_RAW_UW - uw(0,367)["obj_unweighted"]
c_both  = OBJ_RAW_UW - uw(423,367)["obj_unweighted"]
interaction = c_both - c_young - c_girls
chk("interaction unweighted", interaction, INTERACTION_UW_EXPECTED, 0.05)

if not all(checks):
    raise SystemExit("Cross-check FAILED. Do not publish a table that contradicts the figure.")
p("ALL CROSS-CHECKS PASSED")

# ---- build tidy policy table (unweighted = primary) ----
# Using ONLY valid grid points: young=[0, 146, 423], girls=[0, 146, 367]
scenarios = [
    ("raw_top_risk",                       0,   0),
    ("young_floor_20pct",                146,   0),
    ("young_proportional_57.9pct",       423,   0),
    ("girls_floor_40pct",                  0, 292),
    ("girls_population_parity_50.3pct",    0, 367),
    ("young_prop_plus_girls_pop_parity", 423, 367),
]

rows=[]
for name,y,gr in scenarios:
    r = uw(y,gr); rw = wt(y,gr)
    rows.append({
        "policy": name,
        "young_floor": y, "girls_floor": gr,
        "obj_unweighted": float(r["obj_unweighted"]),
        "obj_drop_vs_raw": float(OBJ_RAW_UW - r["obj_unweighted"]),
        "obj_drop_pct_of_raw": float((OBJ_RAW_UW - r["obj_unweighted"])/OBJ_RAW_UW*100),
        "observed_oos_captured": int(r["observed_oos_captured"]),
        "capture_rate_unweighted": float(r["observed_oos_captured"]/TOTAL_OOS),
        "capture_drop_vs_raw": int(CAP_RAW - r["observed_oos_captured"]),
        "young_selected": int(r["young_selected"]),
        "young_share": float(r["young_selected"]/K),
        "girls_selected": int(r["girls_selected"]),
        "girls_share": float(r["girls_selected"]/K),
        "obj_weighted_sensitivity": float(rw["obj_weighted"]),
    })

tbl = pd.DataFrame(rows)
p(""); p("POLICY TABLE (unweighted primary; weighted as sensitivity)")
p(tbl.to_string(index=False))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
tbl.to_csv(OUT_CSV, index=False)
p(""); p(f"Saved aggregate policy table: {OUT_CSV.resolve()}")

OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
p(f"Report saved: {OUT_REPORT.resolve()}")