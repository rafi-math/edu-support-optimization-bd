from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Unrecognized options detected")

from scipy.optimize import milp, LinearConstraint, Bounds

INPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
REPORT = Path("reports/phase3_girls_curve_fine.txt")
OUT_CSV = Path("results/phase3_girls_curve_fine.csv")

TOP_FRACTION = 0.15
YOUNG_AGE_MAX = 12
OBJ_TOL = 1e-9
TIME_LIMIT = 60.0

YOUNG_FLOORS = [0, 423]
GIRLS_GRID = [146, 160, 171, 172, 180, 200, 212, 213, 240, 270, 292, 320, 345, 367]

lines = []


def p(m=""):
    t = str(m)
    print(t)
    lines.append(t)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def solve_cell(obj, ymask, gmask, n, K, y_min, g_min):
    ones = np.ones((1, n), dtype=float)
    cons = [
        LinearConstraint(ones, np.array([float(K)]), np.array([float(K)])),
        LinearConstraint(ymask.reshape(1, -1).astype(float),
                         np.array([float(y_min)]), np.array([np.inf])),
        LinearConstraint(gmask.reshape(1, -1).astype(float),
                         np.array([float(g_min)]), np.array([np.inf])),
    ]
    res = milp(c=-obj, constraints=cons,
               integrality=np.ones(n, dtype=int),
               bounds=Bounds(np.zeros(n), np.ones(n)),
               options={"time_limit": TIME_LIMIT,
                        "mip_rel_gap": 0.0, "mip_abs_gap": 1e-12})
    if res.x is None:
        return None, int(res.status), False
    sel = np.rint(res.x).astype(bool)
    proven = (int(res.status) == 0)
    if int(sel.sum()) != K:
        sel = np.zeros(n, dtype=bool)
        sel[np.argsort(-res.x)[:K]] = True
        proven = False
    return sel, int(res.status), proven


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT.resolve()}")

    p("=" * 70)
    p("PHASE 3 FINE GIRLS PRICE CURVE (unweighted objective only)")
    p("=" * 70)

    df = pd.read_csv(INPUT, low_memory=False)
    for c in ["age", "female", "wealth_q", "Out_of_School",
              "child_weight_fsweight", "risk_score_oof"]:
        df[c] = num(df[c])
    df = df.loc[df["child_weight_fsweight"] > 0].reset_index(drop=True)

    n = len(df)
    K = int(math.ceil(TOP_FRACTION * n))
    risk = df["risk_score_oof"].to_numpy(float)
    y = df["Out_of_School"].to_numpy(int)
    ymask = df["age"].to_numpy(float) <= YOUNG_AGE_MAX
    gmask = df["female"].to_numpy(float) == 1
    total_oos = int(y.sum())

    p(f"n={n}, K={K}, observed OOS={total_oos}")
    p(f"young floors={YOUNG_FLOORS}, girls grid={GIRLS_GRID}")
    p("objective: unweighted sum of risk_score_oof (weighted deferred to codebook check)")

    rows = []
    kink_report = []
    for yv in YOUNG_FLOORS:
        sel0, st0, pr0 = solve_cell(risk, ymask, gmask, n, K, yv, 0)
        if sel0 is None:
            p(f"young={yv}: g=0 baseline FAILED status={st0}")
            continue
        obj0 = float(risk[sel0].sum())
        natural = int((sel0 & gmask).sum())
        cap0 = int(y[sel0].sum())
        p("")
        p(f"YOUNG FLOOR = {yv}: baseline obj={obj0:.6f}, "
          f"natural girls={natural}, captured OOS={cap0}, proven={pr0}")

        first_bind = None
        viol = []
        for gv in GIRLS_GRID:
            sel, st, pr = solve_cell(risk, ymask, gmask, n, K, yv, gv)
            if sel is None:
                rows.append({"young_floor": yv, "girls_floor": gv,
                             "status": f"infeasible({st})", "proven_optimal": False})
                continue
            obj = float(risk[sel].sum())
            drop = obj0 - obj
            girls = int((sel & gmask).sum())
            cap = int(y[sel].sum())
            rows.append({
                "young_floor": yv, "girls_floor": gv, "status": "optimal",
                "proven_optimal": bool(pr), "obj_unweighted": round(obj, 6),
                "drop_vs_g0": round(drop, 6),
                "girls_selected": girls,
                "girls_share": round(girls / K, 4),
                "observed_oos_captured": cap,
                "unweighted_capture_rate": round(cap / total_oos, 4),
            })
            if gv <= natural and abs(drop) > OBJ_TOL:
                viol.append(f"g={gv}<=natural but drop={drop:.9f}>tol")
            if gv > natural and drop > OBJ_TOL and first_bind is None:
                first_bind = gv
        free_top = first_bind - 1 if first_bind is not None else max(GIRLS_GRID)
        tie_gap = (first_bind - (natural + 1)) if first_bind is not None else None
        kink_report.append({
            "young_floor": yv, "natural_girls": natural,
            "first_binding_girls": first_bind,
            "free_up_to": free_top,
            "tie_driven_free_steps": tie_gap,
            "violations": "; ".join(viol) if viol else "none",
        })
        p(f"  kink: natural={natural}, first binding g={first_bind}, "
          f"free up to g={free_top}, tie-driven extra free steps={tie_gap}")
        p(f"  integrity (drop==0 for all g<=natural): "
          f"{'PASS' if not viol else 'VIOLATIONS: ' + '; '.join(viol)}")

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_CSV, index=False)
    p("")
    p(f"Saved fine girls curve (aggregate only): {OUT_CSV.resolve()}")
    p(f"cells={len(grid)}, proven_optimal True={int(grid['proven_optimal'].astype(bool).sum())}")

    p("")
    p("FINE GIRLS PRICE CURVE, unweighted")
    p("-" * 70)
    for yv in YOUNG_FLOORS:
        d = grid[grid["young_floor"] == yv].sort_values("girls_floor")
        if len(d) == 0:
            continue
        p("")
        p(f"young_floor={yv}")
        p(d[["girls_floor", "obj_unweighted", "drop_vs_g0",
             "girls_share", "observed_oos_captured",
             "unweighted_capture_rate", "proven_optimal"]].to_string(index=False))

    p("")
    p("KINK SUMMARY")
    p("-" * 70)
    p(pd.DataFrame(kink_report).to_string(index=False))

    p("")
    p("VERDICT")
    p("-" * 70)
    all_ok = all(r["violations"] == "none" for r in kink_report)
    if all_ok and int(grid["proven_optimal"].astype(bool).sum()) == len(grid):
        p("FINE GIRLS CURVE STATUS: PASS (kink validated, all cells proven optimal)")
    elif all_ok:
        p("FINE GIRLS CURVE STATUS: PASS on kink; some cells not proven optimal (see proven_optimal)")
    else:
        p("FINE GIRLS CURVE STATUS: CHECK KINK VIOLATIONS ABOVE before using the curve")
    p("")
    p("Note: a tie_driven_free_steps > 0 means equal-objective sets with more girls")
    p("exist just above the natural count, so the floor stays free a few steps past")
    p("natural+1. That is free gender flexibility, not an error.")
    p("")
    p("SEND BACK: reports/phase3_girls_curve_fine.txt")
    p("Do not send data/processed/chattogram_rural_phase3_optimization_input.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()