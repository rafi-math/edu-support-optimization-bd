from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Unrecognized options detected")

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except ImportError:
    raise SystemExit("SciPy is required. Run:\npython -m pip install scipy")


INPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
REPORT = Path("reports/phase3_fairness_grid.txt")
OUT_CSV = Path("results/phase3_fairness_grid.csv")

TOP_FRACTION = 0.15
YOUNG_AGE_MAX = 12
OBJ_TOL = 1e-9
TIME_LIMIT = 60.0

YOUNG_FLOORS = [0, 73, 146, 243, 423]
GIRLS_FLOORS = [0, 146, 292, 367]
SELFCHECK_Y = [0, 73, 423]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_rate(a, b):
    if b is None or b == 0 or not np.isfinite(b):
        return np.nan
    return float(a / b)


def order_by_obj(idx, obj, uid):
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return idx
    return idx[np.lexsort((uid[idx], -obj[idx]))]


def greedy_young(obj, ymask, n, K, y_min, uid):
    sel = np.zeros(n, dtype=bool)
    young_pop = int(ymask.sum())
    if y_min > young_pop:
        return None
    if y_min > 0:
        yi = np.flatnonzero(ymask)
        sel[order_by_obj(yi, obj, uid)[:y_min]] = True
    rem = K - int(sel.sum())
    if rem > 0:
        ri = np.flatnonzero(~sel)
        sel[order_by_obj(ri, obj, uid)[:rem]] = True
    return sel


def solve_cell(obj, ymask, gmask, n, K, y_min, g_min):
    ones = np.ones((1, n), dtype=float)
    cons = [
        LinearConstraint(ones, np.array([float(K)]), np.array([float(K)])),
        LinearConstraint(ymask.reshape(1, -1).astype(float),
                         np.array([float(y_min)]), np.array([np.inf])),
        LinearConstraint(gmask.reshape(1, -1).astype(float),
                         np.array([float(g_min)]), np.array([np.inf])),
    ]
    res = milp(
        c=-obj, constraints=cons,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.0, "mip_abs_gap": 1e-12},
    )
    if res.x is None:
        return None, int(res.status), False
    sel = np.rint(res.x).astype(bool)
    proven = (int(res.status) == 0)
    if int(sel.sum()) != K:
        sel = np.zeros(n, dtype=bool)
        sel[np.argsort(-res.x)[:K]] = True
        proven = False
    return sel, int(res.status), proven


def metrics(sel, y, w, risk, wrisk, ymask, gmask, pmask, band, total_oos, total_wmass):
    n_sel = int(sel.sum())
    captured = int(y[sel].sum())
    wmass = float((w[sel] * y[sel]).sum())
    return {
        "selected_n": n_sel,
        "obj_unweighted": float(risk[sel].sum()),
        "obj_weighted": float(wrisk[sel].sum()),
        "observed_oos_captured": captured,
        "unweighted_capture_rate": safe_rate(captured, total_oos),
        "weighted_capture_rate": safe_rate(wmass, total_wmass),
        "young_selected": int((sel & ymask).sum()),
        "girls_selected": int((sel & gmask).sum()),
        "girls_share": safe_rate(int((sel & gmask).sum()), n_sel),
        "poorest_selected": int((sel & pmask).sum()),
        "poorest_share": safe_rate(int((sel & pmask).sum()), n_sel),
        "band_6_9": int((sel & (band == "6-9")).sum()),
        "band_10_12": int((sel & (band == "10-12")).sum()),
        "band_13_15": int((sel & (band == "13-15")).sum()),
        "band_16_17": int((sel & (band == "16-17")).sum()),
    }


def pivot(df, index, columns, values):
    t = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    return t.reindex(index=sorted(df[index].unique()), columns=sorted(df[columns].unique()))


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT.resolve()}")

    p("=" * 80)
    p("PHASE 3 FAIRNESS GRID (SciPy HiGHS, proven optimal, interrupt-safe)")
    p("=" * 80)
    p(f"INPUT: {INPUT.resolve()}")

    df = pd.read_csv(INPUT, low_memory=False)
    need = ["child_uid", "age", "_band", "female", "wealth_q",
            "Out_of_School", "child_weight_fsweight", "risk_score_oof", "PSU"]
    missing = [c for c in need if c not in df.columns]
    p(f"Input shape: {df.shape}; missing columns: {missing if missing else 'none'}")
    if missing:
        p("VERDICT: FAIL - missing columns.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        return

    for c in ["age", "female", "wealth_q", "Out_of_School",
              "child_weight_fsweight", "risk_score_oof"]:
        df[c] = num(df[c])
    df = df.loc[df["child_weight_fsweight"] > 0].reset_index(drop=True)

    n = len(df)
    K = int(math.ceil(TOP_FRACTION * n))
    uid = df["child_uid"].astype(str).to_numpy()
    risk = df["risk_score_oof"].to_numpy(float)
    w = df["child_weight_fsweight"].to_numpy(float)
    y = df["Out_of_School"].to_numpy(int)
    band = df["_band"].astype("string").to_numpy()
    wrisk = w * risk
    ymask = df["age"].to_numpy(float) <= YOUNG_AGE_MAX
    gmask = df["female"].to_numpy(float) == 1
    pmask = df["wealth_q"].to_numpy(float) == 1

    total_oos = int(y.sum())
    total_wmass = float((w * y).sum())
    prop_young = int(round(K * int(ymask.sum()) / n))
    prop_girls = int(round(K * int(gmask.sum()) / n))

    p("")
    p("SETUP")
    p("-" * 80)
    p(f"n={n}, K={K}, observed OOS={total_oos}, weighted OOS mass={total_wmass:.4f}")
    p(f"young floor grid={YOUNG_FLOORS}, proportional young={prop_young}")
    p(f"girls floor grid={GIRLS_FLOORS}, proportional girls={prop_girls}")
    p(f"solver: mip_rel_gap=0.0, mip_abs_gap=1e-12, time_limit={TIME_LIMIT}s")

    p("")
    p("SELF-CHECK (SciPy objective vs in-script greedy reference, girls=0)")
    p("-" * 80)
    p("Gate is on SUM OBJ (the objective). Capture is a tie diagnostic only.")
    p("Expected greedy objectives (must match earlier greedy report):")
    p("  unweighted y0/y73/y423 = 325.663745 / 318.084313 / 222.022851")
    p("  weighted   y0/y73/y423 = 835.475663 / 833.698418 / 656.653056")

    check_ok = True
    interrupted = False
    try:
        for obj_name, obj in [("unweighted", risk), ("weighted", wrisk)]:
            for yv in SELFCHECK_Y:
                gsel = greedy_young(obj, ymask, n, K, yv, uid)
                if gsel is None:
                    p(f"{obj_name} y_min={yv}: greedy infeasible (floor>young pop)")
                    check_ok = False
                    continue
                g_obj = float(obj[gsel].sum())
                g_cap = int(y[gsel].sum())
                ssel, st, proven = solve_cell(obj, ymask, gmask, n, K, yv, 0)
                if ssel is None:
                    p(f"{obj_name} y_min={yv}: SCIPY FAILED status={st}")
                    check_ok = False
                    continue
                s_obj = float(obj[ssel].sum())
                s_cap = int(y[ssel].sum())
                obj_match = abs(s_obj - g_obj) <= OBJ_TOL
                not_above = s_obj <= g_obj + OBJ_TOL
                feas = int(ssel.sum()) == K and int((ssel & ymask).sum()) >= yv
                ok = obj_match and not_above and feas and proven
                check_ok = check_ok and ok
                cap_note = "same" if s_cap == g_cap else f"DIFFERS by {s_cap - g_cap} (tie in obj)"
                p(f"{obj_name} y_min={yv}: greedy_obj={g_obj:.9f} cap={g_cap} | "
                  f"scipy_obj={s_obj:.9f} cap={s_cap} [{cap_note}] | "
                  f"proven={proven} -> {'OK' if ok else 'MISMATCH'}")
    except KeyboardInterrupt:
        interrupted = True
        p("INTERRUPTED during self-check.")
        check_ok = False

    p(f"SELF-CHECK: {'PASS' if check_ok else 'FAIL'}")
    if not check_ok:
        p("")
        if interrupted:
            p("VERDICT: INCOMPLETE - self-check interrupted; no grid produced.")
        else:
            p("VERDICT: FAIL - SciPy did not reproduce the proven greedy optimum at gap 0.")
            p("Stop here; do not use any grid. Send this report for investigation.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    rows = []
    grid_interrupted = False
    try:
        for obj_name, obj in [("unweighted", risk), ("weighted", wrisk)]:
            for yv in YOUNG_FLOORS:
                for gv in GIRLS_FLOORS:
                    try:
                        sel, st, proven = solve_cell(obj, ymask, gmask, n, K, yv, gv)
                    except Exception as e:
                        rows.append({"objective": obj_name, "young_floor": yv,
                                     "girls_floor": gv,
                                     "status": f"error:{type(e).__name__}",
                                     "proven_optimal": False})
                        continue
                    if sel is None:
                        rows.append({"objective": obj_name, "young_floor": yv,
                                     "girls_floor": gv,
                                     "status": f"infeasible({st})",
                                     "proven_optimal": False})
                        continue
                    m = metrics(sel, y, w, risk, wrisk, ymask, gmask, pmask, band,
                                total_oos, total_wmass)
                    m.update({"objective": obj_name, "young_floor": yv,
                              "girls_floor": gv, "status": "optimal",
                              "proven_optimal": bool(proven)})
                    rows.append(m)
    except KeyboardInterrupt:
        grid_interrupted = True
        p("INTERRUPTED during grid; writing partial results.")

    if not rows:
        p("")
        p("VERDICT: FAIL - no grid cells produced.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_CSV, index=False)
    p("")
    p(f"Saved grid (aggregate only): {OUT_CSV.resolve()}")

    n_proven = int(grid["proven_optimal"].astype(bool).sum())
    n_cells = len(grid)
    p(f"Cells present: {n_cells}/{len(YOUNG_FLOORS) * len(GIRLS_FLOORS) * 2}")
    p(f"Cells proven optimal: {n_proven}/{n_cells}")
    if n_proven < n_cells:
        p(f"WARNING: {n_cells - n_proven} cells not proven optimal (time limit or fallback); "
          f"treat those rows as upper-bound estimates, not final.")

    unw = grid[grid["objective"] == "unweighted"].copy()
    wgt = grid[grid["objective"] == "weighted"].copy()

    def show_pivot(dframe, values, title, round_dp=None):
        p("")
        p(title)
        p("-" * 80)
        if len(dframe) == 0:
            p("no rows")
            return
        t = pivot(dframe, "young_floor", "girls_floor", values)
        if round_dp is not None:
            t = t.round(round_dp)
        p(t.to_string())

    show_pivot(unw, "observed_oos_captured", "PIVOT 1 - unweighted: observed OOS captured")
    show_pivot(unw, "girls_share", "PIVOT 2 - unweighted: girls share of selected", 4)
    show_pivot(wgt, "observed_oos_captured", "PIVOT 3 - weighted (sensitivity): observed OOS captured")
    show_pivot(wgt, "girls_share", "PIVOT 4 - weighted (sensitivity): girls share of selected", 4)

    p("")
    p("PRICE OF THE GIRLS FLOOR (unweighted, young_floor = proportional)")
    p("-" * 80)
    base = unw[(unw["young_floor"] == prop_young) & (unw["girls_floor"] == 0)]
    if len(base) > 0:
        b_cap = int(base["observed_oos_captured"].iloc[0])
        sub = unw[unw["young_floor"] == prop_young].sort_values("girls_floor")
        out = pd.DataFrame({
            "girls_floor": sub["girls_floor"],
            "captured_oos": sub["observed_oos_captured"],
            "capture_drop_vs_g0": b_cap - sub["observed_oos_captured"],
            "girls_share": sub["girls_share"].round(4),
            "poorest_share": sub["poorest_share"].round(4),
        })
        p(out.to_string(index=False))
        p(f"(baseline at g=0: captured={b_cap})")
    else:
        p("No baseline row at young_floor = proportional (cell missing or interrupted).")

    p("")
    p("VERDICT")
    p("-" * 80)
    if grid_interrupted:
        p(f"FAIRNESS GRID STATUS: PARTIAL_INTERRUPTED ({n_cells} cells written before stop)")
    elif n_proven == n_cells:
        p("FAIRNESS GRID STATUS: PASS (all cells proven optimal, self-check passed)")
    else:
        p(f"FAIRNESS GRID STATUS: PARTIAL ({n_proven}/{n_cells} cells proven optimal)")

    p("")
    p("SEND BACK: reports/phase3_fairness_grid.txt and results/phase3_fairness_grid.csv.")
    p("Do not send data/processed/chattogram_rural_phase3_optimization_input.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()