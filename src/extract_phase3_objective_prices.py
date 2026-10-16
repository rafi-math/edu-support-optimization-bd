from pathlib import Path
import numpy as np
import pandas as pd

CSV = Path("results/phase3_fairness_grid.csv")
REPORT = Path("reports/phase3_objective_prices.txt")
lines = []

def p(m=""):
    t = str(m); print(t); lines.append(t)

EXP_UNW_Y0 = {0: 325.663745, 73: 318.084313, 146: 304.270078,
              243: 280.045189, 423: 222.022851}
EXP_W_Y0 = {0: 835.475663, 73: 833.698418, 146: 817.442884,
            243: 777.302153, 423: 656.653056}
TOL = 1e-4

def curve(df, obj, fixed_col, fixed_val, vary_col):
    d = df[df[fixed_col] == fixed_val].sort_values(vary_col)
    base = d[d[vary_col] == d[vary_col].min()][obj].iloc[0]
    out = pd.DataFrame({vary_col: d[vary_col], obj: d[obj].round(6)})
    out["drop_vs_first"] = (base - d[obj]).round(6)
    step = out[vary_col].diff()
    out["drop_per_unit"] = (out["drop_vs_first"].diff() / step).round(6)
    return out

def mono_check(df, obj):
    bad = []
    for yv, g in df.groupby("young_floor"):
        g = g.sort_values("girls_floor")
        if (np.diff(g[obj].to_numpy()) > 1e-9).any():
            bad.append(f"obj increases with girls_floor at young={yv}")
    for gv, g in df.groupby("girls_floor"):
        g = g.sort_values("young_floor")
        if (np.diff(g[obj].to_numpy()) > 1e-9).any():
            bad.append(f"obj increases with young_floor at girls={gv}")
    return bad

def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        raise SystemExit(f"CSV not found: {CSV.resolve()}")
    df = pd.read_csv(CSV)
    p("=" * 70); p("PHASE 3 OBJECTIVE PRICES AND INTERACTION"); p("=" * 70)
    p(f"rows={len(df)}; proven_optimal True={int(df['proven_optimal'].astype(bool).sum())}")

    p(""); p("CROSS-CHECK (y, g=0) vs independent greedy reference"); p("-" * 70)
    for tag, col, exp in [("unweighted", "obj_unweighted", EXP_UNW_Y0),
                          ("weighted", "obj_weighted", EXP_W_Y0)]:
        d = df[(df["objective"] == tag) & (df["girls_floor"] == 0)].set_index("young_floor")
        for yv, e in exp.items():
            got = float(d.loc[yv, col])
            ok = abs(got - e) <= TOL
            p(f"{tag} y={yv}: grid={got:.6f} greedy={e:.6f} -> {'OK' if ok else 'MISMATCH'}")

    p(""); p("MONOTONICITY INTEGRITY CHECK"); p("-" * 70)
    for tag, col in [("unweighted", "obj_unweighted"), ("weighted", "obj_weighted")]:
        d = df[df["objective"] == tag]
        bad = mono_check(d, col)
        p(f"{tag}: {'PASS (non-increasing in both floors)' if not bad else 'FAIL'}")
        for b in bad:
            p(f"  - {b}")

    for tag, col in [("unweighted", "obj_unweighted"), ("weighted", "obj_weighted")]:
        d = df[df["objective"] == tag]
        p(""); p(f"YOUNG PRICE CURVE, {tag} (girls_floor=0)"); p("-" * 70)
        p(curve(d, col, "girls_floor", 0, "young_floor").to_string(index=False))
        for yv in (0, 423):
            p(""); p(f"GIRLS PRICE CURVE, {tag} (young_floor={yv})"); p("-" * 70)
            p(curve(d, col, "young_floor", yv, "girls_floor").to_string(index=False))

    p(""); p("2x2 INTERACTION IN OBJECTIVE UNITS"); p("-" * 70)
    for tag, col in [("unweighted", "obj_unweighted"), ("weighted", "obj_weighted")]:
        d = df[df["objective"] == tag].set_index(["young_floor", "girls_floor"])[col]
        o00, o4230 = d[(0, 0)], d[(423, 0)]
        o0367, o423367 = d[(0, 367)], d[(423, 367)]
        dy = o00 - o4230          # cost of young floor alone
        dg = o00 - o0367          # cost of girls floor alone
        db = o00 - o423367        # cost of both
        inter = db - (dy + dg)    # >0 superadditive, <0 subadditive
        p(f"{tag}: young_cost={dy:.6f} girls_cost={dg:.6f} "
          f"both_cost={db:.6f} interaction={inter:.6f} "
          f"({'superadditive' if inter > 1e-9 else 'subadditive' if inter < -1e-9 else 'additive'})")

    p(""); p("SEND BACK: reports/phase3_objective_prices.txt")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    p(f"REPORT SAVED: {REPORT.resolve()}")

if __name__ == "__main__":
    main()