from pathlib import Path

import numpy as np
import pandas as pd


OPT_INPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
GRID_CSV = Path("results/phase3_fairness_grid.csv")
FINE_CSV = Path("results/phase3_girls_curve_fine.csv")

REPORT = Path("reports/phase3_fairness_summary.txt")
SUMMARY_CSV = Path("results/phase3_fairness_summary.csv")
CURVES_CSV = Path("results/phase3_fairness_curves.csv")

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def require_file(path, label):
    if not path.exists():
        raise SystemExit(f"{label} not found: {path.resolve()}")


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{label} missing required columns: {missing}")


def weighted_share(mask, weights):
    m = np.asarray(mask, dtype=float)
    w = num(pd.Series(weights)).to_numpy(dtype=float)

    good = np.isfinite(m) & np.isfinite(w) & (w > 0)

    if int(good.sum()) == 0:
        return np.nan

    return float(np.average(m[good], weights=w[good]))


def get_grid_row(grid, y, g):
    row = grid.loc[
        (grid["objective"] == "unweighted")
        & (num(grid["young_floor"]) == y)
        & (num(grid["girls_floor"]) == g)
    ]

    if len(row) == 0:
        return None

    return row.iloc[0]


def fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{100.0 * float(x):.2f}%"


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    CURVES_CSV.parent.mkdir(parents=True, exist_ok=True)

    require_file(OPT_INPUT, "Optimization input")
    require_file(GRID_CSV, "Fairness grid CSV")
    require_file(FINE_CSV, "Fine girls curve CSV")

    p("=" * 80)
    p("PHASE 3 FAIRNESS SUMMARY AND PLOT DATA")
    p("=" * 80)
    p(f"OPTIMIZATION INPUT: {OPT_INPUT.resolve()}")
    p(f"GRID CSV: {GRID_CSV.resolve()}")
    p(f"FINE GIRLS CSV: {FINE_CSV.resolve()}")
    p(f"REPORT: {REPORT.resolve()}")
    p(f"SUMMARY CSV: {SUMMARY_CSV.resolve()}")
    p(f"CURVES CSV: {CURVES_CSV.resolve()}")

    opt = pd.read_csv(OPT_INPUT, low_memory=False)

    require_columns(
        opt,
        [
            "age",
            "female",
            "wealth_q",
            "Out_of_School",
            "child_weight_fsweight",
            "risk_score_oof",
            "PSU",
        ],
        "Optimization input",
    )

    for col in [
        "age",
        "female",
        "wealth_q",
        "Out_of_School",
        "child_weight_fsweight",
        "risk_score_oof",
    ]:
        opt[col] = num(opt[col])

    opt = opt.loc[opt["child_weight_fsweight"] > 0].copy().reset_index(drop=True)

    n = len(opt)
    K = int(np.ceil(0.15 * n))

    young_mask = opt["age"] <= 12
    girl_mask = opt["female"] == 1
    poorest_mask = opt["wealth_q"] == 1
    oos_mask = opt["Out_of_School"] == 1

    p("")
    p("SAMPLE COMPOSITION, AGGREGATE ONLY")
    p("-" * 80)
    p(f"Modeling-ready rows: {n}")
    p(f"Selection count K: {K}")
    p(f"Observed out-of-school children: {int(oos_mask.sum())}")

    comp_rows = []

    for name, mask in [
        ("young_age_le_12", young_mask),
        ("girls", girl_mask),
        ("poorest_quintile", poorest_mask),
    ]:
        pop_share = float(mask.mean())
        oos_share = float(mask[oos_mask].mean()) if int(oos_mask.sum()) > 0 else np.nan
        w_pop_share = weighted_share(mask, opt["child_weight_fsweight"])
        w_oos_share = weighted_share(mask & oos_mask, opt["child_weight_fsweight"])

        comp_rows.append(
            {
                "group": name,
                "population_share": pop_share,
                "observed_oos_pool_share": oos_share,
                "weighted_population_share_provisional": w_pop_share,
                "weighted_oos_pool_share_provisional": w_oos_share,
            }
        )

    comp = pd.DataFrame(comp_rows)

    p("")
    p("Unweighted shares are sample shares. Weighted shares use child_weight_fsweight and remain provisional.")
    p(comp.to_string(index=False))

    grid = pd.read_csv(GRID_CSV, low_memory=False)
    fine = pd.read_csv(FINE_CSV, low_memory=False)

    require_columns(
        grid,
        [
            "objective",
            "young_floor",
            "girls_floor",
            "status",
            "proven_optimal",
            "selected_n",
            "obj_unweighted",
            "observed_oos_captured",
            "young_selected",
            "girls_selected",
            "girls_share",
            "poorest_selected",
            "poorest_share",
        ],
        "Grid CSV",
    )

    require_columns(
        fine,
        [
            "young_floor",
            "girls_floor",
            "status",
            "proven_optimal",
            "obj_unweighted",
            "drop_vs_g0",
            "girls_share",
            "observed_oos_captured",
            "unweighted_capture_rate",
        ],
        "Fine girls CSV",
    )

    grid = grid.loc[grid["objective"] == "unweighted"].copy()

    for df in [grid, fine]:
        for col in [
            "young_floor",
            "girls_floor",
            "selected_n",
            "obj_unweighted",
            "observed_oos_captured",
            "young_selected",
            "girls_selected",
            "girls_share",
            "poorest_selected",
            "poorest_share",
            "drop_vs_g0",
            "unweighted_capture_rate",
        ]:
            if col in df.columns:
                df[col] = num(df[col])

    raw = get_grid_row(grid, 0, 0)

    if raw is None:
        raise SystemExit("Grid CSV missing unweighted baseline row (young_floor=0, girls_floor=0).")

    raw_obj = float(raw["obj_unweighted"])
    raw_cap = int(raw["observed_oos_captured"])
    raw_K = int(raw["selected_n"])

    if raw_K != K:
        p("")
        p(f"WARNING: grid selected_n={raw_K}, computed K={K}. Using grid selected_n for shares.")
        K = raw_K

    p("")
    p("BASELINE")
    p("-" * 80)
    p(f"Raw baseline objective: {raw_obj:.6f}")
    p(f"Raw baseline observed OOS captured: {raw_cap}")
    p(f"Raw baseline selected count: {K}")

    scenario_defs = [
        ("raw_risk_no_floors", 0, 0),
        ("girls_40pct_only", 0, 292),
        ("girls_parity_only", 0, 367),
        ("young_10pct_only", 73, 0),
        ("young_20pct_only", 146, 0),
        ("young_33pct_only", 243, 0),
        ("young_proportional_only", 423, 0),
        ("young_33pct_girls_40pct", 243, 292),
        ("young_33pct_girls_parity", 243, 367),
        ("young_proportional_girls_40pct", 423, 292),
        ("young_proportional_girls_parity", 423, 367),
    ]

    summary_rows = []

    for label, y, g in scenario_defs:
        row = get_grid_row(grid, y, g)

        if row is None:
            summary_rows.append(
                {
                    "scenario": label,
                    "young_floor": y,
                    "girls_floor": g,
                    "status": "MISSING_FROM_GRID",
                }
            )
            continue

        y_only = get_grid_row(grid, y, 0)
        g_only = get_grid_row(grid, 0, g)

        obj = float(row["obj_unweighted"])
        cap = int(row["observed_oos_captured"])

        obj_drop = raw_obj - obj
        cap_drop = raw_cap - cap

        young_obj_drop = np.nan
        girls_obj_drop = np.nan
        additive_obj_drop = np.nan
        interaction_obj_drop = np.nan

        young_cap_drop = np.nan
        girls_cap_drop = np.nan
        additive_cap_drop = np.nan
        interaction_cap_drop = np.nan

        if y_only is not None:
            young_obj_drop = raw_obj - float(y_only["obj_unweighted"])
            young_cap_drop = raw_cap - int(y_only["observed_oos_captured"])

        if g_only is not None:
            girls_obj_drop = raw_obj - float(g_only["obj_unweighted"])
            girls_cap_drop = raw_cap - int(g_only["observed_oos_captured"])

        if np.isfinite(young_obj_drop) and np.isfinite(girls_obj_drop):
            additive_obj_drop = young_obj_drop + girls_obj_drop
            interaction_obj_drop = obj_drop - additive_obj_drop

        if np.isfinite(young_cap_drop) and np.isfinite(girls_cap_drop):
            additive_cap_drop = young_cap_drop + girls_cap_drop
            interaction_cap_drop = cap_drop - additive_cap_drop

        summary_rows.append(
            {
                "scenario": label,
                "young_floor": y,
                "girls_floor": g,
                "young_share": float(row["young_selected"]) / K,
                "girls_share": float(row["girls_share"]),
                "poorest_share": float(row["poorest_share"]),
                "obj_unweighted": obj,
                "obj_drop_vs_raw": obj_drop,
                "obj_drop_pct_vs_raw": obj_drop / raw_obj if raw_obj > 0 else np.nan,
                "observed_oos_captured": cap,
                "capture_drop_vs_raw": cap_drop,
                "capture_drop_pct_vs_raw": cap_drop / raw_cap if raw_cap > 0 else np.nan,
                "young_only_obj_drop": young_obj_drop,
                "girls_only_obj_drop": girls_obj_drop,
                "additive_obj_drop": additive_obj_drop,
                "interaction_obj_drop": interaction_obj_drop,
                "capture_interaction": interaction_cap_drop,
                "status": str(row["status"]),
                "proven_optimal": bool(row["proven_optimal"]),
            }
        )

    summary = pd.DataFrame(summary_rows)

    p("")
    p("POLICY SCENARIO SUMMARY")
    p("-" * 80)
    p("Objective is unweighted sum of out-of-fold risk scores. Observed OOS is evaluation only.")

    show_cols = [
        "scenario",
        "young_floor",
        "girls_floor",
        "young_share",
        "girls_share",
        "poorest_share",
        "obj_unweighted",
        "obj_drop_vs_raw",
        "obj_drop_pct_vs_raw",
        "observed_oos_captured",
        "capture_drop_vs_raw",
        "interaction_obj_drop",
        "status",
        "proven_optimal",
    ]

    show_cols = [c for c in show_cols if c in summary.columns]

    disp = summary[show_cols].copy()

    for col in [
        "young_share",
        "girls_share",
        "poorest_share",
        "obj_unweighted",
        "obj_drop_vs_raw",
        "interaction_obj_drop",
    ]:
        if col in disp.columns:
            disp[col] = disp[col].astype(float).round(6)

    for col in ["obj_drop_pct_vs_raw", "capture_drop_pct_vs_raw"]:
        if col in disp.columns:
            disp[col] = (disp[col].astype(float) * 100).round(2)

    p(disp.to_string(index=False))

    summary.to_csv(SUMMARY_CSV, index=False)

    p("")
    p(f"Saved summary CSV: {SUMMARY_CSV.resolve()}")

    curve_rows = []

    # Young curve at girls_floor=0 from the grid.
    for y in [0, 73, 146, 243, 423]:
        row = get_grid_row(grid, y, 0)

        if row is None:
            continue

        obj = float(row["obj_unweighted"])
        cap = int(row["observed_oos_captured"])

        curve_rows.append(
            {
                "curve": "young_floor_at_girls_0",
                "x_floor": y,
                "x_share": y / K,
                "young_floor": y,
                "girls_floor": 0,
                "obj_unweighted": obj,
                "obj_drop_vs_raw": raw_obj - obj,
                "observed_oos_captured": cap,
                "capture_drop_vs_raw": raw_cap - cap,
                "girls_share": float(row["girls_share"]),
                "poorest_share": float(row["poorest_share"]),
                "source": "phase3_fairness_grid.csv",
            }
        )

    # Fine girls curves from the fine CSV.
    for y in [0, 423]:
        sub = fine.loc[num(fine["young_floor"]) == y].sort_values("girls_floor")

        for _, row in sub.iterrows():
            g = int(row["girls_floor"])
            obj = float(row["obj_unweighted"])
            cap = int(row["observed_oos_captured"])

            curve_rows.append(
                {
                    "curve": f"girls_floor_at_young_{y}",
                    "x_floor": g,
                    "x_share": g / K,
                    "young_floor": y,
                    "girls_floor": g,
                    "obj_unweighted": obj,
                    "obj_drop_vs_raw": raw_obj - obj,
                    "observed_oos_captured": cap,
                    "capture_drop_vs_raw": raw_cap - cap,
                    "girls_share": float(row["girls_share"]),
                    "poorest_share": np.nan,
                    "source": "phase3_girls_curve_fine.csv",
                }
            )

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(CURVES_CSV, index=False)

    p("")
    p(f"Saved curves CSV: {CURVES_CSV.resolve()}")
    p("Use this file for figures. It contains aggregate curve points only.")

    p("")
    p("INTERACTION READ")
    p("-" * 80)

    inter = summary.loc[
        summary["scenario"].isin(
            [
                "young_33pct_girls_40pct",
                "young_33pct_girls_parity",
                "young_proportional_girls_40pct",
                "young_proportional_girls_parity",
            ]
        )
    ]

    if len(inter) == 0:
        p("No interaction rows available.")
    else:
        show_inter = [
            "scenario",
            "young_floor",
            "girls_floor",
            "obj_drop_vs_raw",
            "young_only_obj_drop",
            "girls_only_obj_drop",
            "additive_obj_drop",
            "interaction_obj_drop",
            "capture_interaction",
        ]

        show_inter = [c for c in show_inter if c in inter.columns]

        disp_inter = inter[show_inter].copy()

        for col in [
            "obj_drop_vs_raw",
            "young_only_obj_drop",
            "girls_only_obj_drop",
            "additive_obj_drop",
            "interaction_obj_drop",
        ]:
            if col in disp_inter.columns:
                disp_inter[col] = disp_inter[col].astype(float).round(6)

        p(disp_inter.to_string(index=False))
        p("")
        p("Negative interaction_obj_drop means subadditive: combined cost is less than the sum of separate costs.")
        p("Positive interaction_obj_drop means superadditive at that corner.")
        p("capture_interaction is diagnostic only because observed OOS is not the optimized objective.")

    p("")
    p("CAVEATS")
    p("-" * 80)
    p("- This summary uses the unweighted objective only.")
    p("- Weighted results remain provisional until the MICS codebook check on fsweight is completed.")
    p("- girls_floor=146 is not a meaningful fairness floor here because it is below the natural girl count.")
    p("- Observed OOS capture can wiggle non-monotonically; the objective monotonicity is the integrity check.")
    p("- The old Phase 2 superadditivity claim was based on the unreproducible Colab risk score and should not be used as current evidence.")

    p("")
    p("VERDICT")
    p("-" * 80)

    missing = summary.loc[summary["status"].astype(str) == "MISSING_FROM_GRID"]

    if len(missing) > 0:
        p("SUMMARY STATUS: PARTIAL")
        p("Some scenario rows were missing from the grid CSV.")
        p(missing[["scenario", "young_floor", "girls_floor"]].to_string(index=False))
    elif bool(summary["proven_optimal"].astype(bool).all()):
        p("SUMMARY STATUS: PASS")
        p("All displayed policy scenarios are present and proven optimal in the grid.")
    else:
        p("SUMMARY STATUS: PASS WITH WARNING")
        p("All displayed policy scenarios are present, but not all are marked proven_optimal.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/phase3_fairness_summary.txt.")
    p("You may also send results/phase3_fairness_summary.csv and results/phase3_fairness_curves.csv.")
    p("Do not send data/processed/chattogram_rural_phase3_optimization_input.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()