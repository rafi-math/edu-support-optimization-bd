from pathlib import Path
import math

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:
    raise SystemExit(
        "PuLP is not installed. Run:\n"
        "python -m pip install --upgrade pulp"
    )


INPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
REPORT = Path("reports/phase3_milp_young_floor_sweep.txt")
OUT_CSV = Path("results/phase3_milp_young_floor_sweep.csv")

TOP_FRACTION = 0.15
YOUNG_AGE_MAX = 12  # young means age <= 12, i.e. age < 13

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def status_name(prob):
    st = prob.status

    if hasattr(pulp, "LpStatus"):
        try:
            return pulp.LpStatus[st]
        except Exception:
            pass

    mapping = {
        -1: "Not solved",
        0: "Optimal",
        1: "Infeasible",
        2: "Unbounded",
        3: "Undefined",
        4: "Stalled",
    }

    return mapping.get(st, str(st))


def choose_solver():
    attempts = []

    class_names = [
        "PULP_CBC_CMD",
        "CBC",
        "HiGHS",
        "HiGHS_CMD",
        "SCIP_PY",
        "SCIP",
        "COIN_CMD",
    ]

    kwarg_options = [
        {"msg": False},
        {"msg": 0},
        {"msg": False, "timeLimit": 120},
        {},
    ]

    for cls_name in class_names:
        cls = getattr(pulp, cls_name, None)

        if cls is None:
            attempts.append(f"{cls_name}: class not found")
            continue

        for kwargs in kwarg_options:
            try:
                solver = cls(**kwargs)

                if solver.available():
                    return cls_name, solver, attempts

                attempts.append(
                    f"{cls_name}({kwargs}): reported not available"
                )

            except Exception as exc:
                attempts.append(
                    f"{cls_name}({kwargs}): {type(exc).__name__}: {exc}"
                )

    try:
        available = pulp.listSolvers(onlyAvailable=True)

        for name in available:
            for kwargs in kwarg_options:
                try:
                    solver = pulp.getSolver(name, **kwargs)

                    if solver.available():
                        return name, solver, attempts

                    attempts.append(
                        f"getSolver({name}, {kwargs}): reported not available"
                    )

                except Exception as exc:
                    attempts.append(
                        f"getSolver({name}, {kwargs}): "
                        f"{type(exc).__name__}: {exc}"
                    )

    except Exception as exc:
        attempts.append(
            f"listSolvers failed: {type(exc).__name__}: {exc}"
        )

    return None, None, attempts


def safe_rate(numer, denom):
    if denom is None or denom == 0 or not np.isfinite(denom):
        return np.nan

    return float(numer / denom)


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise SystemExit(f"Optimization input not found: {INPUT.resolve()}")

    p("=" * 80)
    p("PHASE 3 MILP YOUNG-FLOOR SWEEP")
    p("=" * 80)
    p(f"INPUT: {INPUT.resolve()}")
    p(f"REPORT: {REPORT.resolve()}")
    p(f"AGGREGATE CSV: {OUT_CSV.resolve()}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = [
        "child_uid",
        "age",
        "_band",
        "female",
        "wealth_q",
        "Out_of_School",
        "child_weight_fsweight",
        "risk_score_oof",
        "PSU",
    ]

    missing = [c for c in required if c not in df.columns]
    p("")
    p(f"Input shape: {df.shape}")
    p(f"Missing required columns: {missing if missing else 'none'}")

    if missing:
        p("")
        p("VERDICT: FAIL - required columns missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    for col in [
        "age",
        "female",
        "wealth_q",
        "Out_of_School",
        "child_weight_fsweight",
        "risk_score_oof",
        "PSU",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    critical_missing = df[required].isna().sum()
    p("")
    p("Missing values in required columns:")
    p(critical_missing.to_string())

    if int(critical_missing.sum()) > 0:
        p("")
        p("VERDICT: FAIL - required values missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    df = df.loc[df["child_weight_fsweight"] > 0].copy().reset_index(drop=True)

    n = len(df)
    K = int(math.ceil(TOP_FRACTION * n))

    risk = df["risk_score_oof"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    female = df["female"].to_numpy(dtype=float)
    wealth = df["wealth_q"].to_numpy(dtype=float)
    y = df["Out_of_School"].to_numpy(dtype=int)
    w = df["child_weight_fsweight"].to_numpy(dtype=float)
    band = df["_band"].astype("string").to_numpy()

    young = age <= YOUNG_AGE_MAX
    girl = female == 1
    poorest = wealth == 1

    young_idx = np.flatnonzero(young)

    total_oos = int(y.sum())
    total_weight = float(w.sum())
    total_weighted_oos = float((w * y).sum())

    young_pop = int(young.sum())
    girl_pop = int(girl.sum())
    poorest_pop = int(poorest.sum())

    proportional_young_floor = int(round(K * young_pop / n))
    max_possible_young_floor = min(K, young_pop)

    p("")
    p("PROBLEM SETUP")
    p("-" * 80)
    p(f"Modeling-ready rows: {n}")
    p(f"Selection fraction: {TOP_FRACTION:.2%}")
    p(f"Number to select, K: {K}")
    p(f"Observed out-of-school children: {total_oos}")
    p(f"Unweighted out-of-school rate: {safe_rate(total_oos, n):.6f}")
    p(f"Total child weight: {total_weight:.6f}")
    p(f"Total weighted out-of-school mass: {total_weighted_oos:.6f}")
    p(f"Young population, age <= {YOUNG_AGE_MAX}: {young_pop} "
      f"({safe_rate(young_pop, n):.4%})")
    p(f"Girls population: {girl_pop} ({safe_rate(girl_pop, n):.4%})")
    p(f"Poorest quintile population: {poorest_pop} "
      f"({safe_rate(poorest_pop, n):.4%})")
    p(f"Proportional young floor: {proportional_young_floor}")
    p(f"Maximum possible young floor: {max_possible_young_floor}")

    candidate_fracs = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        1 / 3,
        0.50,
        0.60,
        0.70,
    ]

    y_mins = [int(round(f * K)) for f in candidate_fracs]
    y_mins.append(proportional_young_floor)

    y_mins = sorted(
        {
            max(0, min(int(v), max_possible_young_floor))
            for v in y_mins
        }
    )

    p("")
    p("Young-floor scenarios to run:")
    p(str(y_mins))

    solver_name, solver, attempts = choose_solver()

    p("")
    p("SOLVER")
    p("-" * 80)

    if solver is None:
        p("VERDICT: FAIL - no available MILP solver found.")
        p("")
        p("Solver attempts:")
        for item in attempts:
            p(f"- {item}")
        p("")
        p("Try:")
        p("python -m pip install --upgrade pulp")
        p("")
        p("If PuLP is already installed and no solver is available, the next")
        p("task will be to install a solver backend or switch to OR-Tools.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p(f"Selected solver: {solver_name}")
    p(f"Solver attempts log entries: {len(attempts)}")

    rows = []

    for y_min in y_mins:
        p("")
        p(f"Solving scenario y_min={y_min} ...")

        prob = pulp.LpProblem(
            f"phase3_young_floor_{y_min}",
            pulp.LpMaximize,
        )

        x = [
            pulp.LpVariable(
                f"x_{i}",
                lowBound=0,
                upBound=1,
                cat="Binary",
            )
            for i in range(n)
        ]

        prob += (
            pulp.lpSum(risk[i] * x[i] for i in range(n)),
            "maximize_predicted_out_of_school_need",
        )

        prob += (
            pulp.lpSum(x) == K,
            "select_exactly_K_children",
        )

        if y_min > 0:
            prob += (
                pulp.lpSum(x[i] for i in young_idx) >= y_min,
                "minimum_young_coverage_floor",
            )

        try:
            prob.solve(solver)
        except Exception as exc:
            p(f"  solve error: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "y_min": y_min,
                    "status": f"solve_error: {type(exc).__name__}",
                }
            )
            continue

        st = status_name(prob)
        p(f"  status: {st}")

        row = {
            "y_min": y_min,
            "y_min_share_of_K": safe_rate(y_min, K),
            "status": st,
        }

        if st != "Optimal":
            rows.append(row)
            continue

        selected_bool = np.array(
            [
                x[i].varValue is not None and x[i].varValue > 0.5
                for i in range(n)
            ],
            dtype=bool,
        )

        selected_n = int(selected_bool.sum())
        young_selected = int((selected_bool & young).sum())
        girl_selected = int((selected_bool & girl).sum())
        poorest_selected = int((selected_bool & poorest).sum())

        observed_oos_captured = int(y[selected_bool].sum())
        weighted_oos_captured = float((w[selected_bool] * y[selected_bool]).sum())

        objective_raw_risk = float(risk[selected_bool].sum())
        weighted_objective_raw_risk = float(
            (w[selected_bool] * risk[selected_bool]).sum()
        )

        total_weight_selected = float(w[selected_bool].sum())

        mean_risk_selected = (
            float(risk[selected_bool].mean())
            if selected_n > 0
            else np.nan
        )

        mean_risk_not_selected = (
            float(risk[~selected_bool].mean())
            if (n - selected_n) > 0
            else np.nan
        )

        row.update(
            {
                "selected_n": selected_n,
                "young_selected": young_selected,
                "young_share_selected": safe_rate(young_selected, selected_n),
                "young_share_population": safe_rate(young_pop, n),
                "girls_selected": girl_selected,
                "girls_share_selected": safe_rate(girl_selected, selected_n),
                "girls_share_population": safe_rate(girl_pop, n),
                "poorest_selected": poorest_selected,
                "poorest_share_selected": safe_rate(poorest_selected, selected_n),
                "poorest_share_population": safe_rate(poorest_pop, n),
                "objective_raw_risk": objective_raw_risk,
                "weighted_objective_raw_risk": weighted_objective_raw_risk,
                "total_weight_selected": total_weight_selected,
                "weighted_population_share_selected": safe_rate(
                    total_weight_selected,
                    total_weight,
                ),
                "observed_oos_captured": observed_oos_captured,
                "unweighted_capture_rate": safe_rate(
                    observed_oos_captured,
                    total_oos,
                ),
                "weighted_capture_rate": safe_rate(
                    weighted_oos_captured,
                    total_weighted_oos,
                ),
                "mean_risk_selected": mean_risk_selected,
                "mean_risk_not_selected": mean_risk_not_selected,
            }
        )

        for b in ["6-9", "10-12", "13-15", "16-17"]:
            mask = band == b
            cnt = int((selected_bool & mask).sum())
            row[f"band_{b}_selected"] = cnt
            row[f"band_{b}_share_selected"] = safe_rate(cnt, selected_n)
            row[f"band_{b}_share_population"] = safe_rate(int(mask.sum()), n)

        # Sanity checks.
        checks_ok = True

        if selected_n != K:
            row["check_selected_n_equals_K"] = False
            checks_ok = False
        else:
            row["check_selected_n_equals_K"] = True

        if young_selected < y_min:
            row["check_young_floor_met"] = False
            checks_ok = False
        else:
            row["check_young_floor_met"] = True

        if not checks_ok:
            row["status"] = f"{st} but solution check failed"

        rows.append(row)

    if not rows:
        p("")
        p("VERDICT: FAIL - no scenarios were solved.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    res = pd.DataFrame(rows)

    p("")
    p("AGGREGATE RESULTS")
    p("-" * 80)

    key_cols = [
        "y_min",
        "status",
        "young_selected",
        "young_share_selected",
        "girls_share_selected",
        "poorest_share_selected",
        "observed_oos_captured",
        "unweighted_capture_rate",
        "weighted_capture_rate",
        "objective_raw_risk",
        "weighted_population_share_selected",
    ]

    key_cols = [c for c in key_cols if c in res.columns]

    p(res[key_cols].to_string(index=False))

    p("")
    p("BAND COMPOSITION")
    p("-" * 80)

    band_cols = [
        "y_min",
        "band_6-9_share_selected",
        "band_10-12_share_selected",
        "band_13-15_share_selected",
        "band_16-17_share_selected",
    ]

    band_cols = [c for c in band_cols if c in res.columns]

    p(res[band_cols].to_string(index=False))

    p("")
    p("PRICE OF YOUNG FLOOR")
    p("-" * 80)

    if "objective_raw_risk" in res.columns and len(res) > 0:
        base = res.loc[res["y_min"] == 0]

        if len(base) > 0 and "objective_raw_risk" in base.columns:
            base_obj = float(base["objective_raw_risk"].iloc[0])
            base_capture = (
                float(base["observed_oos_captured"].iloc[0])
                if "observed_oos_captured" in base.columns
                else np.nan
            )

            res["objective_drop_vs_y0"] = base_obj - res["objective_raw_risk"]
            res["capture_drop_vs_y0"] = base_capture - res["observed_oos_captured"]

            res["drop_per_additional_young"] = np.where(
                res["young_selected"] > 0,
                res["capture_drop_vs_y0"] / res["young_selected"],
                np.nan,
            )

            show_price = [
                "y_min",
                "young_selected",
                "objective_drop_vs_y0",
                "capture_drop_vs_y0",
                "drop_per_additional_young",
            ]

            show_price = [c for c in show_price if c in res.columns]

            p(res[show_price].to_string(index=False))
        else:
            p("No y_min=0 baseline row found, so price columns were not computed.")
    else:
        p("Objective column missing, so price columns were not computed.")

    res.to_csv(OUT_CSV, index=False)

    p("")
    p(f"Saved aggregate CSV: {OUT_CSV.resolve()}")
    p("This CSV contains one row per scenario and no child-level identifiers.")

    p("")
    p("VERDICT")
    p("-" * 80)

    optimal_rows = res.loc[res["status"].astype(str).str.startswith("Optimal")]

    if len(optimal_rows) == 0:
        p("MILP STATUS: FAIL")
        p("No scenario reached Optimal status.")
    else:
        p("MILP STATUS: PASS for solved scenarios")
        p(
            "The model used risk_score_oof as the objective and enforced "
            "minimum young coverage floors. Observed Out_of_School was used "
            "only for evaluation, not as a decision input."
        )

    p("")
    p("INTERPRETATION")
    p("-" * 80)
    p("- y_min=0 is the raw-risk optimum and should capture the most predicted need.")
    p("- As y_min rises, young coverage improves but captured out-of-school children may fall.")
    p("- The proportional young floor is a useful equity benchmark, not automatically the final choice.")
    p("- Do not use age-standardized percentile as the objective; it balanced age but lost too much capture and worsened girls' share.")
    p("- This sweep is sample-mode: it selects 730 sampled children. Population-weighted budgeting is a separate next check.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/phase3_milp_young_floor_sweep.txt.")
    p("You may also send results/phase3_milp_young_floor_sweep.csv because it is aggregate only.")
    p("Do not send data/processed/chattogram_rural_phase3_optimization_input.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()