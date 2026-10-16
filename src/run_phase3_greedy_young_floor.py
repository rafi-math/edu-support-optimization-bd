from pathlib import Path
import math

import numpy as np
import pandas as pd


INPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
REPORT = Path("reports/phase3_greedy_young_floor_sweep.txt")
OUT_CSV = Path("results/phase3_greedy_young_floor_sweep.csv")

TOP_FRACTION = 0.15
YOUNG_AGE_MAX = 12

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def safe_rate(numer, denom):
    if denom is None or denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(numer / denom)


def order_by_objective(indices, obj, uid):
    """
    Return indices sorted by objective descending, then child_uid ascending.
    Tie-breaking is deterministic.
    """
    indices = np.asarray(indices, dtype=int)

    if len(indices) == 0:
        return indices

    # np.lexsort uses the last key as primary.
    # -obj ascending means obj descending.
    # uid ascending breaks ties deterministically.
    order = np.lexsort((uid[indices], -obj[indices]))
    return indices[order]


def select_with_young_floor(obj, y_min, n, K, young_mask, uid):
    """
    Exact greedy solution for:
        maximize sum obj_i x_i
        subject to:
            sum x_i = K
            sum_{young} x_i >= y_min
            x_i binary

    This is exact because every item has equal size in the cardinality
    constraint and the only group constraint is a lower bound.
    """
    young_pop = int(young_mask.sum())
    max_possible_young = min(K, young_pop)

    if y_min < 0 or y_min > max_possible_young:
        return None

    selected = np.zeros(n, dtype=bool)

    # Force the best y_min young children according to the objective.
    if y_min > 0:
        young_idx = np.flatnonzero(young_mask)
        young_order = order_by_objective(young_idx, obj, uid)
        forced = young_order[:y_min]
        selected[forced] = True

    # Fill remaining slots with the best remaining children overall.
    remaining_slots = K - int(selected.sum())

    if remaining_slots > 0:
        remaining_idx = np.flatnonzero(~selected)
        remaining_order = order_by_objective(remaining_idx, obj, uid)
        fill = remaining_order[:remaining_slots]
        selected[fill] = True

    return selected


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise SystemExit(f"Optimization input not found: {INPUT.resolve()}")

    p("=" * 80)
    p("PHASE 3 GREEDY YOUNG-FLOOR SWEEP")
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

    optional_diag = [
        "risk_pct_within_age",
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
    ] + optional_diag:
        if col in df.columns:
            df[col] = num(df[col])

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

    uid = df["child_uid"].astype(str).to_numpy()
    risk = df["risk_score_oof"].to_numpy(dtype=float)
    weight = df["child_weight_fsweight"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    female = df["female"].to_numpy(dtype=float)
    wealth = df["wealth_q"].to_numpy(dtype=float)
    y = df["Out_of_School"].astype(int).to_numpy()
    band = df["_band"].astype("string").to_numpy()

    if "risk_pct_within_age" in df.columns:
        risk_pct_age = df["risk_pct_within_age"].to_numpy(dtype=float)
    else:
        risk_pct_age = None

    young_mask = age <= YOUNG_AGE_MAX
    girl_mask = female == 1
    poorest_mask = wealth == 1

    weighted_risk = weight * risk

    total_oos = int(y.sum())
    total_weight = float(weight.sum())
    total_weighted_oos_mass = float((weight * y).sum())

    young_pop = int(young_mask.sum())
    girl_pop = int(girl_mask.sum())
    poorest_pop = int(poorest_mask.sum())

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
    p(f"Total weighted out-of-school mass: {total_weighted_oos_mass:.6f}")
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

    p("")
    p("METHOD")
    p("-" * 80)
    p("This is not a PuLP MILP run.")
    p("It is an exact greedy solution for the unit-count young-floor problem.")
    p("It avoids the PuLP 4.0.1 API incompatibility.")
    p("It will be replaced by a true MILP only when constraints become more complex,")
    p("for example real costs, PSU caps, population-weighted budgets, or multiple floors.")

    rules = [
        ("max_unweighted_risk", risk),
        ("max_weighted_risk", weighted_risk),
    ]

    rows = []

    def make_row(rule_name, y_min, selected_bool, obj_name):
        selected_n = int(selected_bool.sum())
        young_selected = int((selected_bool & young_mask).sum())
        girls_selected = int((selected_bool & girl_mask).sum())
        poorest_selected = int((selected_bool & poorest_mask).sum())

        observed_oos_captured = int(y[selected_bool].sum())
        weighted_oos_captured = float((weight[selected_bool] * y[selected_bool]).sum())

        risk_sum_selected = float(risk[selected_bool].sum())
        weighted_risk_sum_selected = float(weighted_risk[selected_bool].sum())
        objective_coeff_sum = float(
            np.asarray(
                {
                    "risk": risk,
                    "weighted_risk": weighted_risk,
                    "risk_pct_within_age": risk_pct_age,
                }[obj_name]
            )[selected_bool].sum()
        )

        total_weight_selected = float(weight[selected_bool].sum())

        row = {
            "rule": rule_name,
            "y_min": y_min,
            "y_min_share_of_K": safe_rate(y_min, K),
            "status": "Optimal_greedy_exact",
            "selected_n": selected_n,
            "young_selected": young_selected,
            "young_share_selected": safe_rate(young_selected, selected_n),
            "young_share_population": safe_rate(young_pop, n),
            "girls_selected": girls_selected,
            "girls_share_selected": safe_rate(girls_selected, selected_n),
            "girls_share_population": safe_rate(girl_pop, n),
            "poorest_selected": poorest_selected,
            "poorest_share_selected": safe_rate(poorest_selected, selected_n),
            "poorest_share_population": safe_rate(poorest_pop, n),
            "observed_oos_captured": observed_oos_captured,
            "unweighted_capture_rate": safe_rate(observed_oos_captured, total_oos),
            "weighted_oos_captured": weighted_oos_captured,
            "weighted_capture_rate": safe_rate(
                weighted_oos_captured,
                total_weighted_oos_mass,
            ),
            "risk_sum_selected": risk_sum_selected,
            "weighted_risk_sum_selected": weighted_risk_sum_selected,
            "objective_coeff_sum": objective_coeff_sum,
            "total_weight_selected": total_weight_selected,
            "weighted_population_share_selected": safe_rate(
                total_weight_selected,
                total_weight,
            ),
            "check_selected_n_equals_K": selected_n == K,
            "check_young_floor_met": young_selected >= y_min,
        }

        for b in ["6-9", "10-12", "13-15", "16-17"]:
            mask = band == b
            cnt = int((selected_bool & mask).sum())
            row[f"band_{b}_selected"] = cnt
            row[f"band_{b}_share_selected"] = safe_rate(cnt, selected_n)
            row[f"band_{b}_share_population"] = safe_rate(int(mask.sum()), n)

        return row

    p("")
    p("RUNNING SCENARIOS")
    p("-" * 80)

    for rule_name, obj in rules:
        if rule_name == "max_unweighted_risk":
            obj_name = "risk"
        elif rule_name == "max_weighted_risk":
            obj_name = "weighted_risk"
        else:
            obj_name = "risk"

        for y_min in y_mins:
            selected = select_with_young_floor(
                obj=obj,
                y_min=y_min,
                n=n,
                K=K,
                young_mask=young_mask,
                uid=uid,
            )

            if selected is None:
                rows.append(
                    {
                        "rule": rule_name,
                        "y_min": y_min,
                        "status": "Infeasible_young_floor_too_high",
                        "selected_n": 0,
                    }
                )
                p(f"{rule_name}, y_min={y_min}: infeasible")
                continue

            row = make_row(rule_name, y_min, selected, obj_name)
            rows.append(row)

            p(
                f"{rule_name}, y_min={y_min}: "
                f"young={row['young_selected']}, "
                f"captured_oos={row['observed_oos_captured']}, "
                f"risk_sum={row['risk_sum_selected']:.4f}"
            )

    # Diagnostic: top by age-standardized percentile, no young floor.
    if risk_pct_age is not None:
        selected = select_with_young_floor(
            obj=risk_pct_age,
            y_min=0,
            n=n,
            K=K,
            young_mask=young_mask,
            uid=uid,
        )

        if selected is not None:
            row = make_row(
                "top_age_standardized_percentile",
                0,
                selected,
                "risk_pct_within_age",
            )
            rows.append(row)

            p(
                "top_age_standardized_percentile, y_min=0: "
                f"young={row['young_selected']}, "
                f"captured_oos={row['observed_oos_captured']}, "
                f"risk_sum={row['risk_sum_selected']:.4f}"
            )

    if not rows:
        p("")
        p("VERDICT: FAIL - no scenarios produced.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    res = pd.DataFrame(rows)

    # Sort for readability.
    res = res.sort_values(["rule", "y_min"], na_position="last").reset_index(drop=True)

    # Price of young floor, computed within each rule against that rule's y_min=0 baseline.
    for rule_name in res["rule"].unique():
        mask = res["rule"] == rule_name
        base = res.loc[mask & (res["y_min"] == 0)]

        if len(base) == 0:
            continue

        base_risk = float(base["risk_sum_selected"].iloc[0])
        base_capture = float(base["observed_oos_captured"].iloc[0])
        base_young = float(base["young_selected"].iloc[0])

        res.loc[mask, "risk_sum_drop_vs_rule_y0"] = (
            base_risk - res.loc[mask, "risk_sum_selected"]
        )
        res.loc[mask, "capture_drop_vs_rule_y0"] = (
            base_capture - res.loc[mask, "observed_oos_captured"]
        )
        res.loc[mask, "additional_young_vs_rule_y0"] = (
            res.loc[mask, "young_selected"] - base_young
        )

        additional = res.loc[mask, "additional_young_vs_rule_y0"]
        capture_drop = res.loc[mask, "capture_drop_vs_rule_y0"]

        res.loc[mask, "capture_loss_per_additional_young"] = np.where(
            additional > 0,
            capture_drop / additional,
            np.nan,
        )

    p("")
    p("AGGREGATE RESULTS")
    p("-" * 80)

    key_cols = [
        "rule",
        "y_min",
        "status",
        "selected_n",
        "young_selected",
        "young_share_selected",
        "girls_share_selected",
        "poorest_share_selected",
        "observed_oos_captured",
        "unweighted_capture_rate",
        "weighted_capture_rate",
        "risk_sum_selected",
        "weighted_risk_sum_selected",
        "weighted_population_share_selected",
    ]

    key_cols = [c for c in key_cols if c in res.columns]

    p(res[key_cols].to_string(index=False))

    p("")
    p("BAND COMPOSITION")
    p("-" * 80)

    band_cols = [
        "rule",
        "y_min",
        "band_6-9_share_selected",
        "band_10-12_share_selected",
        "band_13-15_share_selected",
        "band_16-17_share_selected",
    ]

    band_cols = [c for c in band_cols if c in res.columns]

    p(res[band_cols].to_string(index=False))

    p("")
    p("PRICE OF YOUNG FLOOR, WITHIN EACH RULE")
    p("-" * 80)

    price_cols = [
        "rule",
        "y_min",
        "young_selected",
        "additional_young_vs_rule_y0",
        "risk_sum_drop_vs_rule_y0",
        "capture_drop_vs_rule_y0",
        "capture_loss_per_additional_young",
    ]

    price_cols = [c for c in price_cols if c in res.columns]

    p(res[price_cols].to_string(index=False))

    p("")
    p("SAVE AGGREGATE CSV")
    p("-" * 80)

    res.to_csv(OUT_CSV, index=False)

    p(f"Saved aggregate CSV: {OUT_CSV.resolve()}")
    p("This CSV contains one row per scenario and no child-level identifiers.")

    p("")
    p("VERDICT")
    p("-" * 80)

    failed = res.loc[res["status"].astype(str).str.contains("FAIL|Infeasible", case=False, na=False)]

    if len(failed) > 0:
        p("GREEDY SWEEP STATUS: PARTIAL")
        p("Some scenarios were infeasible or failed. See status column.")
    else:
        p("GREEDY SWEEP STATUS: PASS")
        p("All produced scenarios are exact greedy optima for the unit-count young-floor problem.")

    p("")
    p("IMPORTANT INTERPRETATION")
    p("-" * 80)
    p("- max_unweighted_risk selects children to maximize sum of predicted probabilities.")
    p("- max_weighted_risk selects children to maximize weighted predicted population reach.")
    p("- Observed Out_of_School is used only for evaluation, not as a decision input.")
    p("- y_min=0 is the baseline for each rule.")
    p("- The proportional young floor is about 423 selected children under age 13.")
    p("- If weighted risk captures more observed out-of-school children, it is a stronger allocation rule.")
    p("- If unweighted risk captures more but weighted risk gives better population share, report the trade-off.")
    p("- This sweep does not yet enforce gender, wealth, disability, ethnic-minority, or PSU caps.")
    p("- This sweep is sample-mode: it selects 730 sampled children. Population-weighted budgeting is separate.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/phase3_greedy_young_floor_sweep.txt.")
    p("You may also send results/phase3_greedy_young_floor_sweep.csv because it is aggregate only.")
    p("Do not send data/processed/chattogram_rural_phase3_optimization_input.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()