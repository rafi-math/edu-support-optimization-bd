from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/processed/chattogram_rural_phase3_scored.csv")
OUTPUT = Path("data/processed/chattogram_rural_phase3_optimization_input.csv")
REPORT = Path("reports/phase3_optimization_input_audit.txt")

TOP_FRACTION = 0.15

CRITICAL_COLS = [
    "child_uid",
    "HH1",
    "HH2",
    "LN",
    "PSU",
    "stratum",
    "age",
    "_band",
    "female",
    "Out_of_School",
    "child_weight_fsweight",
    "normalized_weight_fshweight",
    "risk_score_oof",
]

OPTIONAL_DIAG_COLS = [
    "wealth_q",
    "mom_edu",
    "eth_minor",
    "disabled",
    "worked",
    "books_missing_data",
    "no_books",
]

COMPOSITION_COLS = [
    "_band",
    "female",
    "wealth_q",
    "eth_minor",
    "disabled",
    "worked",
]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def weighted_rate(values, weights):
    v = num(values)
    w = num(weights)

    mask = v.notna() & w.notna() & (w > 0)

    if int(mask.sum()) == 0:
        return np.nan, 0

    rate = float(np.average(v[mask], weights=w[mask]))
    return rate, int(mask.sum())


def effective_sample_size(weights):
    w = num(weights)
    w = w[w > 0]

    if len(w) == 0:
        return np.nan

    return float((w.sum() ** 2) / (w.pow(2).sum()))


def capture_stats(top, all_df):
    y_top = num(top["Out_of_School"])
    y_all = num(all_df["Out_of_School"])

    w_top = num(top["child_weight_fsweight"])
    w_all = num(all_df["child_weight_fsweight"])

    total_oos = int(y_all.sum())
    captured_oos = int(y_top.sum())

    capture_rate = captured_oos / total_oos if total_oos > 0 else np.nan

    positive_all = y_all == 1
    positive_top = y_top == 1

    weight_total_positive = float(w_all[positive_all].sum())
    weight_captured_positive = float(w_top[positive_top].sum())

    weighted_capture_rate = (
        weight_captured_positive / weight_total_positive
        if weight_total_positive > 0
        else np.nan
    )

    return {
        "n_selected": len(top),
        "selected_share": len(top) / len(all_df) if len(all_df) > 0 else np.nan,
        "total_observed_out_of_school": total_oos,
        "captured_observed_out_of_school": captured_oos,
        "unweighted_capture_rate": capture_rate,
        "weighted_capture_rate_child_weight": weighted_capture_rate,
        "mean_risk_selected": float(num(top["risk_score_oof"]).mean()),
        "mean_risk_not_selected": float(
            num(all_df.loc[~all_df.index.isin(top.index), "risk_score_oof"]).mean()
        ),
    }


def print_composition(top, all_df, label, sort_col):
    p("")
    p(f"TOP {int(TOP_FRACTION * 100)}% BY {label}")
    p("-" * 80)

    n_top = int(np.ceil(TOP_FRACTION * len(all_df)))
    top = top.copy()

    p(f"Requested n_top: {n_top}")
    p(f"Actual top rows: {len(top)}")

    stats = capture_stats(top, all_df)

    for key, value in stats.items():
        if isinstance(value, float):
            p(f"{key}: {value:.6f}")
        else:
            p(f"{key}: {value}")

    p("")
    p("Composition of selected children:")

    for col in COMPOSITION_COLS:
        if col not in top.columns:
            continue

        p("")
        p(f"By {col}:")

        counts = top[col].value_counts(dropna=False).sort_index()
        shares = top[col].value_counts(normalize=True, dropna=False).mul(100).round(2).sort_index()

        comp = pd.DataFrame(
            {
                "selected_count": counts,
                "selected_share_percent": shares,
            }
        )

        if col in all_df.columns:
            all_counts = all_df[col].value_counts(dropna=False).sort_index()
            all_shares = all_df[col].value_counts(normalize=True, dropna=False).mul(100).round(2).sort_index()
            comp["population_count"] = all_counts
            comp["population_share_percent"] = all_shares

        p(comp.to_string())

    p("")
    p(f"Observed out-of-school among selected: {int(num(top['Out_of_School']).sum())}")
    p(f"Observed out-of-school rate among selected: {float(num(top['Out_of_School']).mean()):.6f}")


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise SystemExit(f"Scored input not found: {INPUT.resolve()}")

    p("=" * 80)
    p("PHASE 3 OPTIMIZATION INPUT BUILD AND AUDIT")
    p("=" * 80)
    p(f"INPUT: {INPUT.resolve()}")
    p(f"OUTPUT: {OUTPUT.resolve()}")
    p(f"REPORT: {REPORT.resolve()}")

    df = pd.read_csv(INPUT, low_memory=False)

    p("")
    p("INPUT CHECK")
    p("-" * 80)
    p(f"Scored input shape: {df.shape}")

    if "modeling_ready" in df.columns:
        before = len(df)
        df = df.loc[num(df["modeling_ready"]) == 1].copy().reset_index(drop=True)
        p(f"Rows before modeling_ready filter: {before}")
        p(f"Rows after modeling_ready filter: {len(df)}")
    else:
        p("modeling_ready column not found; assuming input already contains modeling-ready rows only.")

    required_input_cols = [
        "HH1",
        "HH2",
        "LN",
        "PSU",
        "stratum",
        "age",
        "_band",
        "female",
        "Out_of_School",
        "child_weight_fsweight",
        "normalized_weight_fshweight",
        "risk_score_oof",
    ]

    missing = [c for c in required_input_cols if c not in df.columns]
    p(f"Missing required input columns: {missing if missing else 'none'}")

    if missing:
        p("")
        p("VERDICT: FAIL - required columns missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("CREATE TRACEABLE IDs AND PRIORITY SCORES")
    p("-" * 80)

    df = df.copy()

    df["child_uid"] = (
        num(df["HH1"]).astype("Int64").astype(str)
        + "_"
        + num(df["HH2"]).astype("Int64").astype(str)
        + "_"
        + num(df["LN"]).astype("Int64").astype(str)
    )

    df["risk_score_oof"] = num(df["risk_score_oof"])
    df["child_weight_fsweight"] = num(df["child_weight_fsweight"])
    df["normalized_weight_fshweight"] = num(df["normalized_weight_fshweight"])
    df["Out_of_School"] = num(df["Out_of_School"])
    df["age"] = num(df["age"])
    df["female"] = num(df["female"])

    df["risk_pct_overall"] = df["risk_score_oof"].rank(method="average", pct=True)

    age_group = df.groupby("age")["risk_score_oof"]

    age_mean = age_group.transform("mean")
    age_std = age_group.transform(lambda s: s.std(ddof=0))

    df["risk_z_within_age"] = np.where(
        age_std > 1e-12,
        (df["risk_score_oof"] - age_mean) / age_std,
        0.0,
    )

    df["risk_pct_within_age"] = (
        df.groupby("age")["risk_score_oof"]
        .rank(method="average", pct=True)
    )

    p("Added columns:")
    p("- child_uid")
    p("- risk_pct_overall")
    p("- risk_z_within_age")
    p("- risk_pct_within_age")

    p("")
    p("SELECT OPTIMIZATION COLUMNS")
    p("-" * 80)

    opt_cols = [
        "child_uid",
        "HH1",
        "HH2",
        "LN",
        "PSU",
        "stratum",
        "HH6",
        "HH7",
        "HH7A",
        "CB3",
        "CB7",
        "HL4",
        "FS17",
        "age",
        "_band",
        "female",
        "wealth_q",
        "mom_edu",
        "eth_minor",
        "disabled",
        "worked",
        "books_missing_data",
        "no_books",
        "Out_of_School",
        "child_weight_fsweight",
        "normalized_weight_fshweight",
        "risk_score_oof",
        "risk_pct_overall",
        "risk_z_within_age",
        "risk_pct_within_age",
        "risk_model",
    ]

    opt_cols = [c for c in opt_cols if c in df.columns]

    out = df[opt_cols].copy()

    p(f"Output columns: {len(out.columns)}")
    p(str(list(out.columns)))

    p("")
    p("CRITICAL COMPLETENESS CHECK")
    p("-" * 80)

    critical_present = [c for c in CRITICAL_COLS if c in out.columns]
    critical_missing_cols = [c for c in CRITICAL_COLS if c not in out.columns]

    p(f"Critical columns present: {critical_present}")
    p(f"Critical columns absent: {critical_missing_cols if critical_missing_cols else 'none'}")

    if critical_missing_cols:
        p("")
        p("VERDICT: FAIL - critical columns absent.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    missing_summary = out[CRITICAL_COLS].isna().sum()
    p("")
    p("Missing values in critical columns:")
    p(missing_summary.to_string())

    if int(missing_summary.sum()) > 0:
        p("")
        p("VERDICT: FAIL - critical missing values remain.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("DUPLICATE AND ID CHECK")
    p("-" * 80)

    dup_uid = int(out.duplicated("child_uid").sum())
    dup_full = int(out.duplicated().sum())
    unique_children = int(out["child_uid"].nunique())
    unique_psu = int(num(out["PSU"]).nunique())

    p(f"Rows: {len(out)}")
    p(f"Unique child_uid: {unique_children}")
    p(f"Duplicate child_uid: {dup_uid}")
    p(f"Full duplicate rows: {dup_full}")
    p(f"Unique PSU: {unique_psu}")

    if dup_uid > 0:
        p("")
        p("VERDICT: FAIL - duplicate child_uid found.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("TARGET AND BINARY CHECKS")
    p("-" * 80)

    for col in ["female", "Out_of_School"]:
        s = num(out[col])
        p(f"{col}: missing={int(s.isna().sum())}, unique={sorted(s.dropna().unique().tolist())}")
        p(s.value_counts(dropna=False).sort_index().to_string())
        p("")

    bad_target = out.loc[~num(out["Out_of_School"]).isin([0, 1])]
    p(f"Rows with invalid Out_of_School: {len(bad_target)}")

    bad_female = out.loc[~num(out["female"]).isin([0, 1])]
    p(f"Rows with invalid female: {len(bad_female)}")

    if len(bad_target) > 0 or len(bad_female) > 0:
        p("")
        p("VERDICT: FAIL - invalid binary target or sex coding.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("AGE CHECK")
    p("-" * 80)

    age = num(out["age"])
    p(f"age min/max: {age.min()} / {age.max()}")
    p(f"rows outside 6-17: {int(((age < 6) | (age > 17)).sum())}")
    p("age value counts:")
    p(age.value_counts(dropna=False).sort_index().to_string())

    p("")
    p("_band value counts:")
    p(out["_band"].value_counts(dropna=False).to_string())

    if int(((age < 6) | (age > 17)).sum()) > 0:
        p("")
        p("VERDICT: FAIL - age outside expected range.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("WEIGHT CHECKS")
    p("-" * 80)

    for col in ["child_weight_fsweight", "normalized_weight_fshweight"]:
        w = num(out[col])
        p(
            f"{col}: missing={int(w.isna().sum())}, "
            f"zero={int((w == 0).sum())}, "
            f"negative={int((w < 0).sum())}, "
            f"positive={int((w > 0).sum())}, "
            f"min={w.min()}, max={w.max()}, sum={w.sum()}"
        )

    p("")
    p(f"Effective sample size, child_weight_fsweight: {effective_sample_size(out['child_weight_fsweight']):.2f}")
    p(f"Effective sample size, normalized_weight_fshweight: {effective_sample_size(out['normalized_weight_fshweight']):.2f}")

    w_child = num(out["child_weight_fsweight"])

    if int((w_child.isna() | (w_child <= 0)).sum()) > 0:
        p("")
        p("VERDICT: FAIL - child_weight_fsweight has missing or non-positive values.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("RISK SCORE CHECKS")
    p("-" * 80)

    risk = num(out["risk_score_oof"])
    risk_pct = num(out["risk_pct_overall"])
    risk_z = num(out["risk_z_within_age"])
    risk_pct_age = num(out["risk_pct_within_age"])

    p(f"risk_score_oof: missing={int(risk.isna().sum())}, finite={int(np.isfinite(risk).sum())}/{len(risk)}")
    p(f"risk_score_oof min/max: {risk.min():.6f} / {risk.max():.6f}")
    p(f"risk_score_oof outside [0,1]: {int(((risk < 0) | (risk > 1)).sum())}")

    p("")
    p(f"risk_pct_overall min/max: {risk_pct.min():.6f} / {risk_pct.max():.6f}")
    p(f"risk_z_within_age min/max: {risk_z.min():.6f} / {risk_z.max():.6f}")
    p(f"risk_pct_within_age min/max: {risk_pct_age.min():.6f} / {risk_pct_age.max():.6f}")

    p("")
    p("risk_score_oof distribution:")
    p(
        risk.describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).to_string()
    )

    if int(((risk < 0) | (risk > 1)).sum()) > 0 or int((~np.isfinite(risk)).sum()) > 0:
        p("")
        p("VERDICT: FAIL - invalid risk score.")
        out.to_csv(OUTPUT, index=False)
        p(f"Output saved anyway for inspection: {OUTPUT.resolve()}")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    p("")
    p("WEIGHTED OUT-OF-SCHOOL RATE")
    p("-" * 80)

    unweighted = float(num(out["Out_of_School"]).mean())
    child_rate, child_n = weighted_rate(out["Out_of_School"], out["child_weight_fsweight"])
    norm_rate, norm_n = weighted_rate(out["Out_of_School"], out["normalized_weight_fshweight"])

    p(f"Unweighted OOS rate: {unweighted:.6f}")
    p(f"child_weight_fsweight weighted OOS rate: {child_rate:.6f}, usable rows={child_n}")
    p(f"normalized_weight_fshweight weighted OOS rate: {norm_rate:.6f}, usable rows={norm_n}")

    p("")
    p("TOP SELECTION DIAGNOSTICS")
    p("=" * 80)

    n_top = int(np.ceil(TOP_FRACTION * len(out)))

    top_raw = out.nlargest(n_top, "risk_score_oof")
    print_composition(top_raw, out, "RAW RISK SCORE", "risk_score_oof")

    top_age_std = out.nlargest(n_top, "risk_pct_within_age")
    print_composition(top_age_std, out, "AGE-STANDARDIZED RISK PERCENTILE", "risk_pct_within_age")

    p("")
    p("RAW VS AGE-STANDARDIZED OVERLAP")
    p("-" * 80)

    raw_ids = set(top_raw["child_uid"])
    age_std_ids = set(top_age_std["child_uid"])

    overlap = raw_ids & age_std_ids

    p(f"Top raw count: {len(raw_ids)}")
    p(f"Top age-standardized count: {len(age_std_ids)}")
    p(f"Overlap count: {len(overlap)}")
    p(f"Overlap share of top raw: {len(overlap) / len(raw_ids) if raw_ids else np.nan:.6f}")
    p(f"Overlap share of top age-standardized: {len(overlap) / len(age_std_ids) if age_std_ids else np.nan:.6f}")

    p("")
    p("SAVE OUTPUT")
    p("-" * 80)

    out.to_csv(OUTPUT, index=False)

    p(f"Saved optimization input: {OUTPUT.resolve()}")
    p(f"Output shape: {out.shape}")
    p("Do not commit this file. It contains child-level survey identifiers and scores.")

    p("")
    p("VERDICT")
    p("-" * 80)

    p("OPTIMIZATION INPUT STATUS: PASS")
    p("All critical fields are complete, IDs are unique, weights are positive, and risk scores are finite.")
    p("")
    p("Important interpretation:")
    p("- risk_score_oof is a probability-like out-of-fold score and can be used for expected-reach MILP.")
    p("- risk_z_within_age is a within-age standardized score. It can be negative and is not a probability.")
    p("- risk_pct_within_age is a within-age percentile. It is useful for age-balanced ranking diagnostics.")
    p("- Do not use Out_of_School as a model input or optimization decision variable. It is for evaluation only.")
    p("- child_weight_fsweight is the provisional primary weight. normalized_weight_fshweight is for sensitivity only.")

    p("")
    p("NEXT STEP AFTER THIS REPORT")
    p("-" * 80)
    p("1. Send reports/phase3_optimization_input_audit.txt.")
    p("2. If PASS, install a working MILP solver.")
    p("3. Run a small budgeted selection MILP using risk_score_oof.")
    p("4. Compare raw-risk selection against age-standardized selection and young-floor constraints.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()