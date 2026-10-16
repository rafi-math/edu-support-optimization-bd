from pathlib import Path
import argparse
import numpy as np
import pandas as pd


EXPECTED = [
    "sid",
    "risk_score",
    "risk_std",
    "female",
    "age",
    "_band",
    "wealth_q",
    "mom_edu",
    "eth_minor",
    "disabled",
    "worked",
    "books_missing",
    "Out_of_School",
    "FSWEIGHT",
    "_group",
]

CRITICAL = [
    "sid",
    "risk_score",
    "risk_std",
    "female",
    "age",
    "Out_of_School",
    "FSWEIGHT",
    "_group",
]

BINARY = [
    "female",
    "eth_minor",
    "disabled",
    "worked",
    "books_missing",
    "Out_of_School",
]

ORDINAL = [
    "wealth_q",
    "mom_edu",
]


def as_num(series):
    return pd.to_numeric(series, errors="coerce")


def fmt(value):
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(
        description="Audit the rural Chattogram Phase 2 input CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="data/processed/chattogram_rural_phase2_input.csv",
        help="Path to the Phase 2 input CSV.",
    )
    parser.add_argument(
        "--report",
        default="reports/phase2_audit.txt",
        help="Path where the audit report will be saved.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")

    lines = []

    def p(message=""):
        text = str(message)
        print(text)
        lines.append(text)

    p("=" * 80)
    p("PHASE 2 INPUT AUDIT")
    p("=" * 80)
    p(f"PATH: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, low_memory=False)
    n_rows, n_cols = df.shape
    p(f"SHAPE: {df.shape}")

    missing_expected = [col for col in EXPECTED if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED]

    p(f"MISSING EXPECTED COLUMNS: {missing_expected if missing_expected else 'none'}")
    p(f"EXTRA COLUMNS: {extra_cols if extra_cols else 'none'}")

    if missing_expected:
        p("")
        p("VERDICT: FAIL - required columns are missing.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {report_path.resolve()}")
        return

    p("")
    p("DUPLICATES AND ID CHECK")
    p("-" * 80)

    full_dupes = int(df.duplicated().sum())
    p(f"Full duplicate rows: {full_dupes}")

    sid_num = as_num(df["sid"])
    sid_missing = int(sid_num.isna().sum())
    sid_unique = int(sid_num.nunique(dropna=True))
    sid_dupe_count = int(sid_num.duplicated(keep=False).sum())
    sid_min = sid_num.min()
    sid_max = sid_num.max()

    p(f"sid missing: {sid_missing}")
    p(f"sid unique: {sid_unique}")
    p(f"sid duplicate entries: {sid_dupe_count}")
    p(f"sid min/max: {fmt(sid_min)} / {fmt(sid_max)}")

    sid_sequential = False
    if sid_missing == 0 and sid_unique == n_rows:
        try:
            sid_int = sid_num.astype(int)
            sid_sequential = bool(
                sid_int.min() == 0
                and sid_int.max() == n_rows - 1
                and sid_int.sort_values().reset_index(drop=True).equals(
                    pd.Series(range(n_rows), dtype=int)
                )
            )
        except Exception:
            sid_sequential = False

    p(f"sid looks like a reset integer index: {sid_sequential}")
    if sid_sequential:
        p("WARNING: sid is probably not an original MICS identifier.")
        p("This file can still be used for a local dry run, but final reproducibility needs original IDs.")

    p("")
    p("AGE CHECK")
    p("-" * 80)

    age = as_num(df["age"])
    p(f"age missing: {int(age.isna().sum())}")
    p(f"age min/max: {fmt(age.min())} / {fmt(age.max())}")
    p(f"rows with age outside 6-17: {int(((age < 6) | (age > 17)).sum())}")
    p(f"age == 9 count: {int((age == 9).sum())}")
    p("Note: age 9 is a valid child age. Do not recode it as missing.")

    p("")
    p("age value counts:")
    p(age.value_counts(dropna=False).sort_index().to_string())

    p("")
    p("AGE BAND CHECK")
    p("-" * 80)
    p("_band value counts:")
    p(df["_band"].value_counts(dropna=False).to_string())

    p("")
    p("BINARY AND TARGET CHECKS")
    p("-" * 80)

    for col in BINARY:
        s = as_num(df[col])
        unique_vals = sorted(s.dropna().unique().tolist())
        p(f"{col}: missing={int(s.isna().sum())}, unique={unique_vals}")
        p(s.value_counts(dropna=False).sort_index().to_string())

        bad = s.dropna()[~s.dropna().isin([0, 1])]
        if len(bad) > 0:
            p(f"WARNING: {col} has non-binary values: {bad.unique()[:10]}")
        p("")

    p("ORDINAL CHECKS")
    p("-" * 80)

    for col in ORDINAL:
        s = as_num(df[col])
        p(f"{col}: missing={int(s.isna().sum())}, min={fmt(s.min())}, max={fmt(s.max())}")
        p(s.value_counts(dropna=False).sort_index().to_string())
        p("")

    wealth = as_num(df["wealth_q"])
    wealth_zero = int((wealth == 0).sum())
    p(f"wealth_q == 0 count: {wealth_zero}")
    p("In raw MICS wealth index, 0 can mean missing/unclassified. Flag if present.")

    p("")
    p("RISK SCORE CHECKS")
    p("-" * 80)

    risk_score = as_num(df["risk_score"])
    risk_std = as_num(df["risk_std"])

    p(f"risk_score missing: {int(risk_score.isna().sum())}")
    p(f"risk_score min/max: {fmt(risk_score.min())} / {fmt(risk_score.max())}")
    p(f"risk_score outside [0, 1]: {int(((risk_score < 0) | (risk_score > 1)).sum())}")

    p(f"risk_std missing: {int(risk_std.isna().sum())}")
    p(f"risk_std non-finite: {int((~np.isfinite(risk_std)).sum())}")
    p(f"risk_std min/max: {fmt(risk_std.min())} / {fmt(risk_std.max())}")
    p("Note: risk_std appears to be a standardized score, so negative values are expected.")

    p("")
    p("WEIGHT CHECKS")
    p("-" * 80)

    weight = as_num(df["FSWEIGHT"])
    outcome = as_num(df["Out_of_School"])

    p(f"FSWEIGHT missing: {int(weight.isna().sum())}")
    p(f"FSWEIGHT <= 0: {int((weight <= 0).sum())}")
    p(f"FSWEIGHT min/max: {fmt(weight.min())} / {fmt(weight.max())}")
    p(f"FSWEIGHT sum: {fmt(weight.sum())}")

    p("")
    p("FSWEIGHT unique sample:")
    p(str(weight.dropna().unique()[:20]))

    valid_weighted_rows = weight.notna() & (weight > 0) & outcome.notna()
    weighted_rows_used = int(valid_weighted_rows.sum())

    p("")
    p(f"Rows usable for weighted rate: {weighted_rows_used}")

    if weighted_rows_used > 0:
        weighted_oos_rate = float(
            np.average(
                outcome[valid_weighted_rows],
                weights=weight[valid_weighted_rows],
            )
        )
        p(f"Weighted out-of-school rate: {weighted_oos_rate:.6f}")
    else:
        p("Weighted out-of-school rate: not computable")

    unweighted_oos_rate = float(outcome.mean(skipna=True))
    unweighted_oos_count = int(outcome.sum(skipna=True))

    p(f"Unweighted out-of-school rate: {unweighted_oos_rate:.6f}")
    p(f"Unweighted out-of-school count: {unweighted_oos_count}")

    p("")
    p("GROUP / PSU CANDIDATE CHECK")
    p("-" * 80)

    group = as_num(df["_group"])
    group_missing = int(group.isna().sum())
    group_unique = int(group.nunique(dropna=True))

    p(f"_group missing: {group_missing}")
    p(f"_group unique: {group_unique}")
    p(f"_group min/max: {fmt(group.min())} / {fmt(group.max())}")

    p("")
    p("_group value counts, first 20:")
    p(group.value_counts(dropna=False).head(20).to_string())

    if group_missing > 0:
        p("WARNING: _group has missing values.")
    if group_unique == n_rows:
        p("WARNING: _group has one value per row. It is not a cluster/PSU.")
    elif group_unique >= 50:
        p("CANDIDATE PASS: _group has many repeated values and may be PSU.")
        p("This still needs confirmation against the raw fs.sav PSU variable.")
    else:
        p("WARNING: _group has too few unique values to be the expected PSU set.")

    p("")
    p("MISSING VALUE SUMMARY")
    p("-" * 80)

    missing_summary = df[EXPECTED].isna().sum()
    p(missing_summary.to_string())

    p("")
    p("MISSING OVERLAP")
    p("-" * 80)

    miss_weight = df["FSWEIGHT"].isna()
    miss_wealth = df["wealth_q"].isna()
    miss_disabled = df["disabled"].isna()
    miss_worked = df["worked"].isna()

    p(f"FSWEIGHT missing: {int(miss_weight.sum())}")
    p(f"wealth_q missing: {int(miss_wealth.sum())}")
    p(f"disabled missing: {int(miss_disabled.sum())}")
    p(f"worked missing: {int(miss_worked.sum())}")

    p("")
    p(f"FSWEIGHT missing and wealth_q missing: {int((miss_weight & miss_wealth).sum())}")
    p(f"FSWEIGHT missing and disabled missing: {int((miss_weight & miss_disabled).sum())}")
    p(f"FSWEIGHT missing and worked missing: {int((miss_weight & miss_worked).sum())}")
    p(f"wealth_q missing and disabled missing: {int((miss_wealth & miss_disabled).sum())}")

    critical_missing_rows = int(df[CRITICAL].isna().any(axis=1).sum())
    p(f"Rows missing any critical optimization field: {critical_missing_rows}")

    p("")
    p("CODED-MISSING SCAN")
    p("-" * 80)

    magic_codes = [9, 99, 999, 9999, -9, -99, -999]
    found_magic = False

    for col in EXPECTED:
        if col == "_band":
            continue

        s = as_num(df[col])
        hits = {}

        for code in magic_codes:
            count = int((s == code).sum())
            if count > 0:
                hits[code] = count

        if hits:
            found_magic = True
            note = ""

            if col == "age" and 9 in hits:
                note = "  # age 9 is valid"
            elif col == "sid" and (9 in hits or 99 in hits):
                note = "  # sid values can be valid if unique"
            elif col == "wealth_q" and 9 in hits:
                note = "  # possible coded missing, check raw codebook"
            elif col == "mom_edu" and 9 in hits:
                note = "  # possible coded missing, check raw codebook"
            elif col in ["female", "disabled", "worked", "eth_minor", "books_missing"] and 9 in hits:
                note = "  # possible coded missing in yes/no variable"

            p(f"{col}: {hits}{note}")

    if not found_magic:
        p("No common magic codes found in expected numeric columns.")

    p("")
    p("WEIGHTED SEGMENT SUMMARY")
    p("-" * 80)

    segment_columns = [
        "_band",
        "female",
        "disabled",
        "worked",
        "books_missing",
        "wealth_q",
    ]

    for col in segment_columns:
        p("")
        p(f"By {col}:")

        if col == "_band":
            group_series = df[col]
        else:
            group_series = as_num(df[col])

        temp = pd.DataFrame(
            {
                "group": group_series,
                "y": outcome,
                "w": weight,
            }
        )
        temp = temp[temp["y"].notna()].copy()

        rows = []

        for group_value, grp in temp.groupby("group", dropna=False):
            n_all = len(grp)
            n_oos = int(grp["y"].sum())

            grp_weight = grp["w"]
            usable = grp_weight.notna() & (grp_weight > 0)

            if usable.sum() > 0:
                w_all = float(grp_weight[usable].sum())
                y_usable = grp.loc[usable, "y"]
                w_usable = grp_weight[usable]
                w_oos = float(w_usable[y_usable == 1].sum())
                rate = w_oos / w_all if w_all > 0 else np.nan
            else:
                w_all = np.nan
                w_oos = np.nan
                rate = np.nan

            rows.append(
                {
                    "group": group_value,
                    "n_all": n_all,
                    "n_oos": n_oos,
                    "unweighted_rate": n_oos / n_all if n_all > 0 else np.nan,
                    "weighted_n_all": w_all,
                    "weighted_n_oos": w_oos,
                    "weighted_rate": rate,
                }
            )

        seg_df = pd.DataFrame(rows)

        try:
            seg_df = seg_df.sort_values("group")
        except TypeError:
            seg_df = seg_df.sort_values("group", key=lambda x: x.astype(str))

        p(seg_df.to_string(index=False))

    p("")
    p("VERDICT")
    p("-" * 80)

    fails = []
    warnings = []
    readiness_blockers = []

    if sid_dupe_count > 0:
        fails.append("Duplicate sid values found.")

    if int(((age < 6) | (age > 17)).sum()) > 0:
        fails.append("Rows outside age 6-17 found.")

    if int(((risk_score < 0) | (risk_score > 1)).sum()) > 0:
        fails.append("risk_score outside [0, 1].")

    if int((~np.isfinite(risk_std)).sum()) > 0:
        fails.append("risk_std has non-finite values.")

    if group_missing > 0:
        fails.append("_group has missing values.")

    if group_unique == n_rows:
        warnings.append("_group is row-level, not a PSU/cluster.")
        readiness_blockers.append("_group is not usable as PSU")

    if int(weight.isna().sum()) > 0:
        warnings.append(f"FSWEIGHT missing in {int(weight.isna().sum())} rows.")
        readiness_blockers.append("missing FSWEIGHT")

    if int((weight <= 0).sum()) > 0:
        warnings.append(f"FSWEIGHT <= 0 in {int((weight <= 0).sum())} rows.")
        readiness_blockers.append("non-positive FSWEIGHT")

    if sid_sequential:
        warnings.append("sid is synthetic/reset index, not original MICS traceability.")
        readiness_blockers.append("no original MICS IDs")

    warnings.append("HH6 and HH7 are absent, so rural Chattogram filtering cannot be verified from this CSV alone.")
    readiness_blockers.append("geography provenance not present in file")

    if fails:
        p("STRUCTURAL STATUS: FAIL")
        for item in fails:
            p(f"- {item}")
    else:
        p("STRUCTURAL STATUS: PASS")

    if warnings:
        p("")
        p("WARNINGS")
        for item in warnings:
            p(f"- {item}")

    p("")
    if readiness_blockers:
        p("FINAL OPTIMIZATION READY: NO")
        p("Blockers:")
        for item in readiness_blockers:
            p(f"- {item}")
    else:
        p("FINAL OPTIMIZATION READY: YES")

    p("")
    if fails:
        p("LOCAL DRY RUN READY: NO")
    else:
        p("LOCAL DRY RUN READY: YES, after the next cleaning/provenance step")

    p("")
    p("END OF AUDIT")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    p(f"REPORT SAVED: {report_path.resolve()}")


if __name__ == "__main__":
    main()