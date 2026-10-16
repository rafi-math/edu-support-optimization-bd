from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


RAW = Path("data/raw/fs.sav")
OUT = Path("data/processed/chattogram_rural_phase3_clean.csv")
REPORT = Path("reports/phase3_clean_build.txt")

EXPECTED_ROWS = 4891

USECOLS = [
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
    "fsweight",
    "fshweight",
    "windex5",
    "melevel",
    "ethnicity",
    "fsdisability",
    "CL1A",
    "PR3",
]

DERIVED_COLS = [
    "age",
    "female",
    "Out_of_School",
    "_band",
    "wealth_q",
    "mom_edu",
    "eth_minor",
    "disabled",
    "worked",
    "books_missing_data",
    "no_books",
    "child_weight_fsweight",
    "normalized_weight_fshweight",
    "interview_completed",
    "has_positive_child_weight",
    "weighted_analysis_eligible",
    "core_complete",
    "modeling_ready",
]

CORE_COMPLETE_COLS = [
    "age",
    "female",
    "Out_of_School",
    "PSU",
    "wealth_q",
    "mom_edu",
    "eth_minor",
    "disabled",
    "worked",
    "books_missing_data",
]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def as_int64(series):
    return num(series).astype("Int64")


def binary_from_code(series, true_code, missing_codes=()):
    s = num(series)

    if missing_codes:
        s = s.replace({float(c): np.nan for c in missing_codes})

    return pd.Series(
        np.where(s.isna(), np.nan, (s == float(true_code)).astype(float)),
        index=s.index,
    )


def weighted_mean(values, weights):
    v = num(values)
    w = num(weights)

    mask = v.notna() & w.notna() & (w > 0)

    if int(mask.sum()) == 0:
        return np.nan, 0

    return float(np.average(v[mask], weights=w[mask])), int(mask.sum())


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise SystemExit(f"Raw file not found: {RAW.resolve()}")

    p("=" * 80)
    p("PHASE 3 CLEAN INPUT BUILD")
    p("=" * 80)
    p(f"RAW FILE: {RAW.resolve()}")
    p(f"OUTPUT FILE: {OUT.resolve()}")
    p(f"REPORT FILE: {REPORT.resolve()}")

    df, meta = pyreadstat.read_sav(
        str(RAW),
        apply_value_formats=False,
        formats_as_category=False,
        usecols=USECOLS,
    )

    p("")
    p("RAW LOAD")
    p("-" * 80)
    p(f"Raw shape after usecols: {df.shape}")

    missing_cols = [c for c in USECOLS if c not in df.columns]
    p(f"Missing required raw columns: {missing_cols if missing_cols else 'none'}")

    if missing_cols:
        p("")
        p("VERDICT: FAIL - required raw columns missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    hh6 = num(df["HH6"])
    hh7 = num(df["HH7"])
    cb3 = num(df["CB3"])
    cb7 = num(df["CB7"])

    mask_rural = hh6 == 2
    mask_chattogram = hh7 == 20
    mask_age = cb3.between(6, 17)
    mask_valid_cb7 = cb7.isin([1, 2])
    mask = mask_rural & mask_chattogram & mask_age & mask_valid_cb7

    base = df.loc[mask].copy().reset_index(drop=True)

    p("")
    p("FILTER")
    p("-" * 80)
    p(f"Rural rows: {int(mask_rural.sum())}")
    p(f"Chattogram rows: {int(mask_chattogram.sum())}")
    p(f"Age 6-17 rows: {int(mask_age.sum())}")
    p(f"Valid CB7 rows: {int(mask_valid_cb7.sum())}")
    p(f"Rural Chattogram age 6-17 valid CB7 rows: {len(base)}")
    p(f"Expected rows from Phase 2 audit: {EXPECTED_ROWS}")
    p(f"Row count matches expected: {len(base) == EXPECTED_ROWS}")

    if len(base) == 0:
        p("")
        p("VERDICT: FAIL - filtered sample is empty.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    out = pd.DataFrame(index=base.index)

    for col in [
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
        "fsweight",
        "fshweight",
        "windex5",
        "melevel",
        "ethnicity",
        "fsdisability",
        "CL1A",
        "PR3",
    ]:
        out[col] = base[col]

    # Convert likely integer ID/geography/source columns to nullable integer where possible.
    for col in [
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
        "windex5",
        "melevel",
        "ethnicity",
        "fsdisability",
        "CL1A",
        "PR3",
    ]:
        out[col] = as_int64(out[col])

    # Keep weights numeric, not integer.
    out["fsweight"] = num(out["fsweight"])
    out["fshweight"] = num(out["fshweight"])

    p("")
    p("DERIVED FEATURES")
    p("-" * 80)

    out["age"] = num(out["CB3"])

    out["female"] = binary_from_code(out["HL4"], true_code=2, missing_codes=[9])
    out["Out_of_School"] = binary_from_code(out["CB7"], true_code=2, missing_codes=[9])

    bands = pd.cut(
        out["age"],
        bins=[5, 9, 12, 15, 17],
        labels=["6-9", "10-12", "13-15", "16-17"],
        right=True,
        include_lowest=True,
    )
    out["_band"] = bands.astype("string")

    # Verified from Phase 2: windex5 == 0 becomes missing.
    out["wealth_q"] = num(out["windex5"]).replace(0.0, np.nan)

    # Mother's education: code 9 is missing/DK.
    out["mom_edu"] = num(out["melevel"]).replace({9.0: np.nan})

    # Ethnic minority: ethnicity == 2 means other than Bengali.
    out["eth_minor"] = binary_from_code(out["ethnicity"], true_code=2, missing_codes=[9])

    # Child functional difficulty: fsdisability == 1 means has difficulty.
    out["disabled"] = binary_from_code(out["fsdisability"], true_code=1, missing_codes=[9])

    # Worked in past week: CL1A == 1 means yes. Code 9 is no response.
    out["worked"] = binary_from_code(out["CL1A"], true_code=1, missing_codes=[9])

    pr3 = num(out["PR3"])

    # Verified current Phase 2 rule: books_missing was actually PR3 missing/no response.
    out["books_missing_data"] = (pr3.isna() | (pr3 == 99)).astype(float)

    # True no-books flag, only defined when PR3 is valid.
    valid_pr3 = ~(pr3.isna() | (pr3 == 99))
    out["no_books"] = pd.Series(
        np.where(valid_pr3, (pr3 == 0).astype(float), np.nan),
        index=out.index,
    )

    # Weight columns.
    out["child_weight_fsweight"] = num(out["fsweight"])
    out["normalized_weight_fshweight"] = num(out["fshweight"])

    # Completion and eligibility flags.
    out["interview_completed"] = (num(out["FS17"]) == 1).astype(float)
    out["has_positive_child_weight"] = (out["child_weight_fsweight"] > 0).astype(float)

    out["weighted_analysis_eligible"] = (
        (out["interview_completed"] == 1)
        & (out["child_weight_fsweight"] > 0)
    ).astype(float)

    out["core_complete"] = out[CORE_COMPLETE_COLS].notna().all(axis=1).astype(float)

    out["modeling_ready"] = (
        (out["weighted_analysis_eligible"] == 1)
        & (out["core_complete"] == 1)
    ).astype(float)

    p("Derived columns added:")
    for col in DERIVED_COLS:
        p(f"- {col}")

    p("")
    p("MISSING VALUES IN DERIVED COLUMNS")
    p("-" * 80)

    missing_summary = out[DERIVED_COLS].isna().sum()
    p(missing_summary.to_string())

    p("")
    p("BINARY / OUTCOME CHECKS")
    p("-" * 80)

    for col in [
        "female",
        "Out_of_School",
        "eth_minor",
        "disabled",
        "worked",
        "books_missing_data",
        "no_books",
        "interview_completed",
        "has_positive_child_weight",
        "weighted_analysis_eligible",
        "core_complete",
        "modeling_ready",
    ]:
        p("")
        p(f"{col}:")
        p(out[col].value_counts(dropna=False).sort_index().to_string())

    p("")
    p("AGE BAND CHECK")
    p("-" * 80)
    p(out["_band"].value_counts(dropna=False).to_string())

    p("")
    p("WEIGHT CHECKS")
    p("-" * 80)

    for col in ["child_weight_fsweight", "normalized_weight_fshweight"]:
        s = out[col]
        p(
            f"{col}: missing={int(s.isna().sum())}, "
            f"zero={int((s == 0).sum())}, "
            f"positive={int((s > 0).sum())}, "
            f"min={s.min()}, max={s.max()}, sum={s.sum()}"
        )

    p("")
    p("WEIGHTED OUT-OF-SCHOOL RATES")
    p("-" * 80)

    unweighted_rate = float(out["Out_of_School"].mean(skipna=True))
    p(f"Unweighted OOS rate: {unweighted_rate:.6f}")

    fs_rate, fs_n = weighted_mean(out["Out_of_School"], out["child_weight_fsweight"])
    fsh_rate, fsh_n = weighted_mean(
        out["Out_of_School"],
        out["normalized_weight_fshweight"],
    )

    p(f"child_weight_fsweight weighted OOS rate: {fs_rate:.6f}, usable rows={fs_n}")
    p(f"normalized_weight_fshweight weighted OOS rate: {fsh_rate:.6f}, usable rows={fsh_n}")

    p("")
    p("BOOKS FEATURE CHECK")
    p("-" * 80)

    p(f"PR3 missing or 99: {int((pr3.isna() | (pr3 == 99)).sum())}")
    p(f"PR3 valid: {int(valid_pr3.sum())}")
    p(f"no_books == 1 among valid PR3: {int((out['no_books'] == 1).sum())}")
    p(f"no_books == 0 among valid PR3: {int((out['no_books'] == 0).sum())}")
    p(f"no_books missing: {int(out['no_books'].isna().sum())}")
    p(f"books_missing_data == 1: {int((out['books_missing_data'] == 1).sum())}")
    p(f"books_missing_data == 0: {int((out['books_missing_data'] == 0).sum())}")

    p("")
    p("ELIGIBILITY CHECK")
    p("-" * 80)

    p(f"Total rows: {len(out)}")
    p(f"Interview completed FS17 == 1: {int((out['interview_completed'] == 1).sum())}")
    p(f"Positive child_weight_fsweight: {int((out['has_positive_child_weight'] == 1).sum())}")
    p(f"Weighted analysis eligible: {int((out['weighted_analysis_eligible'] == 1).sum())}")
    p(f"Core complete: {int((out['core_complete'] == 1).sum())}")
    p(f"Modeling ready: {int((out['modeling_ready'] == 1).sum())}")

    p("")
    p("MODELING READY ROWS BY WEIGHT")
    p("-" * 80)

    ready = out.loc[out["modeling_ready"] == 1]

    if len(ready) > 0:
        ready_unweighted = float(ready["Out_of_School"].mean(skipna=True))
        ready_fs_rate, ready_fs_n = weighted_mean(
            ready["Out_of_School"],
            ready["child_weight_fsweight"],
        )
        ready_fsh_rate, ready_fsh_n = weighted_mean(
            ready["Out_of_School"],
            ready["normalized_weight_fshweight"],
        )

        p(f"Modeling ready rows: {len(ready)}")
        p(f"Modeling ready unweighted OOS rate: {ready_unweighted:.6f}")
        p(f"Modeling ready child_weight_fsweight OOS rate: {ready_fs_rate:.6f}, usable rows={ready_fs_n}")
        p(f"Modeling ready normalized_weight_fshweight OOS rate: {ready_fsh_rate:.6f}, usable rows={ready_fsh_n}")
    else:
        p("No modeling ready rows.")

    p("")
    p("SAVE OUTPUT")
    p("-" * 80)

    out.to_csv(OUT, index=False)

    p(f"Saved Phase 3 clean input: {OUT.resolve()}")
    p(f"Output shape: {out.shape}")

    p("")
    p("IMPORTANT NOTES")
    p("-" * 80)
    p("This file intentionally does not contain risk_score or risk_std.")
    p("The old Phase 2 risk scores are not locally reproducible, so they are not used here.")
    p("child_weight_fsweight is the provisional primary child-level weight.")
    p("normalized_weight_fshweight is retained for sensitivity analysis.")
    p("books_missing_data is a missing-data indicator, not a no-books indicator.")
    p("no_books is the true no-books indicator, but it is missing when PR3 is missing/no response.")
    p("Do not commit this CSV if it contains survey microdata identifiers. Keep it local unless your data policy allows publishing derived microdata.")

    p("")
    p("VERDICT")
    p("-" * 80)

    if len(base) == EXPECTED_ROWS:
        p("PHASE 3 CLEAN INPUT STATUS: PASS")
        p("Row count matches the verified Phase 2 rural Chattogram sample.")
    else:
        p("PHASE 3 CLEAN INPUT STATUS: WARNING")
        p("Row count does not match the expected Phase 2 sample. Check the filter before modeling.")

    p("")
    p("NEXT STEP AFTER THIS REPORT")
    p("-" * 80)
    p("1. Send reports/phase3_clean_build.txt.")
    p("2. If PASS, build a local reproducible risk-scoring script using only modeling_ready rows.")
    p("3. Use grouped cross-validation by PSU.")
    p("4. Compare child_weight_fsweight and normalized_weight_fshweight in evaluation.")
    p("5. Only after that, rebuild the optimization input with verified risk scores and original IDs.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()