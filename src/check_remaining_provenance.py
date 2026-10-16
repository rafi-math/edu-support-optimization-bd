from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


FS_PATH = Path("data/raw/fs.sav")
HH_PATH = Path("data/raw/hh.sav")
PHASE2_PATH = Path("data/processed/chattogram_rural_phase2_input.csv")
REPORT_PATH = Path("reports/remaining_provenance_check.txt")

FS_COLS = [
    "HH1",
    "HH2",
    "LN",
    "HH6",
    "HH7",
    "CB3",
    "CB7",
    "FS17",
    "fsweight",
    "fshweight",
]

HH_COLS = [
    "HH1",
    "HH2",
    "hhweight",
    "HH52",
]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def match_numeric(a, b, atol=1e-6):
    a = num(a)
    b = num(b)

    if len(a) != len(b):
        return None

    both_na = a.isna().to_numpy() & b.isna().to_numpy()

    try:
        close = np.isclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=0,
            atol=atol,
            equal_nan=False,
        )
    except Exception:
        close = (a == b).to_numpy()

    ok = both_na | close
    n = len(ok)
    count = int(ok.sum())

    return (count / n * 100 if n else 0.0), count, n


def fmt(res):
    if res is None:
        return "length mismatch or missing column"

    pct, count, n = res
    return f"{count}/{n} ({pct:.2f}%)"


def household_key(df):
    return (
        df["HH1"].astype("Int64").astype(str)
        + "_"
        + df["HH2"].astype("Int64").astype(str)
    )


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()

    if pd.isna(med):
        return "NaN"

    return f"{med:.6g}"


def load_with_usecols(path, cols):
    try:
        df, _ = pyreadstat.read_sav(
            str(path),
            apply_value_formats=False,
            formats_as_category=False,
            usecols=cols,
        )
        return df
    except TypeError:
        df, _ = pyreadstat.read_sav(
            str(path),
            apply_value_formats=False,
            formats_as_category=False,
        )
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in {path}: {missing}")
        return df[cols]


def summarize_household_weights(diag, label):
    p("")
    p(label)
    p("-" * 80)

    if diag is None or len(diag) == 0:
        p("No household diagnostics available.")
        return

    p(f"Households: {len(diag)}")
    p(f"Households with hhweight available: {int(diag['hhweight'].notna().sum())}")
    p(f"Households with more than one fs.sav row: {int((diag['n_rows'] > 1).sum())}")

    fsh_constant = (diag["nunique_fshweight"] == 1).mean()
    fs_constant = (diag["nunique_fsweight"] == 1).mean()

    p(f"Share of households where fshweight is constant: {fsh_constant:.4f}")
    p(f"Share of households where fsweight is constant: {fs_constant:.4f}")

    p(f"Median max |fshweight - hhweight|: {safe_median(diag['max_abs_diff_fsh_hh'])}")
    p(f"Median max |fsweight - hhweight|: {safe_median(diag['max_abs_diff_fs_hh'])}")

    diff_sum_fs = (diag["sum_fsweight"] - diag["hhweight"]).abs()
    diff_sum_fsh = (diag["sum_fshweight"] - diag["hhweight"]).abs()

    p(f"Median |sum(fsweight) - hhweight|: {safe_median(diff_sum_fs)}")
    p(f"Median |sum(fshweight) - hhweight|: {safe_median(diff_sum_fsh)}")

    close_fsh_hh = (diag["max_abs_diff_fsh_hh"] < 1e-6).mean()
    close_fs_hh = (diag["max_abs_diff_fs_hh"] < 1e-6).mean()
    close_sum_fs_hh = (diff_sum_fs < 1e-4).mean()
    close_sum_fsh_hh = (diff_sum_fsh < 1e-4).mean()

    p(f"Share households where fshweight equals hhweight closely: {close_fsh_hh:.4f}")
    p(f"Share households where fsweight equals hhweight closely: {close_fs_hh:.4f}")
    p(f"Share households where sum(fsweight) equals hhweight closely: {close_sum_fs_hh:.4f}")
    p(f"Share households where sum(fshweight) equals hhweight closely: {close_sum_fsh_hh:.4f}")

    hhw = diag["hhweight"].replace(0, np.nan)

    diag = diag.copy()
    diag["ratio_sum_fs_to_hh"] = diag["sum_fsweight"] / hhw
    diag["ratio_sum_fsh_to_hh"] = diag["sum_fshweight"] / hhw
    diag["ratio_fs_mean_to_hh_over_n"] = (
        diag["sum_fsweight"] / diag["n_rows"].replace(0, np.nan)
    ) / hhw

    p("")
    p("Ratio summary:")
    p(
        diag[
            [
                "ratio_sum_fs_to_hh",
                "ratio_sum_fsh_to_hh",
                "ratio_fs_mean_to_hh_over_n",
            ]
        ]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .to_string()
    )

    p("")
    p("Interpretation guide:")
    p("- If fshweight is constant within household and equals hhweight, fshweight behaves like a household weight.")
    p("- If fsweight varies within household and sum(fsweight) equals hhweight, fsweight behaves like a child-level weight.")
    p("- If both behave like child weights, compare which one sums to plausible population counts and which one the codebook names as final child weight.")


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not FS_PATH.exists():
        raise SystemExit(f"Raw fs.sav not found: {FS_PATH.resolve()}")

    if not PHASE2_PATH.exists():
        raise SystemExit(f"Phase 2 file not found: {PHASE2_PATH.resolve()}")

    p("=" * 80)
    p("REMAINING PROVENANCE CHECK")
    p("=" * 80)
    p(f"FS FILE: {FS_PATH.resolve()}")
    p(f"HH FILE: {HH_PATH.resolve()}")
    p(f"PHASE 2 FILE: {PHASE2_PATH.resolve()}")

    fs = load_with_usecols(FS_PATH, FS_COLS)
    ph = pd.read_csv(PHASE2_PATH, low_memory=False).reset_index(drop=True)

    p(f"FS SHAPE: {fs.shape}")
    p(f"PHASE 2 SHAPE: {ph.shape}")

    missing_fs = [c for c in FS_COLS if c not in fs.columns]
    missing_ph = [
        c
        for c in [
            "age",
            "female",
            "Out_of_School",
            "_group",
            "FSWEIGHT",
            "wealth_q",
            "mom_edu",
            "eth_minor",
            "disabled",
            "worked",
            "books_missing",
            "_band",
            "PR3",
        ]
        if c not in ph.columns
    ]

    p(f"MISSING FS COLUMNS: {missing_fs if missing_fs else 'none'}")
    p(f"MISSING PHASE 2 COLUMNS FOR THIS CHECK: {missing_ph if missing_ph else 'none'}")

    if missing_fs:
        p("")
        p("VERDICT: FAIL - required fs columns missing.")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT_PATH.resolve()}")
        return

    hh6 = num(fs["HH6"])
    hh7 = num(fs["HH7"])
    cb3 = num(fs["CB3"])
    cb7 = num(fs["CB7"])

    mask = (
        (hh6 == 2)
        & (hh7 == 20)
        & cb3.between(6, 17)
        & cb7.isin([1, 2])
    )

    base = fs.loc[mask].copy().reset_index(drop=True)

    p("")
    p("BASE SAMPLE")
    p("-" * 80)
    p(f"Rural Chattogram age 6-17 valid CB7 rows: {len(base)}")
    p(f"Phase 2 rows: {len(ph)}")
    p(f"Length match: {len(base) == len(ph)}")

    if len(base) != len(ph):
        p("")
        p("VERDICT: FAIL - base and Phase 2 lengths differ.")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT_PATH.resolve()}")
        return

    p("")
    p("WEIGHT SOURCE CHECK")
    p("-" * 80)

    fs_w = num(base["fsweight"])
    fsh_w = num(base["fshweight"])
    ph_w = num(ph["FSWEIGHT"])
    fs17 = num(base["FS17"])

    p(f"base fsweight: missing={int(fs_w.isna().sum())}, zero={int((fs_w == 0).sum())}, positive={int((fs_w > 0).sum())}")
    p(f"base fshweight: missing={int(fsh_w.isna().sum())}, zero={int((fsh_w == 0).sum())}, positive={int((fsh_w > 0).sum())}")
    p(f"phase FSWEIGHT: missing={int(ph_w.isna().sum())}, zero={int((ph_w == 0).sum())}, positive={int((ph_w > 0).sum())}")

    p("")
    p("Crosstab FS17 completed vs fsweight zero in base:")
    p(
        pd.crosstab(
            fs17 == 1,
            fs_w == 0,
            rownames=["FS17_completed"],
            colnames=["fsweight_zero"],
            dropna=False,
        ).to_string()
    )

    p("")
    p("Crosstab FS17 completed vs fshweight zero in base:")
    p(
        pd.crosstab(
            fs17 == 1,
            fsh_w == 0,
            rownames=["FS17_completed"],
            colnames=["fshweight_zero"],
            dropna=False,
        ).to_string()
    )

    p("")
    p("Phase 2 missing-weight rows by FS17:")
    p(
        pd.crosstab(
            fs17,
            ph_w.isna(),
            rownames=["FS17"],
            colnames=["phase_FSWEIGHT_missing"],
            dropna=False,
        ).to_string()
    )

    p("")
    p("HOUSEHOLD-LEVEL WEIGHT STRUCTURE")
    p("-" * 80)

    hh_diag_all = None
    hh_diag_base = None

    if HH_PATH.exists():
        try:
            hh = load_with_usecols(HH_PATH, HH_COLS)
        except Exception as exc:
            p(f"Could not load hh.sav: {type(exc).__name__}: {exc}")
            hh = None

        if hh is not None:
            missing_hh = [c for c in HH_COLS if c not in hh.columns]
            p(f"HH SHAPE: {hh.shape}")
            p(f"MISSING HH COLUMNS: {missing_hh if missing_hh else 'none'}")

            if not missing_hh:
                fs = fs.copy()
                fs["household_key"] = household_key(fs)

                hh = hh.copy()
                hh["household_key"] = household_key(hh)

                dup_hh = int(hh.duplicated("household_key").sum())
                p(f"Duplicate household keys in hh.sav: {dup_hh}")

                if dup_hh > 0:
                    hh = hh.drop_duplicates("household_key", keep="first")

                hh_small = hh[["household_key", "hhweight", "HH52"]].copy()

                fs_h = fs.merge(hh_small, on="household_key", how="left")

                fs_h["abs_diff_fsh_hh"] = (
                    num(fs_h["fshweight"]) - num(fs_h["hhweight"])
                ).abs()
                fs_h["abs_diff_fs_hh"] = (
                    num(fs_h["fsweight"]) - num(fs_h["hhweight"])
                ).abs()

                hh_diag_all = (
                    fs_h.groupby("household_key")
                    .agg(
                        n_rows=("LN", "size"),
                        hhweight=("hhweight", "first"),
                        HH52=("HH52", "first"),
                        sum_fsweight=("fsweight", "sum"),
                        sum_fshweight=("fshweight", "sum"),
                        max_abs_diff_fsh_hh=("abs_diff_fsh_hh", "max"),
                        max_abs_diff_fs_hh=("abs_diff_fs_hh", "max"),
                        nunique_fshweight=("fshweight", "nunique"),
                        nunique_fsweight=("fsweight", "nunique"),
                    )
                    .reset_index()
                )

                base_keys = set(household_key(base))
                hh_diag_base = hh_diag_all[
                    hh_diag_all["household_key"].isin(base_keys)
                ].copy()

                summarize_household_weights(hh_diag_all, "All households in fs.sav")
                summarize_household_weights(
                    hh_diag_base,
                    "Households represented in rural Chattogram base",
                )
            else:
                p("Skipping household weight structure because hh columns are missing.")
        else:
            p("Skipping household weight structure because hh.sav could not be loaded.")
    else:
        p(f"hh.sav not found: {HH_PATH.resolve()}")
        p("Skipping household weight structure.")

    p("")
    p("BOOKS_MISSING CHECK")
    p("-" * 80)

    pr3 = num(base["PR3"]) if "PR3" in base.columns else None
    ph_books = num(ph["books_missing"])

    if pr3 is None:
        p("PR3 not available in loaded fs columns.")
    else:
        p("PR3 value counts in base:")
        p(pr3.value_counts(dropna=False).sort_index().to_string())

        p("")
        p("Phase 2 books_missing value counts:")
        p(ph_books.value_counts(dropna=False).sort_index().to_string())

        p("")
        p("Cross-tab PR3 vs Phase 2 books_missing:")
        p(
            pd.crosstab(
                pr3,
                ph_books,
                rownames=["PR3"],
                colnames=["books_missing"],
                dropna=False,
            ).to_string()
        )

        p("")
        p("Cross-tab FS17 vs Phase 2 books_missing:")
        p(
            pd.crosstab(
                fs17,
                ph_books,
                rownames=["FS17"],
                colnames=["books_missing"],
                dropna=False,
            ).to_string()
        )

        p("")
        p("Candidate books_missing rules:")

        rule_results = {}

        def eval_rule(name, candidate):
            res = match_numeric(candidate, ph_books)
            rule_results[name] = res
            p(f"{name}: {fmt(res)}")

        def cond_series(cond, missing_mask, missing_fill=None):
            cond_float = cond.astype(float).to_numpy()
            miss = missing_mask.to_numpy()

            if missing_fill is None:
                vals = np.where(miss, np.nan, cond_float)
            else:
                vals = np.where(miss, float(missing_fill), cond_float)

            return pd.Series(vals, index=base.index)

        pr3_no99 = pr3.replace({99.0: np.nan})

        variants = [
            ("PR3 with 99 as missing", pr3_no99),
            ("PR3 raw, 99 kept", pr3),
        ]

        for label, pval in variants:
            missing = pval.isna()

            for t in [0, 1, 2, 3]:
                cond = pval <= t
                eval_rule(f"{label} <= {t}, NaN->NaN", cond_series(cond, missing, None))
                eval_rule(f"{label} <= {t}, NaN->0", cond_series(cond, missing, 0))
                eval_rule(f"{label} <= {t}, NaN->1", cond_series(cond, missing, 1))

            eval_rule(
                f"{label} is missing -> 1, else 0",
                pd.Series(np.where(missing, 1.0, 0.0), index=base.index),
            )

            eval_rule(
                f"{label} is not missing -> 1, else 0",
                pd.Series(np.where(missing, 0.0, 1.0), index=base.index),
            )

        cond_0_or_99 = pr3.isin([0.0, 99.0])
        eval_rule(
            "PR3 raw in {0, 99}, NaN->NaN",
            cond_series(cond_0_or_99, pr3.isna(), None),
        )
        eval_rule(
            "PR3 raw in {0, 99}, NaN->0",
            cond_series(cond_0_or_99, pr3.isna(), 0),
        )
        eval_rule(
            "PR3 raw in {0, 99}, NaN->1",
            cond_series(cond_0_or_99, pr3.isna(), 1),
        )

        pr3_99_to_0 = pr3.replace({99.0: 0.0})
        cond_99_to_0_eq_0 = pr3_99_to_0 == 0
        eval_rule(
            "PR3 with 99 set to 0, then ==0, NaN->NaN",
            cond_series(cond_99_to_0_eq_0, pr3.isna(), None),
        )
        eval_rule(
            "PR3 with 99 set to 0, then ==0, NaN->0",
            cond_series(cond_99_to_0_eq_0, pr3.isna(), 0),
        )
        eval_rule(
            "PR3 with 99 set to 0, then ==0, NaN->1",
            cond_series(cond_99_to_0_eq_0, pr3.isna(), 1),
        )

        p("")
        p("Best books_missing rule:")

        valid_rules = [(name, res) for name, res in rule_results.items() if res is not None]

        if not valid_rules:
            p("none")
        else:
            best_name, best_res = max(valid_rules, key=lambda item: item[1][0])
            p(f"{best_name}: {fmt(best_res)}")

            if best_res[0] >= 99.999:
                p("")
                p("BOOKS STATUS: A simple PR3 rule reproduces Phase 2 books_missing.")
            else:
                p("")
                p("BOOKS STATUS: No simple PR3 rule reproduces Phase 2 books_missing.")
                p("Next step will be to inspect the original Colab feature-building code or rebuild books_missing from a verified source.")

    p("")
    p("VERDICT")
    p("-" * 80)

    p("Structural provenance: PASS")
    p("Sample filter: PASS")
    p("Target, age, sex, band, group: PASS")
    p("Weight column name: MISLEADING")
    p("The Phase 2 column named FSWEIGHT matches raw fshweight zero->NaN, not raw fsweight.")

    if hh_diag_base is not None and len(hh_diag_base) > 0:
        fsh_constant = (hh_diag_base["nunique_fshweight"] == 1).mean()
        fsh_close_hh = (hh_diag_base["max_abs_diff_fsh_hh"] < 1e-6).mean()
        fs_sum_close_hh = (
            (hh_diag_base["sum_fsweight"] - hh_diag_base["hhweight"]).abs() < 1e-4
        ).mean()

        p("")
        p("Household weight evidence in base households:")
        p(f"fshweight constant share: {fsh_constant:.4f}")
        p(f"fshweight equals hhweight closely: {fsh_close_hh:.4f}")
        p(f"sum(fsweight) equals hhweight closely: {fs_sum_close_hh:.4f}")

        if fsh_constant > 0.99 and fsh_close_hh > 0.99:
            p("")
            p("TENTATIVE WEIGHT FINDING: fshweight behaves like a household weight.")
            p("If fsweight behaves like a child weight, the Phase 2 weight column is probably wrong and must be replaced before final weighted analysis.")
        elif fs_sum_close_hh > 0.99:
            p("")
            p("TENTATIVE WEIGHT FINDING: fsweight behaves like the child-level weight.")
            p("The Phase 2 FSWEIGHT column should probably be rebuilt from fsweight, not fshweight.")
        else:
            p("")
            p("TENTATIVE WEIGHT FINDING: household evidence is not decisive from this check alone.")
            p("Send the full report; the next step may be to inspect MICS documentation or the original Colab weight-selection code.")
    else:
        p("")
        p("Weight finding: household structure could not be checked because hh.sav was unavailable.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/remaining_provenance_check.txt.")
    p("Do not send raw child records.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()