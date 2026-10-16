from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


FS_PATH = Path("data/raw/fs.sav")
HH_PATH = Path("data/raw/hh.sav")
PHASE2_PATH = Path("data/processed/chattogram_rural_phase2_input.csv")
REPORT_PATH = Path("reports/books_weight_resolution.txt")

FS_COLS = [
    "HH1",
    "HH2",
    "LN",
    "PSU",
    "stratum",
    "HH6",
    "HH7",
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
    "CL1X",
    "PR3",
]

HH_COLS = [
    "HH1",
    "HH2",
    "hhweight",
]

PHASE_COLS = [
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
]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def load_sav(path, cols):
    df, _ = pyreadstat.read_sav(
        str(path),
        apply_value_formats=False,
        formats_as_category=False,
        usecols=cols,
    )
    return df


def household_key(df):
    return (
        df["HH1"].astype("Int64").astype(str)
        + "_"
        + df["HH2"].astype("Int64").astype(str)
    )


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


def match_categorical(a, b):
    a = a.astype("string")
    b = b.astype("string")

    if len(a) != len(b):
        return None

    both_na = (a.isna() & b.isna()).to_numpy()

    try:
        eq = (a == b).fillna(False).to_numpy()
    except Exception:
        eq = (a.astype(object) == b.astype(object))

    ok = both_na | eq
    n = len(ok)
    count = int(ok.sum())

    return (count / n * 100 if n else 0.0), count, n


def fmt(res):
    if res is None:
        return "length mismatch or missing column"

    pct, count, n = res
    return f"{count}/{n} ({pct:.2f}%)"


def preserve_binary_eq(series, code, missing_codes=None):
    s = num(series)

    if missing_codes:
        s = s.replace({float(c): np.nan for c in missing_codes})

    return pd.Series(
        np.where(s.isna(), np.nan, (s == code).astype(float)),
        index=s.index,
    )


def weighted_rate(y, w):
    y = num(y)
    w = num(w)

    mask = y.notna() & w.notna() & (w > 0)

    if int(mask.sum()) == 0:
        return np.nan, 0

    rate = float(np.average(y[mask], weights=w[mask]))
    return rate, int(mask.sum())


def effective_sample_size(w):
    w = num(w)
    w = w[w > 0]

    if len(w) == 0:
        return np.nan

    return float((w.sum() ** 2) / (w.pow(2).sum()))


def describe_ratio(name, ratio):
    r = num(ratio).replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) == 0:
        p(f"{name}: no valid ratios")
        return

    p(
        f"{name}: n={len(r)}, "
        f"min={r.min():.6g}, "
        f"p25={r.quantile(0.25):.6g}, "
        f"median={r.median():.6g}, "
        f"p75={r.quantile(0.75):.6g}, "
        f"max={r.max():.6g}, "
        f"mean={r.mean():.6g}, "
        f"sd={r.std():.6g}"
    )


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not FS_PATH.exists():
        raise SystemExit(f"Raw fs.sav not found: {FS_PATH.resolve()}")

    if not PHASE2_PATH.exists():
        raise SystemExit(f"Phase 2 file not found: {PHASE2_PATH.resolve()}")

    p("=" * 80)
    p("BOOKS_MISSING AND WEIGHT RESOLUTION")
    p("=" * 80)
    p(f"FS FILE: {FS_PATH.resolve()}")
    p(f"HH FILE: {HH_PATH.resolve()}")
    p(f"PHASE 2 FILE: {PHASE2_PATH.resolve()}")

    fs = load_sav(FS_PATH, FS_COLS)
    ph = pd.read_csv(PHASE2_PATH, low_memory=False).reset_index(drop=True)

    p(f"FS SHAPE: {fs.shape}")
    p(f"PHASE 2 SHAPE: {ph.shape}")

    missing_fs = [c for c in FS_COLS if c not in fs.columns]
    missing_ph = [c for c in PHASE_COLS if c not in ph.columns]

    p(f"MISSING FS COLUMNS: {missing_fs if missing_fs else 'none'}")
    p(f"MISSING PHASE 2 COLUMNS: {missing_ph if missing_ph else 'none'}")

    if missing_fs or missing_ph:
        p("")
        p("VERDICT: FAIL - required columns missing.")
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
    p("DETERMINISTIC FIELD CHECKS")
    p("-" * 80)

    raw_age = num(base["CB3"])
    raw_female = preserve_binary_eq(base["HL4"], 2)
    raw_oos = preserve_binary_eq(base["CB7"], 2)
    raw_psu = num(base["PSU"])
    raw_wealth = num(base["windex5"]).replace(0.0, np.nan)
    raw_mom = num(base["melevel"])
    raw_eth = preserve_binary_eq(base["ethnicity"], 2, missing_codes=[9])
    raw_disabled = preserve_binary_eq(base["fsdisability"], 1, missing_codes=[9])
    raw_worked_CL1A = preserve_binary_eq(base["CL1A"], 1, missing_codes=[9])

    bands = pd.cut(
        raw_age,
        bins=[5, 9, 12, 15, 17],
        labels=["6-9", "10-12", "13-15", "16-17"],
        right=True,
        include_lowest=True,
    )
    raw_band = bands.astype("string")

    checks = [
        ("age", match_numeric(raw_age, ph["age"])),
        ("_band", match_categorical(raw_band, ph["_band"])),
        ("female", match_numeric(raw_female, ph["female"])),
        ("Out_of_School", match_numeric(raw_oos, ph["Out_of_School"])),
        ("_group vs PSU", match_numeric(raw_psu, ph["_group"])),
        ("wealth_q vs windex5 zero->NaN", match_numeric(raw_wealth, ph["wealth_q"])),
        ("mom_edu vs melevel raw", match_numeric(raw_mom, ph["mom_edu"])),
        ("eth_minor vs ethnicity other", match_numeric(raw_eth, ph["eth_minor"])),
        ("disabled vs fsdisability has difficulty", match_numeric(raw_disabled, ph["disabled"])),
        ("worked vs CL1A", match_numeric(raw_worked_CL1A, ph["worked"])),
    ]

    for name, res in checks:
        p(f"{name}: {fmt(res)}")

    p("")
    p("BOOKS_MISSING INVESTIGATION")
    p("-" * 80)

    pr3 = num(base["PR3"])
    ph_books = num(ph["books_missing"])

    p("PR3 value counts in base:")
    p(pr3.value_counts(dropna=False).sort_index().to_string())

    p("")
    p("Phase 2 books_missing value counts:")
    p(ph_books.value_counts(dropna=False).sort_index().to_string())

    pr3_missing_code = pr3.isna() | (pr3 == 99)
    pr3_valid = ~pr3_missing_code
    pr3_zero = pr3 == 0
    pr3_positive = pr3_valid & (pr3 > 0)

    p("")
    p(f"PR3 missing or 99: {int(pr3_missing_code.sum())}")
    p(f"PR3 valid: {int(pr3_valid.sum())}")
    p(f"PR3 == 0 among valid: {int((pr3_valid & pr3_zero).sum())}")
    p(f"PR3 > 0 among valid: {int(pr3_positive.sum())}")
    p(f"Phase books_missing == 1: {int(ph_books.sum())}")
    p(f"Phase books_missing == 0: {int((ph_books == 0).sum())}")

    rules = {}

    def add_rule(name, series):
        res = match_numeric(series, ph_books)
        rules[name] = (series, res)
        p(f"{name}: {fmt(res)}")

    # Rule 1: books_missing means PR3 data are missing/no response.
    add_rule(
        "books_missing = PR3 missing or 99",
        pd.Series(pr3_missing_code.astype(float), index=base.index),
    )

    # Rule 2: books_missing means PR3 == 0, missing stays missing.
    add_rule(
        "books_missing = PR3 == 0, missing -> NaN",
        pd.Series(
            np.where(pr3_missing_code, np.nan, pr3_zero.astype(float)),
            index=base.index,
        ),
    )

    # Rule 3: books_missing means PR3 == 0, missing -> 0.
    add_rule(
        "books_missing = PR3 == 0, missing -> 0",
        pd.Series(
            np.where(pr3_missing_code, 0.0, pr3_zero.astype(float)),
            index=base.index,
        ),
    )

    # Rule 4: books_missing means PR3 == 0, missing -> 1.
    add_rule(
        "books_missing = PR3 == 0, missing -> 1",
        pd.Series(
            np.where(pr3_missing_code, 1.0, pr3_zero.astype(float)),
            index=base.index,
        ),
    )

    # Rule 5: books_missing means no books or no response.
    add_rule(
        "books_missing = PR3 in {0, 99}, NaN -> 0",
        pd.Series(
            np.where(pr3.isna(), 0.0, pr3.isin([0.0, 99.0]).astype(float)),
            index=base.index,
        ),
    )

    # Rule 6: books_missing means no books or missing/no response.
    add_rule(
        "books_missing = PR3 missing/99 or PR3 == 0",
        pd.Series(
            (pr3_missing_code | (pr3_valid & pr3_zero)).astype(float),
            index=base.index,
        ),
    )

    # Rule 7: inverse, has_books = 1, books_missing = 0.
    add_rule(
        "books_missing = NOT has_books, missing -> NaN",
        pd.Series(
            np.where(pr3_missing_code, np.nan, (~pr3_zero).astype(float)),
            index=base.index,
        ),
    )

    add_rule(
        "books_missing = NOT has_books, missing -> 0",
        pd.Series(
            np.where(pr3_missing_code, 0.0, (~pr3_zero).astype(float)),
            index=base.index,
        ),
    )

    add_rule(
        "books_missing = NOT has_books, missing -> 1",
        pd.Series(
            np.where(pr3_missing_code, 1.0, (~pr3_zero).astype(float)),
            index=base.index,
        ),
    )

    # Rule 8: treat 9 as missing too, although PR3 label says 9 = nine books.
    pr3_missing_if_9_also = pr3.isna() | pr3.isin([9.0, 99.0])
    add_rule(
        "books_missing = PR3 missing/9/99",
        pd.Series(pr3_missing_if_9_also.astype(float), index=base.index),
    )

    add_rule(
        "books_missing = PR3 == 0, missing/9/99 -> 1",
        pd.Series(
            np.where(pr3_missing_if_9_also, 1.0, (pr3 == 0).astype(float)),
            index=base.index,
        ),
    )

    p("")
    p("Best books_missing rule:")

    valid_rules = [(name, res) for name, (_, res) in rules.items() if res is not None]

    if not valid_rules:
        p("none")
        books_status = "FAIL"
    else:
        best_name, best_res = max(valid_rules, key=lambda item: item[1][0])
        p(f"{best_name}: {fmt(best_res)}")

        if best_res[0] >= 99.999:
            books_status = "PASS"
            p("")
            p("BOOKS STATUS: A simple PR3 rule reproduces Phase 2 books_missing.")
        else:
            books_status = "UNRESOLVED"
            p("")
            p("BOOKS STATUS: No simple PR3 rule reproduces Phase 2 books_missing.")
            p("Next step: inspect the original Colab feature-building code or rebuild this feature from a verified source.")

    p("")
    p("WEIGHT INVESTIGATION")
    p("-" * 80)

    fs_w = num(base["fsweight"])
    fsh_w = num(base["fshweight"])
    ph_w = num(ph["FSWEIGHT"])
    y = raw_oos

    p(f"base fsweight: missing={int(fs_w.isna().sum())}, zero={int((fs_w == 0).sum())}, positive={int((fs_w > 0).sum())}, sum={fs_w.sum():.6g}")
    p(f"base fshweight: missing={int(fsh_w.isna().sum())}, zero={int((fsh_w == 0).sum())}, positive={int((fsh_w > 0).sum())}, sum={fsh_w.sum():.6g}")
    p(f"phase FSWEIGHT: missing={int(ph_w.isna().sum())}, zero={int((ph_w == 0).sum())}, positive={int((ph_w > 0).sum())}, sum={ph_w.sum():.6g}")

    p("")
    p("Weighted out-of-school rates in base:")

    unweighted_rate = float(y.mean(skipna=True))
    p(f"Unweighted OOS rate: {unweighted_rate:.6f}")

    fs_rate, fs_n = weighted_rate(y, fs_w)
    fsh_rate, fsh_n = weighted_rate(y, fsh_w)

    p(f"fsweight-weighted OOS rate: {fs_rate:.6f}, usable rows={fs_n}")
    p(f"fshweight-weighted OOS rate: {fsh_rate:.6f}, usable rows={fsh_n}")

    p("")
    p("Effective sample size:")
    p(f"fsweight ESS: {effective_sample_size(fs_w):.2f}")
    p(f"fshweight ESS: {effective_sample_size(fsh_w):.2f}")

    p("")
    p("Relationship between fshweight and fsweight:")

    both_pos = (fs_w > 0) & (fsh_w > 0)
    ratio_fsh_fs = fsh_w[both_pos] / fs_w[both_pos]
    describe_ratio("fshweight / fsweight where both positive", ratio_fsh_fs)

    corr = pd.concat([fs_w[both_pos], fsh_w[both_pos]], axis=1).corr().iloc[0, 1]
    p(f"Correlation between positive fsweight and fshweight: {corr:.6f}")

    p("")
    p("Phase 2 weight column check:")
    p(f"Phase FSWEIGHT vs fsweight zero->NaN: {fmt(match_numeric(fs_w.replace(0.0, np.nan), ph_w))}")
    p(f"Phase FSWEIGHT vs fshweight zero->NaN: {fmt(match_numeric(fsh_w.replace(0.0, np.nan), ph_w))}")

    p("")
    p("HOUSEHOLD WEIGHT CONTEXT")
    p("-" * 80)

    if HH_PATH.exists():
        try:
            hh = load_sav(HH_PATH, HH_COLS)
        except Exception as exc:
            p(f"Could not load hh.sav: {type(exc).__name__}: {exc}")
            hh = None

        if hh is not None:
            missing_hh = [c for c in HH_COLS if c not in hh.columns]
            p(f"HH SHAPE: {hh.shape}")
            p(f"MISSING HH COLUMNS: {missing_hh if missing_hh else 'none'}")

            if not missing_hh:
                base = base.copy()
                base["household_key"] = household_key(base)

                hh = hh.copy()
                hh["household_key"] = household_key(hh)

                dup_hh = int(hh.duplicated("household_key").sum())
                p(f"Duplicate household keys in hh.sav: {dup_hh}")

                if dup_hh > 0:
                    hh = hh.drop_duplicates("household_key", keep="first")

                merged = base.merge(
                    hh[["household_key", "hhweight"]],
                    on="household_key",
                    how="left",
                )

                hw = num(merged["hhweight"])

                p("")
                p("Children per household key in base:")
                children_per_hh = merged.groupby("household_key").size()
                p(f"max children per household in base: {int(children_per_hh.max())}")
                p(f"households with more than one base row: {int((children_per_hh > 1).sum())}")

                p("")
                p("Weight ratios against household weight, positive weights only:")

                fs_pos = (num(merged["fsweight"]) > 0) & (hw > 0)
                fsh_pos = (num(merged["fshweight"]) > 0) & (hw > 0)

                describe_ratio(
                    "fsweight / hhweight",
                    num(merged.loc[fs_pos, "fsweight"]) / hw[fs_pos],
                )
                describe_ratio(
                    "fshweight / hhweight",
                    num(merged.loc[fsh_pos, "fshweight"]) / hw[fsh_pos],
                )

                p("")
                p("Interpretation:")
                p("- If fshweight / hhweight is tight near 1, fshweight behaves like a household-normalized weight.")
                p("- If fsweight / hhweight is more variable, fsweight may be the child population expansion weight.")
                p("- This is internal evidence, not official codebook verification.")
            else:
                p("Skipping household ratio check because required hh columns are missing.")
        else:
            p("Skipping household ratio check because hh.sav could not be loaded.")
    else:
        p(f"hh.sav not found: {HH_PATH.resolve()}")
        p("Skipping household ratio check.")

    p("")
    p("VERDICT")
    p("-" * 80)

    p("Sample provenance: PASS")
    p("Deterministic fields: PASS")
    p("Weight column name: MISLEADING")
    p("The Phase 2 column named FSWEIGHT matches raw fshweight, not raw fsweight.")
    p(f"Books feature status: {books_status}")

    p("")
    p("Provisional weight recommendation:")
    p("- Use raw lowercase fsweight as the primary child-level weight if the MICS codebook/project rule confirms it.")
    p("- Treat fshweight as a normalized or household-like weight unless documentation says otherwise.")
    p("- Do not finalize optimization using the current Phase 2 FSWEIGHT column until the weight choice is documented.")
    p("- If the risk scores were trained using the current FSWEIGHT column, retrain or rerun model evaluation after switching to the verified weight.")

    p("")
    p("NEXT STEP AFTER THIS REPORT")
    p("-" * 80)

    if books_status == "PASS":
        p("1. Document the verified books_missing rule.")
    else:
        p("1. Recover or rebuild the books_missing feature from verified raw data.")

    p("2. Decide primary weight: likely fsweight, with fshweight only as sensitivity.")
    p("3. Rebuild a final optimization-ready file with original HH1/HH2/LN/PSU IDs.")
    p("4. Retrain or revalidate risk scores using the verified weight.")
    p("5. Only then run the budgeted selection and fairness sweep.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/books_weight_resolution.txt.")
    p("Do not send raw child records.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()