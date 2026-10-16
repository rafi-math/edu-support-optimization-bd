from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


RAW = Path("data/raw/fs.sav")
PHASE2 = Path("data/processed/chattogram_rural_phase2_input.csv")
REPORT = Path("reports/raw_provenance_check_v2.txt")

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def get_value_labels(meta, col):
    for attr in [
        "variable_value_labels",
        "column_names_to_value_labels",
        "value_labels",
    ]:
        obj = getattr(meta, attr, None)
        if isinstance(obj, dict) and col in obj:
            val = obj[col]
            if isinstance(val, dict):
                return val
    return {}


def code_by_label(labels, terms, exact=False):
    terms = [str(t).lower() for t in terms]

    for val, lab in labels.items():
        s = str(lab).strip().lower()

        if exact:
            if s in terms:
                return val
        else:
            if any(t in s for t in terms):
                return val

    return None


def num(series):
    return pd.to_numeric(series, errors="coerce")


def recode_codes_to_nan(series, codes):
    s = num(series)

    if codes:
        s = s.replace(list(codes), np.nan)

    return s


def as_binary_eq(series, code, missing_codes=None):
    s = recode_codes_to_nan(series, missing_codes or [])

    return pd.Series(
        np.where(s.isna(), np.nan, (s == code).astype(float)),
        index=s.index,
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

    ok = (a.isna() & b.isna()) | (a == b)
    n = len(ok)
    count = int(ok.sum())

    return (count / n * 100 if n else 0.0), count, n


def fmt_match(res):
    if res is None:
        return "length mismatch"

    pct, count, n = res
    return f"{count}/{n} ({pct:.2f}%)"


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise SystemExit(f"Raw file not found: {RAW.resolve()}")

    if not PHASE2.exists():
        raise SystemExit(f"Phase 2 file not found: {PHASE2.resolve()}")

    p("=" * 80)
    p("RAW PROVENANCE CHECK V2")
    p("=" * 80)
    p(f"RAW FILE: {RAW.resolve()}")
    p(f"PHASE 2 FILE: {PHASE2.resolve()}")

    df, meta = pyreadstat.read_sav(
        RAW,
        convert_categoricals=False,
        apply_value_formats=False,
    )

    ph = pd.read_csv(PHASE2, low_memory=False).reset_index(drop=True)

    p(f"RAW SHAPE: {df.shape}")
    p(f"PHASE 2 SHAPE: {ph.shape}")

    required_raw = [
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
        "PR3",
        "CL1A",
        "CL1X",
    ]

    required_phase = [
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

    missing_raw = [c for c in required_raw if c not in df.columns]
    missing_phase = [c for c in required_phase if c not in ph.columns]

    p(f"MISSING RAW COLUMNS: {missing_raw if missing_raw else 'none'}")
    p(f"MISSING PHASE 2 COLUMNS: {missing_phase if missing_phase else 'none'}")

    if missing_raw or missing_phase:
        p("")
        p("VERDICT: FAIL - cannot compare because required columns are missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    label_cols = [
        "HH6",
        "HH7",
        "CB7",
        "HL4",
        "ethnicity",
        "fsdisability",
        "CL1A",
        "CL1X",
        "PR3",
        "melevel",
        "FS17",
    ]

    vl = {c: get_value_labels(meta, c) for c in label_cols}

    rural_code = code_by_label(vl["HH6"], ["rural"]) or 2
    division_code = code_by_label(vl["HH7"], ["chattogram", "chittagong"]) or 20
    cb7_yes_code = code_by_label(vl["CB7"], ["yes"], exact=True) or 1
    cb7_no_code = code_by_label(vl["CB7"], ["no"], exact=True) or 2
    female_code = code_by_label(vl["HL4"], ["female"]) or 2
    eth_other_code = code_by_label(vl["ethnicity"], ["other"]) or 2
    dis_has_code = code_by_label(vl["fsdisability"], ["has functional difficulty"]) or 1
    work_yes_code = code_by_label(vl["CL1A"], ["yes"]) or 1
    pr3_none_code = code_by_label(vl["PR3"], ["none"]) or 0
    pr3_nr_code = code_by_label(vl["PR3"], ["no response"]) or 99
    melevel_dk_code = code_by_label(vl["melevel"], ["missing", "dk"]) or 9

    p("")
    p("DETECTED CODES")
    p("-" * 80)
    p(f"HH6 rural: {rural_code}")
    p(f"HH7 Chattogram: {division_code}")
    p(f"CB7 yes: {cb7_yes_code}")
    p(f"CB7 no: {cb7_no_code}")
    p(f"HL4 female: {female_code}")
    p(f"ethnicity other: {eth_other_code}")
    p(f"fsdisability has difficulty: {dis_has_code}")
    p(f"CL1A yes: {work_yes_code}")
    p(f"PR3 none: {pr3_none_code}")
    p(f"PR3 no response: {pr3_nr_code}")
    p(f"melevel missing/DK: {melevel_dk_code}")

    hh6 = num(df["HH6"])
    hh7 = num(df["HH7"])
    cb3 = num(df["CB3"])
    cb7 = num(df["CB7"])

    mask_rural = hh6 == rural_code
    mask_div = hh7 == division_code
    mask_age = cb3.between(6, 17)
    mask_cb7 = cb7.isin([cb7_yes_code, cb7_no_code])
    base_mask = mask_rural & mask_div & mask_age & mask_cb7

    p("")
    p("FILTER COUNTS")
    p("-" * 80)
    p(f"Total raw rows: {len(df)}")
    p(f"Rural rows: {int(mask_rural.sum())}")
    p(f"Rural Chattogram rows: {int((mask_rural & mask_div).sum())}")
    p(f"Rural Chattogram age 6-17 rows: {int((mask_rural & mask_div & mask_age).sum())}")
    p(f"Rural Chattogram age 6-17 valid CB7 rows: {int(base_mask.sum())}")

    base = df.loc[base_mask].copy().reset_index(drop=True)
    raw = pd.DataFrame(index=base.index)

    for col in required_raw:
        raw[col] = num(base[col])

    raw["age"] = raw["CB3"]
    raw["female"] = as_binary_eq(raw["HL4"], female_code)
    raw["Out_of_School"] = as_binary_eq(raw["CB7"], cb7_no_code)
    raw["_group_PSU"] = raw["PSU"]
    raw["_group_HH1"] = raw["HH1"]

    raw["fsweight_zero_to_nan"] = raw["fsweight"].replace(0, np.nan)
    raw["fshweight_zero_to_nan"] = raw["fshweight"].replace(0, np.nan)

    raw["wealth_q"] = recode_codes_to_nan(raw["windex5"], [9, 99])

    raw["mom_edu_raw"] = raw["melevel"]
    raw["mom_edu_9_to_nan"] = recode_codes_to_nan(raw["melevel"], [melevel_dk_code])
    raw["mom_edu_9_to_0"] = raw["melevel"].replace({melevel_dk_code: 0.0})

    eth_clean = recode_codes_to_nan(raw["ethnicity"], [9])
    raw["eth_minor_other"] = pd.Series(
        np.where(
            eth_clean.isna(),
            np.nan,
            (eth_clean == eth_other_code).astype(float),
        ),
        index=raw.index,
    )

    eth_9_to_0 = raw["ethnicity"].replace({9: 0.0})
    raw["eth_minor_other_9_to_0"] = pd.Series(
        (eth_9_to_0 == eth_other_code).astype(float),
        index=raw.index,
    )

    raw["disabled_has"] = as_binary_eq(
        raw["fsdisability"],
        dis_has_code,
        missing_codes=[9],
    )

    cl1a = as_binary_eq(raw["CL1A"], work_yes_code, missing_codes=[9])
    cl1x = as_binary_eq(raw["CL1X"], work_yes_code, missing_codes=[9])

    raw["worked_CL1A"] = cl1a
    raw["worked_CL1X"] = cl1x

    raw["worked_any"] = pd.Series(
        np.where(
            (cl1a == 1) | (cl1x == 1),
            1.0,
            np.where(cl1a.isna() | cl1x.isna(), np.nan, 0.0),
        ),
        index=raw.index,
    )

    raw["books_missing_PR3_none"] = as_binary_eq(
        raw["PR3"],
        pr3_none_code,
        missing_codes=[pr3_nr_code],
    )

    pr3_99_to_0 = raw["PR3"].replace({pr3_nr_code: 0.0})
    raw["books_missing_PR3_99_to_0"] = pd.Series(
        np.where(
            pr3_99_to_0.isna(),
            np.nan,
            (pr3_99_to_0 == pr3_none_code).astype(float),
        ),
        index=raw.index,
    )

    bands = pd.cut(
        raw["age"],
        bins=[5, 9, 12, 15, 17],
        labels=["6-9", "10-12", "13-15", "16-17"],
        right=True,
        include_lowest=True,
    )

    raw["_band"] = bands.astype("string")

    p("")
    p("RAW BASE DIAGNOSTICS")
    p("-" * 80)
    p(f"Base rows: {len(base)}")
    p(f"Phase rows: {len(ph)}")
    p(f"Length match: {len(base) == len(ph)}")

    p("")
    p("FS17 counts in base:")
    p(raw["FS17"].value_counts(dropna=False).sort_index().to_string())

    p("")
    p("Weight diagnostics in base:")

    for wname in ["fsweight", "fshweight"]:
        s = raw[wname]
        p(
            f"{wname}: missing={int(s.isna().sum())}, "
            f"zero={int((s == 0).sum())}, "
            f"positive={int((s > 0).sum())}, "
            f"min={s.min()}, max={s.max()}, sum={s.sum()}"
        )

    ph_w = num(ph["FSWEIGHT"])

    p(
        f"phase FSWEIGHT: missing={int(ph_w.isna().sum())}, "
        f"zero={int((ph_w == 0).sum())}, "
        f"positive={int((ph_w > 0).sum())}, "
        f"min={ph_w.min()}, max={ph_w.max()}, sum={ph_w.sum()}"
    )

    p("")
    p("Crosstab raw fsweight zero vs FS17:")
    p(pd.crosstab(raw["fsweight"].eq(0), raw["FS17"], dropna=False).to_string())

    p("")
    p("Crosstab raw fshweight zero vs FS17:")
    p(pd.crosstab(raw["fshweight"].eq(0), raw["FS17"], dropna=False).to_string())

    p("")
    p("Candidate missing-code diagnostics:")

    for col in [
        "windex5",
        "melevel",
        "ethnicity",
        "fsdisability",
        "CL1A",
        "CL1X",
        "PR3",
    ]:
        s = raw[col]
        p(
            f"{col}: missing={int(s.isna().sum())}, "
            f"9={int((s == 9).sum())}, "
            f"99={int((s == 99).sum())}"
        )

    p("")
    p("POSITIONAL MATCH CHECKS")
    p("-" * 80)

    pairs = [
        ("age", raw["age"], ph["age"], "numeric"),
        ("_band", raw["_band"], ph["_band"], "categorical"),
        ("female", raw["female"], ph["female"], "numeric"),
        ("Out_of_School", raw["Out_of_School"], ph["Out_of_School"], "numeric"),
        ("_group vs PSU", raw["_group_PSU"], ph["_group"], "numeric"),
        ("_group vs HH1", raw["_group_HH1"], ph["_group"], "numeric"),
        ("FSWEIGHT vs fsweight raw", raw["fsweight"], ph["FSWEIGHT"], "numeric"),
        (
            "FSWEIGHT vs fsweight zero->NaN",
            raw["fsweight_zero_to_nan"],
            ph["FSWEIGHT"],
            "numeric",
        ),
        ("FSWEIGHT vs fshweight raw", raw["fshweight"], ph["FSWEIGHT"], "numeric"),
        (
            "FSWEIGHT vs fshweight zero->NaN",
            raw["fshweight_zero_to_nan"],
            ph["FSWEIGHT"],
            "numeric",
        ),
        ("wealth_q vs windex5", raw["wealth_q"], ph["wealth_q"], "numeric"),
        ("mom_edu vs melevel raw", raw["mom_edu_raw"], ph["mom_edu"], "numeric"),
        ("mom_edu vs melevel 9->NaN", raw["mom_edu_9_to_nan"], ph["mom_edu"], "numeric"),
        ("mom_edu vs melevel 9->0", raw["mom_edu_9_to_0"], ph["mom_edu"], "numeric"),
        ("eth_minor vs ethnicity other", raw["eth_minor_other"], ph["eth_minor"], "numeric"),
        (
            "eth_minor vs ethnicity other 9->0",
            raw["eth_minor_other_9_to_0"],
            ph["eth_minor"],
            "numeric",
        ),
        (
            "disabled vs fsdisability has difficulty",
            raw["disabled_has"],
            ph["disabled"],
            "numeric",
        ),
        ("worked vs CL1A", raw["worked_CL1A"], ph["worked"], "numeric"),
        ("worked vs CL1X", raw["worked_CL1X"], ph["worked"], "numeric"),
        ("worked vs CL1A or CL1X", raw["worked_any"], ph["worked"], "numeric"),
        ("books_missing vs PR3 none", raw["books_missing_PR3_none"], ph["books_missing"], "numeric"),
        (
            "books_missing vs PR3 none, 99->0",
            raw["books_missing_PR3_99_to_0"],
            ph["books_missing"],
            "numeric",
        ),
    ]

    match_results = {}

    if len(base) != len(ph):
        p("Length mismatch. Positional checks skipped.")
        p("Likely next step: rebuild Phase 2 from raw with original IDs, or find the exact filter used before scoring.")
    else:
        for name, a, b, kind in pairs:
            if kind == "numeric":
                res = match_numeric(a, b)
            else:
                res = match_categorical(a, b)

            match_results[name] = res
            p(f"{name}: {fmt_match(res)}")

        p("")
        p("PSU / _group set check:")

        psu_set = set(raw["PSU"].dropna().unique().tolist())
        grp_set = set(num(ph["_group"]).dropna().unique().tolist())

        p(f"Raw base unique PSU count: {len(psu_set)}")
        p(f"Phase unique _group count: {len(grp_set)}")
        p(f"Exact set match: {psu_set == grp_set}")

        if psu_set != grp_set:
            raw_only = sorted(list(psu_set - grp_set))[:10]
            phase_only = sorted(list(grp_set - psu_set))[:10]

            p(f"Raw-only sample: {raw_only}")
            p(f"Phase-only sample: {phase_only}")

    p("")
    p("VERDICT")
    p("-" * 80)

    if len(base) != len(ph):
        p("PROVENANCE STATUS: FAIL")
        p("Reason: row count mismatch.")
    else:
        key_checks = [
            "age",
            "female",
            "Out_of_School",
            "_group vs PSU",
            "FSWEIGHT vs fsweight zero->NaN",
        ]

        key_pcts = []

        for k in key_checks:
            res = match_results.get(k)

            if res is None:
                key_pcts.append(None)
            else:
                key_pcts.append(res[0])

        if all(x is not None and x >= 99.999 for x in key_pcts):
            p("PROVENANCE STATUS: PASS for deterministic fields")
            p(
                "Interpretation: the Phase 2 file appears to be the raw rural "
                "Chattogram age 6-17 valid CB7 sample, in the same row order, "
                "with lowercase fsweight renamed to FSWEIGHT and zero weights "
                "converted to missing."
            )
        else:
            p("PROVENANCE STATUS: PARTIAL/FAIL")
            p(
                "At least one key deterministic field does not match positionally. "
                "Do not attach IDs or finalize optimization yet."
            )

        weight_candidates = {
            "fsweight zero->NaN": match_results.get("FSWEIGHT vs fsweight zero->NaN"),
            "fshweight zero->NaN": match_results.get("FSWEIGHT vs fshweight zero->NaN"),
            "fsweight raw": match_results.get("FSWEIGHT vs fsweight raw"),
            "fshweight raw": match_results.get("FSWEIGHT vs fshweight raw"),
        }

        p("")
        p("Best weight candidate:")

        best = None
        best_pct = -1.0

        for name, res in weight_candidates.items():
            if res is not None and res[0] > best_pct:
                best_pct = res[0]
                best = name

        if best is None:
            p("none")
        else:
            p(f"{best}: {best_pct:.2f}%")

        soft_checks = [
            "wealth_q vs windex5",
            "mom_edu vs melevel 9->NaN",
            "mom_edu vs melevel 9->0",
            "eth_minor vs ethnicity other",
            "eth_minor vs ethnicity other 9->0",
            "disabled vs fsdisability has difficulty",
            "worked vs CL1A",
            "worked vs CL1X",
            "worked vs CL1A or CL1X",
            "books_missing vs PR3 none",
            "books_missing vs PR3 none, 99->0",
        ]

        imperfect = []

        for name in soft_checks:
            res = match_results.get(name)

            if res is not None and res[0] < 99.999:
                imperfect.append((name, res[0]))

        if imperfect:
            p("")
            p("Transformation rules still need confirmation:")

            for name, pct in imperfect:
                p(f"- {name}: {pct:.2f}%")
        else:
            p("")
            p("All tested soft transformations matched positionally.")

    p("")
    p("NEXT ACTION AFTER YOU SEND THIS REPORT")
    p("-" * 80)
    p(
        "If PASS: attach original HH1/HH2/LN/PSU IDs to Phase 2, document the "
        "weight rule, drop or flag zero-weight rows, then run optimization dry run."
    )
    p(
        "If PARTIAL/FAIL: rebuild the Phase 2 input from raw fs.sav with original "
        "IDs and verified transformations before modeling."
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()