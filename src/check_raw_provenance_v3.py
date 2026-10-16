from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


RAW = Path("data/raw/fs.sav")
PHASE2 = Path("data/processed/chattogram_rural_phase2_input.csv")
REPORT = Path("reports/raw_provenance_check_v3.txt")

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def num(series):
    return pd.to_numeric(series, errors="coerce")


def preserve_binary(series, true_code, missing_codes=None):
    s = num(series)

    if missing_codes:
        s = s.replace({float(c): np.nan for c in missing_codes})

    return pd.Series(
        np.where(s.isna(), np.nan, (s == true_code).astype(float)),
        index=s.index,
    )


def preserve_gt0(series, missing_codes=None):
    s = num(series)

    if missing_codes:
        s = s.replace({float(c): np.nan for c in missing_codes})

    return pd.Series(
        np.where(s.isna(), np.nan, (s > 0).astype(float)),
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


def build_raw_features(base):
    f = pd.DataFrame(index=base.index)

    f["age"] = num(base["CB3"])
    f["female"] = preserve_binary(base["HL4"], 2)
    f["Out_of_School"] = preserve_binary(base["CB7"], 2)

    f["_group_PSU"] = num(base["PSU"])
    f["_group_HH1"] = num(base["HH1"])

    fs = num(base["fsweight"])
    fsh = num(base["fshweight"])

    f["weight_fs_raw"] = fs
    f["weight_fs_zero_to_nan"] = fs.replace(0.0, np.nan)
    f["weight_fsh_raw"] = fsh
    f["weight_fsh_zero_to_nan"] = fsh.replace(0.0, np.nan)

    w5 = num(base["windex5"])
    f["wealth_windex5_raw"] = w5
    f["wealth_windex5_zero_to_nan"] = w5.replace(0.0, np.nan)

    for col in ["windex5r", "windex5u"]:
        if col in base.columns:
            s = num(base[col])
            f[f"wealth_{col}_zero_to_nan"] = s.replace(0.0, np.nan)

    me = num(base["melevel"])
    f["mom_edu_raw"] = me
    f["mom_edu_9_to_0"] = me.replace({9.0: 0.0})
    f["mom_edu_9_nan_to_0"] = me.replace({9.0: 0.0}).fillna(0.0)
    f["mom_edu_nan_to_0"] = me.fillna(0.0)

    f["eth_minor_other"] = preserve_binary(base["ethnicity"], 2, missing_codes=[9])
    f["disabled_has"] = preserve_binary(base["fsdisability"], 1, missing_codes=[9])

    cl1a = num(base["CL1A"]).replace({9.0: np.nan})
    cl1x = num(base["CL1X"]).replace({9.0: np.nan})

    f["worked_CL1A"] = pd.Series(
        np.where(cl1a.isna(), np.nan, (cl1a == 1).astype(float)),
        index=base.index,
    )
    f["worked_CL1X"] = pd.Series(
        np.where(cl1x.isna(), np.nan, (cl1x == 1).astype(float)),
        index=base.index,
    )

    any_na = cl1a.isna() | cl1x.isna()
    any_yes = (cl1a == 1) | (cl1x == 1)

    f["worked_any"] = pd.Series(
        np.where(any_na, np.nan, any_yes.astype(float)),
        index=base.index,
    )

    if "CL3" in base.columns:
        f["worked_CL3_gt0"] = preserve_gt0(base["CL3"], missing_codes=[99])

    pr3 = num(base["PR3"]).replace({99.0: np.nan})

    f["books_PR3_none"] = pd.Series(
        np.where(pr3.isna(), np.nan, (pr3 == 0).astype(float)),
        index=base.index,
    )
    f["books_PR3_none_nan_to_1"] = pd.Series(
        np.where(pr3.isna(), 1.0, (pr3 == 0).astype(float)),
        index=base.index,
    )
    f["books_PR3_none_nan_to_0"] = pd.Series(
        np.where(pr3.isna(), 0.0, (pr3 == 0).astype(float)),
        index=base.index,
    )

    bands = pd.cut(
        f["age"],
        bins=[5, 9, 12, 15, 17],
        labels=["6-9", "10-12", "13-15", "16-17"],
        right=True,
        include_lowest=True,
    )

    f["_band"] = bands.astype("string")

    return f


def comparison_pairs():
    pairs = [
        ("age", "age", "age", "numeric"),
        ("_band", "_band", "_band", "categorical"),
        ("female", "female", "female", "numeric"),
        ("Out_of_School", "Out_of_School", "Out_of_School", "numeric"),
        ("_group vs PSU", "_group_PSU", "_group", "numeric"),
        ("_group vs HH1", "_group_HH1", "_group", "numeric"),
        ("FSWEIGHT vs fsweight zero->NaN", "weight_fs_zero_to_nan", "FSWEIGHT", "numeric"),
        ("FSWEIGHT vs fshweight zero->NaN", "weight_fsh_zero_to_nan", "FSWEIGHT", "numeric"),
        ("FSWEIGHT vs fsweight raw", "weight_fs_raw", "FSWEIGHT", "numeric"),
        ("FSWEIGHT vs fshweight raw", "weight_fsh_raw", "FSWEIGHT", "numeric"),
        ("wealth_q vs windex5 zero->NaN", "wealth_windex5_zero_to_nan", "wealth_q", "numeric"),
        ("wealth_q vs windex5 raw", "wealth_windex5_raw", "wealth_q", "numeric"),
        ("mom_edu vs melevel raw", "mom_edu_raw", "mom_edu", "numeric"),
        ("mom_edu vs melevel 9->0", "mom_edu_9_to_0", "mom_edu", "numeric"),
        ("mom_edu vs melevel 9/NaN->0", "mom_edu_9_nan_to_0", "mom_edu", "numeric"),
        ("mom_edu vs melevel NaN->0", "mom_edu_nan_to_0", "mom_edu", "numeric"),
        ("eth_minor vs ethnicity other", "eth_minor_other", "eth_minor", "numeric"),
        ("disabled vs fsdisability has difficulty", "disabled_has", "disabled", "numeric"),
        ("worked vs CL1A", "worked_CL1A", "worked", "numeric"),
        ("worked vs CL1X", "worked_CL1X", "worked", "numeric"),
        ("worked vs CL1A or CL1X", "worked_any", "worked", "numeric"),
        ("books_missing vs PR3 none", "books_PR3_none", "books_missing", "numeric"),
        ("books_missing vs PR3 none, NaN->1", "books_PR3_none_nan_to_1", "books_missing", "numeric"),
        ("books_missing vs PR3 none, NaN->0", "books_PR3_none_nan_to_0", "books_missing", "numeric"),
    ]

    return pairs


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise SystemExit(f"Raw file not found: {RAW.resolve()}")

    if not PHASE2.exists():
        raise SystemExit(f"Phase 2 file not found: {PHASE2.resolve()}")

    p("=" * 80)
    p("RAW PROVENANCE CHECK V3")
    p("=" * 80)
    p(f"RAW FILE: {RAW.resolve()}")
    p(f"PHASE 2 FILE: {PHASE2.resolve()}")

    df, meta = pyreadstat.read_sav(
        str(RAW),
        apply_value_formats=False,
        formats_as_category=False,
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
        p("VERDICT: FAIL - required columns missing.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    hh6 = num(df["HH6"])
    hh7 = num(df["HH7"])
    cb3 = num(df["CB3"])
    cb7 = num(df["CB7"])
    fs17 = num(df["FS17"])
    fs = num(df["fsweight"])

    mask_rural = hh6 == 2
    mask_chattogram = hh7 == 20
    mask_age = cb3.between(6, 17)
    mask_valid_cb7 = cb7.isin([1, 2])
    mask_completed = fs17 == 1
    mask_positive_weight = fs > 0

    candidates = {
        "A_rural_Chattogram_age6_17_valid_CB7": (
            mask_rural & mask_chattogram & mask_age & mask_valid_cb7
        ),
        "B_A_plus_FS17_completed": (
            mask_rural & mask_chattogram & mask_age & mask_valid_cb7 & mask_completed
        ),
        "C_A_plus_positive_fsweight": (
            mask_rural & mask_chattogram & mask_age & mask_valid_cb7 & mask_positive_weight
        ),
        "D_B_plus_positive_fsweight": (
            mask_rural
            & mask_chattogram
            & mask_age
            & mask_valid_cb7
            & mask_completed
            & mask_positive_weight
        ),
    }

    p("")
    p("CANDIDATE FILTER COUNTS")
    p("-" * 80)
    p(f"Phase 2 rows: {len(ph)}")

    for name, mask in candidates.items():
        p(f"{name}: {int(mask.sum())}")

    candidate_results = {}

    p("")
    p("POSITIONAL COMPARISONS")
    p("-" * 80)

    for name, mask in candidates.items():
        p("")
        p(f"CANDIDATE: {name}")
        p("-" * 80)

        base = df.loc[mask].copy().reset_index(drop=True)
        p(f"Raw candidate rows: {len(base)}")
        p(f"Phase 2 rows: {len(ph)}")
        p(f"Length match: {len(base) == len(ph)}")

        if len(base) != len(ph):
            p("Skipping positional comparison because lengths differ.")
            continue

        raw_feat = build_raw_features(base)

        results = {}

        for display, raw_col, phase_col, kind in comparison_pairs():
            if raw_col not in raw_feat.columns:
                res = None
            elif phase_col not in ph.columns:
                res = None
            elif kind == "numeric":
                res = match_numeric(raw_feat[raw_col], ph[phase_col])
            else:
                res = match_categorical(raw_feat[raw_col], ph[phase_col])

            results[display] = res
            p(f"{display}: {fmt(res)}")

        if "worked_CL3_gt0" in raw_feat.columns:
            res = match_numeric(raw_feat["worked_CL3_gt0"], ph["worked"])
            results["worked vs CL3 > 0"] = res
            p(f"worked vs CL3 > 0: {fmt(res)}")

        for col in ["windex5r", "windex5u"]:
            raw_col = f"wealth_{col}_zero_to_nan"
            if raw_col in raw_feat.columns:
                res = match_numeric(raw_feat[raw_col], ph["wealth_q"])
                display = f"wealth_q vs {col} zero->NaN"
                results[display] = res
                p(f"{display}: {fmt(res)}")

        p("")
        p("GROUP SET CHECKS")

        phase_group = num(ph["_group"]).dropna().unique().tolist()
        phase_group_set = set(phase_group)

        for group_name in ["_group_PSU", "_group_HH1"]:
            raw_group_set = set(raw_feat[group_name].dropna().unique().tolist())
            exact = raw_group_set == phase_group_set

            p(f"{group_name} unique count: {len(raw_group_set)}")
            p(f"Phase _group unique count: {len(phase_group_set)}")
            p(f"{group_name} exact set match: {exact}")

            if not exact:
                raw_only = sorted(list(raw_group_set - phase_group_set))[:10]
                phase_only = sorted(list(phase_group_set - raw_group_set))[:10]
                p(f"{group_name} raw-only sample: {raw_only}")
                p(f"{group_name} phase-only sample: {phase_only}")

        candidate_results[name] = results

    p("")
    p("VERDICT")
    p("-" * 80)

    if not candidate_results:
        p("PROVENANCE STATUS: FAIL")
        p("Reason: no candidate raw filter had the same row count as Phase 2.")
    else:
        key_names = [
            "age",
            "female",
            "Out_of_School",
            "_group vs PSU",
        ]

        weight_names = [
            "FSWEIGHT vs fsweight zero->NaN",
            "FSWEIGHT vs fshweight zero->NaN",
        ]

        passing_candidates = []

        for cand_name, res in candidate_results.items():
            key_ok = all(
                res.get(k) is not None and res[k][0] >= 99.999
                for k in key_names
            )

            weight_ok = any(
                res.get(w) is not None and res[w][0] >= 99.999
                for w in weight_names
            )

            if key_ok and weight_ok:
                passing_candidates.append(cand_name)

        if passing_candidates:
            p("PROVENANCE STATUS: PASS")
            p("Passing candidate filters:")
            for name in passing_candidates:
                p(f"- {name}")
            p("")
            p(
                "Interpretation: the Phase 2 file appears to be traceable to raw fs.sav "
                "under one of the candidate filters above, with row order preserved."
            )
        else:
            p("PROVENANCE STATUS: PARTIAL/FAIL")
            p(
                "At least one key deterministic field did not match positionally, "
                "or no candidate had both key fields and weight matching."
            )

        p("")
        p("BEST WEIGHT CANDIDATE")

        best_name = None
        best_pct = -1.0
        best_candidate = None

        all_weight_checks = weight_names + [
            "FSWEIGHT vs fsweight raw",
            "FSWEIGHT vs fshweight raw",
        ]

        for cand_name, res in candidate_results.items():
            for w in all_weight_checks:
                if res.get(w) is not None and res[w][0] > best_pct:
                    best_pct = res[w][0]
                    best_name = w
                    best_candidate = cand_name

        if best_name is None:
            p("none")
        else:
            p(f"Best match: {best_name}")
            p(f"In candidate: {best_candidate}")
            p(f"Match: {best_pct:.2f}%")

        p("")
        p("TRANSFORMATION RULES STILL NEEDING CONFIRMATION")

        soft_checks = [
            "_band",
            "wealth_q vs windex5 zero->NaN",
            "wealth_q vs windex5 raw",
            "mom_edu vs melevel raw",
            "mom_edu vs melevel 9->0",
            "mom_edu vs melevel 9/NaN->0",
            "mom_edu vs melevel NaN->0",
            "eth_minor vs ethnicity other",
            "disabled vs fsdisability has difficulty",
            "worked vs CL1A",
            "worked vs CL1X",
            "worked vs CL1A or CL1X",
            "worked vs CL3 > 0",
            "books_missing vs PR3 none",
            "books_missing vs PR3 none, NaN->1",
            "books_missing vs PR3 none, NaN->0",
        ]

        imperfect = []

        for cand_name, res in candidate_results.items():
            for check in soft_checks:
                if check in res and res[check] is not None and res[check][0] < 99.999:
                    imperfect.append((cand_name, check, res[check][0]))

        if imperfect:
            for cand_name, check, pct in imperfect:
                p(f"- {cand_name} | {check}: {pct:.2f}%")
        else:
            p("All tested soft transformations matched positionally in at least one candidate.")

    p("")
    p("NEXT ACTION AFTER YOU SEND THIS REPORT")
    p("-" * 80)
    p(
        "If PASS: attach original HH1/HH2/LN/PSU IDs to Phase 2, document the weight rule, "
        "decide how to handle zero-weight rows, then run the optimization dry run."
    )
    p(
        "If PARTIAL/FAIL: rebuild the Phase 2 input from raw fs.sav with original IDs and "
        "verified transformations before final modeling."
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()