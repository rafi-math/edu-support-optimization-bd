from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


RAW_DIR = Path("data/raw")
RAW_DEFAULT = RAW_DIR / "fs.sav"
PHASE2 = Path("data/processed/chattogram_rural_phase2_input.csv")
REPORT = Path("reports/raw_provenance_check.txt")
PRIVATE_IDS = Path("data/processed/raw_provenance_ids.csv")

REQUIRED = [
    "HH1",
    "HH2",
    "LN",
    "HH6",
    "HH7",
    "CB3",
    "CB7",
    "HL4",
    "FSWEIGHT",
    "PSU",
]

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def find_raw_file():
    if RAW_DEFAULT.exists():
        return RAW_DEFAULT

    if RAW_DIR.exists():
        savs = sorted(RAW_DIR.glob("*.sav"))
        if savs:
            return savs[0]

    for candidate in [Path("fs.sav"), Path("data/fs.sav")]:
        if candidate.exists():
            return candidate

    return None


def get_labels(meta, col):
    for attr in [
        "variable_value_labels",
        "column_names_to_value_labels",
        "value_labels",
    ]:
        obj = getattr(meta, attr, None)
        if isinstance(obj, dict) and col in obj:
            return obj[col] or {}
    return {}


def format_labels(labels, limit=20):
    if not labels:
        return "{}"

    items = list(labels.items())[:limit]
    return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in items) + "}"


def substring_code(labels, terms):
    for val, lab in labels.items():
        s = str(lab).strip().lower()
        if any(term in s for term in terms):
            return val
    return None


def exact_code(labels, wanted):
    wanted = wanted.lower()
    for val, lab in labels.items():
        if str(lab).strip().lower() == wanted:
            return val
    return None


def num(series):
    return pd.to_numeric(series, errors="coerce")


def code_to_binary(series, code):
    s = num(series)
    return pd.Series(
        np.where(s.isna(), np.nan, (s == code).astype(float)),
        index=s.index,
    )


def match_count(raw, phase, raw_zero_to_nan=False, atol=1e-6):
    a = num(raw)
    b = num(phase)

    if len(a) != len(b):
        return 0, 0

    if raw_zero_to_nan:
        a = a.replace(0, np.nan)

    both_na = a.isna() & b.isna()

    try:
        close = np.isclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=0,
            atol=atol,
            equal_nan=False,
        )
        close = pd.Series(close, index=a.index)
    except Exception:
        close = a == b

    ok = both_na | close.fillna(False)
    return int(ok.sum()), int(len(ok))


def pct(ok, n):
    if n == 0:
        return "0/0 (0.00%)"
    return f"{ok}/{n} ({ok / n * 100:.2f}%)"


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_IDS.parent.mkdir(parents=True, exist_ok=True)

    raw_path = find_raw_file()
    if raw_path is None:
        raise SystemExit(
            "Raw .sav not found. Expected data/raw/fs.sav "
            "or another .sav file inside data/raw/."
        )

    p("=" * 80)
    p("RAW PROVENANCE CHECK")
    p("=" * 80)
    p(f"RAW FILE: {raw_path.resolve()}")

    df, meta = pyreadstat.read_sav(raw_path, apply_value_formats=False)
    p(f"RAW SHAPE: {df.shape}")

    missing_required = [c for c in REQUIRED if c not in df.columns]
    p(f"MISSING REQUIRED RAW COLUMNS: {missing_required if missing_required else 'none'}")

    if missing_required:
        p("")
        p("Cannot complete full provenance check because required raw columns are missing.")
        p("Send this report back.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    label_cols = ["HH6", "HH7", "CB3", "CB7", "HL4", "FS17", "FSWEIGHT", "PSU"]
    labels = {c: get_labels(meta, c) for c in label_cols}
    col_labels = getattr(meta, "column_names_to_labels", {}) or {}

    p("")
    p("VARIABLE LABELS")
    p("-" * 80)

    for c in label_cols:
        p(f"{c}: {col_labels.get(c, 'NO COLUMN LABEL')}")
        p(f"  value labels: {format_labels(labels[c])}")

    rural_code = substring_code(labels["HH6"], ["rural"])
    division_code = substring_code(labels["HH7"], ["chattogram", "chittagong"])
    cb7_yes = exact_code(labels["CB7"], "yes")
    cb7_no = exact_code(labels["CB7"], "no")
    female_code = substring_code(labels["HL4"], ["female"])
    completed_code = exact_code(labels.get("FS17", {}), "completed")

    if rural_code is None:
        rural_code = 2
        p("NOTE: rural code not found in HH6 labels; using expected 2.")

    if division_code is None:
        division_code = 20
        p("NOTE: Chattogram code not found in HH7 labels; using expected 20.")

    if cb7_yes is None:
        cb7_yes = 1
        p("NOTE: CB7 attended/yes code not found; using expected 1.")

    if cb7_no is None:
        cb7_no = 2
        p("NOTE: CB7 not attended/no code not found; using expected 2.")

    if female_code is None:
        female_code = 2
        p("NOTE: female code not found in HL4 labels; using expected 2.")

    if completed_code is None and "FS17" in df.columns:
        completed_code = 1
        p("NOTE: FS17 completed code not found; using expected 1.")

    p("")
    p("DETECTED CODES")
    p("-" * 80)
    p(f"rural HH6 code: {rural_code}")
    p(f"Chattogram HH7 code: {division_code}")
    p(f"CB7 attended code: {cb7_yes}")
    p(f"CB7 not attended code: {cb7_no}")
    p(f"HL4 female code: {female_code}")
    p(f"FS17 completed code: {completed_code}")

    hh6 = num(df["HH6"])
    hh7 = num(df["HH7"])
    cb3 = num(df["CB3"])
    cb7 = num(df["CB7"])

    mask_rural = hh6 == rural_code
    mask_div = hh7 == division_code
    mask_age = cb3.between(6, 17)
    mask_cb7 = cb7.isin([cb7_yes, cb7_no])
    base_mask = mask_rural & mask_div & mask_age & mask_cb7

    p("")
    p("FILTER COUNTS")
    p("-" * 80)
    p(f"Total rows: {len(df)}")
    p(f"Rural rows: {int(mask_rural.sum())}")
    p(f"Rural Chattogram rows: {int((mask_rural & mask_div).sum())}")
    p(f"Rural Chattogram age 6-17 rows: {int((mask_rural & mask_div & mask_age).sum())}")
    p(f"Rural Chattogram age 6-17 valid CB7 rows: {int(base_mask.sum())}")

    base = df.loc[base_mask].copy().reset_index(drop=True)
    n_base = len(base)

    if "FS17" in base.columns:
        fs17 = num(base["FS17"])
        completed_mask = fs17 == completed_code

        p("")
        p("FS17 INTERVIEW RESULT IN BASE")
        p("-" * 80)
        p(fs17.value_counts(dropna=False).sort_index().to_string())
        p(f"Completed interviews in base: {int(completed_mask.sum())}")
    else:
        completed_mask = pd.Series(True, index=base.index)
        p("FS17 column not found; skipping interview-result split.")

    w = num(base["FSWEIGHT"])

    p("")
    p("FSWEIGHT IN BASE")
    p("-" * 80)
    p(f"missing: {int(w.isna().sum())}")
    p(f"zero: {int((w == 0).sum())}")
    p(f"negative: {int((w < 0).sum())}")
    p(f"positive: {int((w > 0).sum())}")

    raw_age = num(base["CB3"])
    raw_female = code_to_binary(base["HL4"], female_code)
    raw_oos = code_to_binary(base["CB7"], cb7_no)
    raw_psu = num(base["PSU"])

    pos_w = w > 0

    p("")
    p("RAW OUTCOME RATES")
    p("-" * 80)
    p(f"Base unweighted OOS count: {int(raw_oos.sum(skipna=True))}")
    p(f"Base unweighted OOS rate: {float(raw_oos.mean(skipna=True)):.6f}")

    if int(pos_w.sum()) > 0:
        weighted_rate = float(
            np.average(
                raw_oos[pos_w].to_numpy(dtype=float),
                weights=w[pos_w].to_numpy(dtype=float),
            )
        )
        p(f"Base weighted OOS rate using positive FSWEIGHT: {weighted_rate:.6f}")
    else:
        p("Base weighted OOS rate: not computable")

    completed_pos = completed_mask & pos_w

    p("")
    p("CANDIDATE ANALYTIC SAMPLES")
    p("-" * 80)
    p(f"A. valid CB7 base: {n_base}")
    p(f"B. base + FS17 completed: {int(completed_mask.sum())}")
    p(f"C. base + positive FSWEIGHT: {int(pos_w.sum())}")
    p(f"D. base + FS17 completed + positive FSWEIGHT: {int(completed_pos.sum())}")

    p("")
    p("PHASE 2 COMPARISON")
    p("-" * 80)

    if not PHASE2.exists():
        p(f"Phase 2 file not found: {PHASE2}")
    else:
        ph = pd.read_csv(PHASE2, low_memory=False).reset_index(drop=True)
        p(f"Phase 2 shape: {ph.shape}")
        p(f"Length match with raw base: {len(ph) == n_base}")

        if len(ph) == n_base:
            p("")
            p("Positional match checks, assuming exported order was preserved:")

            checks = [
                ("age", raw_age, ph.get("age"), {}),
                ("female", raw_female, ph.get("female"), {}),
                ("Out_of_School", raw_oos, ph.get("Out_of_School"), {}),
                ("PSU/_group", raw_psu, ph.get("_group"), {}),
                ("FSWEIGHT zero-to-NaN", w, ph.get("FSWEIGHT"), {"raw_zero_to_nan": True}),
            ]

            for name, raw_s, ph_s, kwargs in checks:
                if ph_s is None:
                    p(f"{name}: phase 2 column missing")
                else:
                    ok, n = match_count(raw_s, ph_s, **kwargs)
                    p(f"{name}: {pct(ok, n)}")

            psu_set = set(pd.Series(raw_psu).dropna().unique().tolist())
            grp_set = set(
                pd.to_numeric(ph.get("_group"), errors="coerce")
                .dropna()
                .unique()
                .tolist()
            )

            p("")
            p(f"Raw base unique PSU count: {len(psu_set)}")
            p(f"Phase 2 unique _group count: {len(grp_set)}")
            p(f"PSU/_group exact set match: {psu_set == grp_set}")

            if psu_set != grp_set:
                raw_only = sorted(list(psu_set - grp_set))[:10]
                phase_only = sorted(list(grp_set - psu_set))[:10]
                p(f"Raw-only PSU sample: {raw_only}")
                p(f"Phase-only _group sample: {phase_only}")
        else:
            p("")
            p("Row counts differ. Do not attach IDs by position yet.")

    p("")
    p("PRIVATE ID FILE")
    p("-" * 80)

    id_cols = [
        c
        for c in [
            "HH1",
            "HH2",
            "LN",
            "PSU",
            "HH6",
            "HH7",
            "CB3",
            "CB7",
            "HL4",
            "FS17",
            "FSWEIGHT",
        ]
        if c in base.columns
    ]

    private = base[id_cols].copy()
    private.insert(0, "raw_row_order", np.arange(len(private)))
    private.to_csv(PRIVATE_IDS, index=False)

    p(f"Saved private raw provenance IDs: {PRIVATE_IDS.resolve()}")
    p("Do not commit this file. It contains survey microdata identifiers.")

    p("")
    p("SEND BACK")
    p("-" * 80)
    p("Send reports/raw_provenance_check.txt only.")
    p("Do not send data/processed/raw_provenance_ids.csv.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()