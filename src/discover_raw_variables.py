from pathlib import Path
import re

import pyreadstat


RAW_DIR = Path("data/raw")
REPORT = Path("reports/raw_variable_discovery.txt")

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def get_column_names(meta):
    names = getattr(meta, "column_names", None)
    if names is None:
        names = getattr(meta, "names", None)
    if names is None:
        return []
    return list(names)


def get_column_labels(meta, column_names):
    labels = getattr(meta, "column_labels", None)
    if labels is None:
        labels = getattr(meta, "labels", None)

    if labels is None:
        return {c: "" for c in column_names}

    labels = list(labels)

    if len(labels) != len(column_names):
        return {c: "" for c in column_names}

    return dict(zip(column_names, labels))


def get_value_labels(meta, col):
    candidates = [
        "variable_value_labels",
        "column_names_to_value_labels",
        "value_labels",
    ]

    for attr in candidates:
        obj = getattr(meta, attr, None)
        if isinstance(obj, dict) and col in obj:
            val = obj[col]
            if isinstance(val, dict):
                return val

    return {}


def format_labels(labels, limit=30):
    if not labels:
        return "{}"

    items = list(labels.items())[:limit]
    return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in items) + "}"


def is_weight_name(col):
    u = str(col).upper()

    exact = {
        "WGT",
        "WEIGHT",
        "FSWEIGHT",
        "HVWT",
        "CHWT",
        "WMWT",
        "HHWT",
        "SAMPLE_WEIGHT",
        "FINAL_WEIGHT",
        "POPULATION_WEIGHT",
    }

    if u in exact:
        return True

    if "WEIGHT" in u:
        return True

    if "WGT" in u:
        return True

    if u.endswith("WT"):
        return True

    return False


def is_key_column(col, label):
    text = f"{col} {label}".upper()

    terms = [
        "HH1",
        "HH2",
        "LN",
        "HH6",
        "HH7",
        "HH7A",
        "CB3",
        "CB7",
        "HL4",
        "PSU",
        "STRATUM",
        "REGION",
        "DIVISION",
        "DISTRICT",
        "UPAZILA",
        "THANA",
        "AREA",
        "URBAN",
        "RURAL",
        "FS17",
        "EDUCATION",
        "SCHOOL",
        "LITERACY",
        "WEALTH",
        "DISABIL",
        "WORK",
        "BOOK",
        "ETHNIC",
        "RELIGION",
        "MOTHER",
        "MARITAL",
        "SEX",
        "AGE",
    ]

    return any(term in text for term in terms)


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise SystemExit(f"Raw folder not found: {RAW_DIR.resolve()}")

    sav_files = sorted(RAW_DIR.glob("*.sav"))

    if not sav_files:
        raise SystemExit(f"No .sav files found in {RAW_DIR.resolve()}")

    p("=" * 80)
    p("RAW VARIABLE DISCOVERY")
    p("=" * 80)
    p(f"RAW DIR: {RAW_DIR.resolve()}")
    p(f"FILES FOUND: {[str(f.name) for f in sav_files]}")

    fsweight_found_in = []
    all_weight_like = {}
    all_key_columns = {}

    for path in sav_files:
        p("")
        p("=" * 80)
        p(f"FILE: {path.name}")
        p("=" * 80)

        try:
            _, meta = pyreadstat.read_sav(path, metadataonly=True)
        except Exception as exc:
            p(f"ERROR reading metadata: {exc}")
            continue

        cols = get_column_names(meta)
        labels = get_column_labels(meta, cols)

        p(f"COLUMN COUNT: {len(cols)}")

        if "FSWEIGHT" in cols:
            fsweight_found_in.append(path.name)
            p("FSWEIGHT EXACT MATCH: YES")
        else:
            p("FSWEIGHT EXACT MATCH: NO")

        weight_cols = [c for c in cols if is_weight_name(c)]
        all_weight_like[path.name] = weight_cols

        p("")
        p("WEIGHT-LIKE COLUMNS")
        p("-" * 80)

        if not weight_cols:
            p("None found.")
        else:
            for col in weight_cols:
                label = labels.get(col, "")
                p(f"{col} | label: {label}")

        key_cols = []

        for col in cols:
            label = labels.get(col, "")
            if is_key_column(col, label):
                key_cols.append(col)

        all_key_columns[path.name] = key_cols

        p("")
        p("KEY GEOGRAPHY / DEMOGRAPHIC / EDUCATION COLUMNS")
        p("-" * 80)

        if not key_cols:
            p("None found from keyword list.")
        else:
            for col in key_cols:
                label = labels.get(col, "")
                vlabels = get_value_labels(meta, col)

                p("")
                p(f"COLUMN: {col}")
                p(f"LABEL: {label}")

                if vlabels:
                    p(f"VALUE LABELS: {format_labels(vlabels)}")
                else:
                    p("VALUE LABELS: {}")

        p("")
        p("FULL COLUMN LIST")
        p("-" * 80)

        for i, col in enumerate(cols, start=1):
            label = labels.get(col, "")
            p(f"{i:03d}. {col} | {label}")

    p("")
    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)

    p("")
    p("FSWEIGHT exact presence by file:")

    if fsweight_found_in:
        for name in fsweight_found_in:
            p(f"- {name}: YES")
    else:
        p("- No raw .sav file contains a column exactly named FSWEIGHT.")

    p("")
    p("Weight-like columns by file:")

    for name, cols in all_weight_like.items():
        p(f"- {name}: {cols if cols else 'none'}")

    p("")
    p("Important next check:")
    p("If no FSWEIGHT exists, the Phase 2 FSWEIGHT column must have come from:")
    p("1. a renamed weight variable;")
    p("2. a merge from another raw file;")
    p("3. a computed weight;")
    p("4. or an incorrect/unsupported transformation.")
    p("Do not use it for final optimization until this is resolved.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()