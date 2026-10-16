from pathlib import Path
import argparse
import pandas as pd
import pyreadstat

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 100)


def read_survey(path: Path, apply_value_formats: bool = False):
    suffix = path.suffix.lower()

    if suffix == ".sav":
        return pyreadstat.read_sav(str(path), apply_value_formats=apply_value_formats)
    if suffix == ".dta":
        return pyreadstat.read_dta(str(path), apply_value_formats=apply_value_formats)
    if suffix == ".por":
        return pyreadstat.read_por(str(path), apply_value_formats=apply_value_formats)

    raise ValueError(f"Unsupported file type: {suffix}. Expected .sav, .dta, or .por.")


def get_label_map(meta):
    if hasattr(meta, "column_names_to_labels"):
        return meta.column_names_to_labels

    try:
        return dict(zip(meta.column_names, meta.column_labels))
    except AttributeError:
        return {c: "" for c in meta.column_names}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a MICS children age 5-17 dataset."
    )
    parser.add_argument("file", help="Path to .sav/.dta/.por file")
    args = parser.parse_args()

    path = Path(args.file)

    df, meta = read_survey(path, apply_value_formats=False)
    df_lab, _ = read_survey(path, apply_value_formats=True)
    labels = get_label_map(meta)

    print(f"File: {path}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    search_terms = [
        "weight",
        "result",
        "interview",
        "completion",
        "stratum",
        "psu",
        "cluster",
        "household",
        "area",
        "urban",
        "rural",
        "division",
        "district",
        "sex",
        "age",
        "school",
        "attend",
        "grade",
        "reading",
        "number",
        "numeracy",
        "foundational",
        "labour",
        "labor",
        "functioning",
        "disab",
        "transfer",
        "support",
        "tuition",
        "material",
    ]

    print("\nPotentially important variables:")
    matched = []
    for col, lab in labels.items():
        text = f"{col} {lab}".lower()
        if any(term in text for term in search_terms):
            matched.append((col, lab))

    for col, lab in sorted(matched):
        print(f"{col:<12} | {lab}")

    fs17 = next((c for c in df.columns if c.upper() == "FS17"), None)

    if fs17:
        print(f"\nInterview result variable: {fs17}")

        print("\nNumeric counts:")
        print(df[fs17].value_counts(dropna=False).sort_index())

        print("\nLabelled counts:")
        print(df_lab[fs17].value_counts(dropna=False).sort_index())
    else:
        print("\nWarning: FS17 not found. Check variable names/labels for interview result.")

    weight_cols = [
        c
        for c, lab in labels.items()
        if "weight" in str(lab).lower()
        or c.lower() in {"hv005", "mv005", "fv005", "cv005", "sv005"}
    ]

    print("\nWeight-like variables:", weight_cols)

    for c in weight_cols[:10]:
        print(f"\n{c}: {labels.get(c)}")
        print(df[c].describe())
        print(f"Missing: {df[c].isna().sum():,} ({df[c].isna().mean():.2%})")

    candidate_keys = [
        c for c in df.columns if c.upper() in {"FS1", "FS2", "STRATUM", "PSU", "HH1", "HH2"}
    ]

    print("\nPossible design/key variables:", candidate_keys)
    for c in candidate_keys:
        print(f"{c}: nunique={df[c].nunique(dropna=True):,}, missing={df[c].isna().sum():,}")


if __name__ == "__main__":
    main()