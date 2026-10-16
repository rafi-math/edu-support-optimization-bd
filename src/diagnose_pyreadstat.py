from pathlib import Path
import sys
import inspect

import pandas as pd
import pyreadstat


RAW = Path("data/raw/fs.sav")

KEY_COLUMNS = [
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

ID_COLUMNS = {"HH1", "HH2", "LN"}

CATEGORY_VALUE_COUNT_COLUMNS = {
    "HH6",
    "HH7",
    "CB7",
    "HL4",
    "FS17",
    "windex5",
    "melevel",
    "ethnicity",
    "fsdisability",
    "PR3",
    "CL1A",
    "CL1X",
}

NUMERIC_SUMMARY_COLUMNS = {
    "CB3",
    "fsweight",
    "fshweight",
    "PSU",
    "stratum",
}


def is_categorical(series):
    return isinstance(series.dtype, pd.CategoricalDtype)


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


def describe_column(col, series, meta):
    print(f"{col}: dtype={series.dtype}, missing={int(series.isna().sum())}, unique={int(series.nunique(dropna=True))}")

    if is_categorical(series):
        categories = list(series.cat.categories)
        print(f"  categorical category count: {len(categories)}")
        print(f"  category sample: {categories[:10]}")

        labels = get_value_labels(meta, col)

        if labels:
            print(f"  metadata value labels sample: {dict(list(labels.items())[:10])}")

            cat_str = {str(x) for x in categories[:20]}
            label_keys_str = {str(k) for k in list(labels)[:20]}
            label_vals_str = {str(v) for v in list(labels.values())[:20]}

            print(f"  categories overlap with label keys: {bool(cat_str & label_keys_str)}")
            print(f"  categories overlap with label values: {bool(cat_str & label_vals_str)}")
        else:
            print("  metadata value labels: not found")

        return

    if col in ID_COLUMNS:
        print("  ID column values suppressed")
        return

    if col in CATEGORY_VALUE_COUNT_COLUMNS:
        print("  value counts head:")
        print(series.value_counts(dropna=False).head(20).to_string())
        return

    if col in NUMERIC_SUMMARY_COLUMNS:
        numeric = pd.to_numeric(series, errors="coerce")
        print(
            f"  numeric min={numeric.min()}, max={numeric.max()}, "
            f"zeros={int((numeric == 0).sum())}, negatives={int((numeric < 0).sum())}"
        )

        if col == "CB3":
            print("  age value counts:")
            print(series.value_counts(dropna=False).sort_index().head(25).to_string())

        return

    numeric = pd.to_numeric(series, errors="coerce")
    print(f"  numeric min={numeric.min()}, max={numeric.max()}")


def main():
    print("=" * 80)
    print("PYREADSTAT DIAGNOSTIC")
    print("=" * 80)

    print(f"PROJECT ROOT: {Path.cwd().resolve()}")
    print(f"RAW FILE: {RAW.resolve()}")
    print(f"RAW EXISTS: {RAW.exists()}")

    print("")
    print("PYTHON AND PACKAGE INFO")
    print("-" * 80)
    print(f"PYTHON VERSION: {sys.version}")
    print(f"PYTHON EXECUTABLE: {sys.executable}")
    print(f"PYREADSTAT MODULE FILE: {getattr(pyreadstat, '__file__', 'unknown')}")
    print(f"PYREADSTAT VERSION: {getattr(pyreadstat, '__version__', 'no __version__ attribute')}")

    print("")
    print("READ_SAV SIGNATURE")
    print("-" * 80)

    try:
        sig = inspect.signature(pyreadstat.read_sav)
        print(f"SIGNATURE: {sig}")
        params = list(sig.parameters)
        print(f"PARAMETERS: {params}")
        print(f"HAS convert_categoricals: {'convert_categoricals' in params}")
        print(f"HAS apply_value_formats: {'apply_value_formats' in params}")
        print(f"HAS use_labels_not_values: {'use_labels_not_values' in params}")
        print(f"HAS only_read_variables: {'only_read_variables' in params}")
    except Exception as exc:
        print(f"SIGNATURE ERROR: {type(exc).__name__}: {exc}")
        params = []

    print("")
    print("METADATA-ONLY TEST")
    print("-" * 80)

    meta = None

    try:
        result = pyreadstat.read_sav(RAW, metadataonly=True)

        if isinstance(result, tuple):
            meta = result[1] if len(result) > 1 else None
        else:
            meta = result

        print("METADATA-ONLY: OK")

        if meta is not None:
            attrs = [a for a in dir(meta) if not a.startswith("_")]
            print(f"META ATTRIBUTES SAMPLE: {attrs[:50]}")

            column_names = getattr(meta, "column_names", None)
            if column_names is None:
                column_names = getattr(meta, "names", None)

            if column_names is not None:
                print(f"METADATA COLUMN COUNT: {len(column_names)}")
                print(f"METADATA HAS REQUIRED COLUMNS: {[c for c in KEY_COLUMNS if c in column_names]}")
                print(f"METADATA MISSING REQUIRED COLUMNS: {[c for c in KEY_COLUMNS if c not in column_names]}")
            else:
                print("METADATA COLUMN NAMES: not found")
        else:
            print("META OBJECT: None")

    except Exception as exc:
        print(f"METADATA-ONLY FAIL: {type(exc).__name__}: {exc}")

    print("")
    print("LOAD ATTEMPTS")
    print("-" * 80)

    attempt_specs = [
        ("default_no_kwargs", {}),
        ("apply_value_formats_false", {"apply_value_formats": False}),
        ("convert_categoricals_false", {"convert_categoricals": False}),
        ("both_false", {"convert_categoricals": False, "apply_value_formats": False}),
        ("use_labels_not_values_false", {"use_labels_not_values": False}),
        ("only_read_variables", {"only_read_variables": KEY_COLUMNS}),
        ("only_read_variables_apply_false", {"only_read_variables": KEY_COLUMNS, "apply_value_formats": False}),
    ]

    successes = {}

    for name, kwargs in attempt_specs:
        try:
            df_attempt, meta_attempt = pyreadstat.read_sav(RAW, **kwargs)
            successes[name] = (df_attempt, meta_attempt)
            print(f"{name}: OK, shape={df_attempt.shape}")
        except Exception as exc:
            print(f"{name}: FAIL, {type(exc).__name__}: {exc}")

    if not successes:
        print("")
        print("VERDICT: No load attempt succeeded. Send this full output.")
        return

    preferred_order = [
        "convert_categoricals_false",
        "both_false",
        "only_read_variables_apply_false",
        "only_read_variables",
        "apply_value_formats_false",
        "use_labels_not_values_false",
        "default_no_kwargs",
    ]

    chosen = None

    for name in preferred_order:
        if name in successes:
            chosen = name
            break

    df, meta = successes[chosen]

    print("")
    print("CHOSEN SUCCESSFUL LOAD")
    print("-" * 80)
    print(f"CHOSEN METHOD: {chosen}")
    print(f"SHAPE: {df.shape}")
    print(f"COLUMNS: {list(df.columns)[:50]}")

    missing_key = [c for c in KEY_COLUMNS if c not in df.columns]
    present_key = [c for c in KEY_COLUMNS if c in df.columns]

    print(f"PRESENT KEY COLUMNS: {present_key}")
    print(f"MISSING KEY COLUMNS: {missing_key}")

    print("")
    print("KEY COLUMN DETAILS")
    print("-" * 80)

    for col in KEY_COLUMNS:
        if col not in df.columns:
            print(f"{col}: MISSING")
            continue

        describe_column(col, df[col], meta)

    print("")
    print("END OF DIAGNOSTIC")


if __name__ == "__main__":
    main()