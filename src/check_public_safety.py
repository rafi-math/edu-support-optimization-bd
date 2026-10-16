from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCAN_DIRS = [
    ROOT / "docs",
    ROOT / "reports",
    ROOT / "results",
]

FORBIDDEN_EXTENSIONS = {
    ".sav",
    ".dta",
    ".por",
    ".xpt",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
}

FORBIDDEN_CSV_COLUMNS = {
    "child_uid",
    "hh1",
    "hh2",
    "ln",
    "psu",
    "stratum",
    "household_key",
}

issues = []


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def main():
    print("=" * 80)
    print("PUBLIC SAFETY CHECK")
    print("=" * 80)

    for base in SCAN_DIRS:
        print(f"Scanning: {base.resolve()}")

        if not base.exists():
            issues.append(f"Missing expected directory: {base}")
            continue

        for path in base.rglob("*"):
            if not path.is_file():
                continue

            ext = path.suffix.lower()

            if ext in FORBIDDEN_EXTENSIONS:
                issues.append(f"Forbidden binary/data file: {rel(path)}")

            if ext == ".csv":
                try:
                    cols = pd.read_csv(path, nrows=0).columns
                    lowered = {str(c).strip().lower() for c in cols}
                    bad = sorted(lowered & FORBIDDEN_CSV_COLUMNS)

                    if bad:
                        issues.append(
                            f"CSV appears to contain child/household identifiers: "
                            f"{rel(path)} -> {bad}"
                        )
                except Exception as exc:
                    issues.append(
                        f"Could not inspect CSV {rel(path)}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    print("")
    print("ISSUES")
    print("-" * 80)

    if not issues:
        print("none")
    else:
        for item in issues:
            print(f"- {item}")

    print("")
    if issues:
        print("PUBLIC SAFETY CHECK: FAIL")
        print("Do not commit or push. Send this output.")
        sys.exit(1)
    else:
        print("PUBLIC SAFETY CHECK: PASS")
        print("No forbidden binary files or identifier-bearing CSVs were found in docs/, reports/, or results/.")


if __name__ == "__main__":
    main()