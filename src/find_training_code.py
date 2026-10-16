from pathlib import Path
import json
import re


ROOT = Path(".").resolve()
REPORT = Path("reports/training_code_search.txt")

SKIP_DIRS = {
    ".venv",
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "data",
    "reports",
    "results",
    "logs",
    "node_modules",
    "build",
    "dist",
}

EXCLUDE_PARTS = [
    "audit_model_code",
    "check_raw",
    "check_remaining",
    "discover_raw",
    "diagnose_pyreadstat",
    "resolve_books",
    "test_data",
    "inspect_mics",
    "find_training_code",
]

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
}

NOTEBOOK_EXT = ".ipynb"
MAX_BYTES = 2_000_000
MAX_MATCH_LINES_PER_CATEGORY = 5

PATTERNS = {
    "training": re.compile(
        r"(?i)("
        r"\.fit\s*\(|"
        r"fit_predict|"
        r"predict_proba|"
        r"cross_val_predict|"
        r"cross_validate|"
        r"StratifiedGroupKFold|"
        r"LogisticRegression|"
        r"RandomForest|"
        r"MLPClassifier|"
        r"GradientBoosting|"
        r"sample_weight|"
        r"class_weight"
        r")"
    ),
    "risk": re.compile(
        r"(?i)(risk_score|risk_std)"
    ),
    "weight": re.compile(
        r"(?i)("
        r"fsweight|"
        r"fshweight|"
        r"FSWEIGHT|"
        r"sample_weight\s*=|"
        r"weights\s*="
        r")"
    ),
    "books": re.compile(
        r"(?i)(books_missing|PR3)"
    ),
    "export": re.compile(
        r"(?i)("
        r"chattogram_rural_phase2_input|"
        r"phase2_input|"
        r"to_csv"
        r")"
    ),
}

lines = []
counts = {}
examples = {}


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def should_skip(path):
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        parts = path.parts

    return any(part in SKIP_DIRS for part in parts)


def excluded_name(path):
    name = path.name.lower()
    return any(part in name for part in EXCLUDE_PARTS)


def add_match(category, source, line_no, text):
    key = (source, category)
    counts[key] = counts.get(key, 0) + 1

    bucket = examples.setdefault(key, [])

    if len(bucket) < MAX_MATCH_LINES_PER_CATEGORY:
        clean = " ".join(str(text).strip().split())

        if len(clean) > 220:
            clean = clean[:220] + "..."

        bucket.append(f"[{category}] {source}:{line_no}: {clean}")


def scan_lines(source, line_items):
    for line_no, line in line_items:
        for category, regex in PATTERNS.items():
            if regex.search(line):
                add_match(category, source, line_no, line)


def scan_text_file(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        p(f"[read_error] {rel(path)}: {type(exc).__name__}: {exc}")
        return

    scan_lines(rel(path), enumerate(text.splitlines(), start=1))


def scan_notebook(path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
    except Exception as exc:
        p(f"[notebook_parse_error] {rel(path)}: {type(exc).__name__}: {exc}")
        return

    cells = obj.get("cells", [])

    if not isinstance(cells, list):
        p(f"[notebook_parse_warning] {rel(path)}: cells field is not a list")
        return

    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue

        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])

        if isinstance(source, list):
            text = "".join(str(x) for x in source)
        else:
            text = str(source)

        label = f"{rel(path)}#cell{idx}"
        scan_lines(label, enumerate(text.splitlines(), start=1))


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    p("=" * 80)
    p("TRAINING CODE SEARCH")
    p("=" * 80)
    p(f"ROOT: {ROOT}")
    p(f"REPORT: {REPORT.resolve()}")
    p("")
    p("This search ignores data folders, reports, results, .venv, .git,")
    p("and the broad audit/check scripts we already ran.")
    p("It looks only for likely model-training, weighting, books-feature,")
    p("risk-score, and Phase 2 export code.")

    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if should_skip(path):
            continue

        if excluded_name(path):
            continue

        suffix = path.suffix.lower()

        if suffix in TEXT_EXTS or suffix == NOTEBOOK_EXT:
            files.append(path)

    files = sorted(files, key=lambda x: rel(x))

    p("")
    p(f"FILES SCANNED: {len(files)}")

    large_skipped = []

    for path in files:
        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        if size > MAX_BYTES:
            large_skipped.append((rel(path), size))
            continue

        if path.suffix.lower() == NOTEBOOK_EXT:
            scan_notebook(path)
        else:
            scan_text_file(path)

    if large_skipped:
        p("")
        p("LARGE FILES SKIPPED:")
        for path, size in large_skipped:
            p(f"- {path}: {size} bytes")

    p("")
    p("SUMMARY")
    p("-" * 80)

    if not counts:
        p("No relevant training/export code found in the local project files.")
        p("")
        p("This likely means the model-training code is only in Google Colab,")
        p("or it was not saved inside this project folder.")
    else:
        sources = sorted({source for source, category in counts})

        for source in sources:
            p("")
            p(source)

            for category in PATTERNS:
                n = counts.get((source, category), 0)

                if n:
                    p(f"  {category}: {n}")

        p("")
        p("MATCH EXAMPLES")
        p("-" * 80)

        for key in sorted(examples):
            for line in examples[key]:
                p(line)

    p("")
    p("HOW TO READ THIS")
    p("-" * 80)
    p("If training matches include sample_weight, the model probably used weights.")
    p("If weight matches show fshweight or the Phase 2 FSWEIGHT column, the risk score")
    p("may need revalidation using raw fsweight.")
    p("If books matches show PR3.isna(), books_missing is a missing-data flag,")
    p("not a no-books flag.")
    p("If no training matches are found, the next step is to copy the model-training")
    p("cells from Google Colab.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()