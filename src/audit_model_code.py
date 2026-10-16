from pathlib import Path
import json
import re


ROOT = Path(".").resolve()
REPORT = Path("reports/model_code_audit.txt")

SKIP_DIR_NAMES = {
    ".venv",
    ".git",
    "__pycache__",
    "data",
    "reports",
    "results",
    "logs",
    "node_modules",
    "build",
    "dist",
}

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".R",
    ".r",
    ".m",
    ".tex",
    ".cpp",
    ".h",
    ".hpp",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
}

MAX_BYTES = 5_000_000

PATTERNS = [
    (
        "weight_usage",
        re.compile(
            r"(?i)(sample_weight|class_weight|fsweight|fshweight|FSWEIGHT|"
            r"WEIGHT_COL|\bw\s*=|weights\s*=)"
        ),
        30,
    ),
    (
        "books_feature",
        re.compile(
            r"(?i)(books_missing|PR3|children.{0,12}books|book.{0,12}count|"
            r"book.{0,12}missing)"
        ),
        30,
    ),
    (
        "model_training",
        re.compile(
            r"(?i)(StratifiedGroupKFold|cross_val_predict|cross_validate|"
            r"fit\(|predict_proba|risk_score|risk_std|LogisticRegression|"
            r"RandomForest|MLPClassifier|GradientBoosting|XGB|LightGBM)"
        ),
        40,
    ),
    (
        "export_phase2",
        re.compile(
            r"(?i)(chattogram_rural_phase2_input|phase2_input|"
            r"chattogram_rural_model|to_csv|to_parquet)"
        ),
        25,
    ),
]

lines = []
match_counts = {}


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def should_skip(path):
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError:
        parts = path.parts

    return any(part in SKIP_DIR_NAMES for part in parts)


def add_match(label, source, line_no, line):
    key = label
    match_counts[key] = match_counts.get(key, 0) + 1

    limit = dict((lbl, lim) for lbl, _, lim in PATTERNS)[label]

    if match_counts[key] > limit * 20:
        return

    clean = " ".join(line.strip().split())
    if len(clean) > 300:
        clean = clean[:300] + "..."

    lines.append(f"[{label}] {source}:{line_no}: {clean}")


def search_text(label, regex, source, text):
    found = 0
    limit = dict((lbl, lim) for lbl, _, lim in PATTERNS)[label]

    for line_no, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            add_match(label, source, line_no, line)
            found += 1

            if found >= limit:
                lines.append(
                    f"[{label}] ... additional matches suppressed for {source} ..."
                )
                break

    return found


def scan_notebook(path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
    except Exception as exc:
        lines.append(f"[notebook_parse_error] {rel(path)}: {type(exc).__name__}: {exc}")
        return

    cells = obj.get("cells", [])

    if not isinstance(cells, list):
        lines.append(f"[notebook_parse_warning] {rel(path)}: cells field is not a list")
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

        cell_label = f"{rel(path)}#cell{idx}"

        for label, regex, _ in PATTERNS:
            search_text(label, regex, cell_label, text)


def scan_text_file(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        lines.append(f"[read_error] {rel(path)}: {type(exc).__name__}: {exc}")
        return

    for label, regex, _ in PATTERNS:
        search_text(label, regex, rel(path), text)


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    p("=" * 80)
    p("MODEL CODE AUDIT")
    p("=" * 80)
    p(f"ROOT: {ROOT}")
    p(f"REPORT: {REPORT.resolve()}")
    p("")
    p("This scan looks for code that may explain how risk_score, risk_std,")
    p("weights, and books_missing were created. It skips data/, reports/,")
    p("results/, .venv/, and .git/.")

    files = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if should_skip(path):
            continue

        suffix = path.suffix.lower()

        if suffix == ".ipynb":
            files.append(path)
        elif suffix in TEXT_EXTS:
            files.append(path)

    files = sorted(files, key=lambda x: str(rel(x)))

    p("")
    p(f"FILES SCANNED: {len(files)}")

    if not files:
        p("No code/text files found outside skipped folders.")

    large_skipped = []

    for path in files:
        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        if size > MAX_BYTES:
            large_skipped.append((rel(path), size))
            continue

        if path.suffix.lower() == ".ipynb":
            scan_notebook(path)
        else:
            scan_text_file(path)

    if large_skipped:
        p("")
        p("LARGE FILES SKIPPED:")
        for path, size in large_skipped:
            p(f"- {path}: {size} bytes")

    p("")
    p("MATCH COUNTS BY CATEGORY")
    p("-" * 80)

    if not match_counts:
        p("No matches found.")
    else:
        for label, _, _ in PATTERNS:
            p(f"{label}: {match_counts.get(label, 0)}")

    p("")
    p("MATCHES")
    p("-" * 80)

    if not any(line.startswith("[") for line in lines):
        p("No relevant code lines found.")
    else:
        for line in lines:
            if line.startswith("["):
                p(line)

    p("")
    p("HOW TO READ THIS")
    p("-" * 80)
    p("If you see sample_weight or weights passed into fit(), the model used weights.")
    p("If the weight variable is fshweight or the Phase 2 FSWEIGHT column, the risk score")
    p("may need retraining/revalidation with fsweight.")
    p("If books_missing is created from PR3.isna(), it is a missing-data indicator, not")
    p("a no-books indicator.")
    p("If no local code is found, the model code is probably in Colab. Send this report,")
    p("then paste the Colab cell that trained the models and exported the Phase 2 CSV.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()