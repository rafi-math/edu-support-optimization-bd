from pathlib import Path
import html
import shutil

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
DATA = DOCS / "data"

FIG_PNG = ROOT / "results" / "figures" / "phase3_fairness_curves_publication.png"
FIG_PDF = ROOT / "results" / "figures" / "phase3_fairness_curves_publication.pdf"

SUMMARY_CSV = ROOT / "results" / "phase3_fairness_summary.csv"
CURVES_CSV = ROOT / "results" / "phase3_fairness_curves.csv"

REPO = "https://github.com/rafi-math/edu-support-optimization-bd"
AUTHOR = "Md. Ali Arman Rafi"
AFFILIATION = "Department of Mathematics, University of Chittagong, Bangladesh"

TITLE = "Fairness-Constrained Allocation of Educational Support in Rural Chattogram"
SUBTITLE = "A reproducible risk-scoring and optimization project showcase"


def fail(msg):
    raise SystemExit(msg)


def require_file(path, label):
    if not path.exists():
        fail(f"{label} not found: {path.resolve()}")


def fmt_num(x, nd=2):
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


def fmt_pct(x, nd=2):
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.{nd}f}%"


def df_to_html(df):
    if df.empty:
        return "<p>No rows.</p>"

    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if pd.isna(v):
                s = ""
            else:
                s = str(v)
            cells.append(f"<td>{html.escape(s)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table class='data-table'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def build_metrics_table():
    rows = [
        ("Raw rural Chattogram sample", "4,891 children"),
        ("Modeling-ready sample", "4,861 children"),
        ("Selection budget K", "730 children, about 15%"),
        ("Observed out-of-school children", "641"),
        ("Unweighted out-of-school rate", "13.19%"),
        ("Provisional child-weighted out-of-school rate", "14.52%"),
        ("Out-of-fold ROC-AUC, unweighted", "0.8398"),
        ("Out-of-fold ROC-AUC, provisional child-weighted", "0.8520"),
        ("Raw-risk observed OOS captured", "332 of 641, 51.8%"),
        ("Raw-risk selected share aged < 13", "0.14%, only 1 child"),
        ("Raw-risk selected share girls", "29.04%"),
        ("Age+gender interaction, unweighted objective", "-5.740 objective units, subadditive"),
    ]
    return pd.DataFrame(rows, columns=["Item", "Verified value"])


def build_summary_table():
    df = pd.read_csv(SUMMARY_CSV, low_memory=False)

    need = [
        "scenario",
        "young_floor",
        "girls_floor",
        "young_share",
        "girls_share",
        "poorest_share",
        "obj_unweighted",
        "obj_drop_vs_raw",
        "obj_drop_pct_vs_raw",
        "observed_oos_captured",
        "capture_drop_vs_raw",
        "interaction_obj_drop",
        "proven_optimal",
    ]

    missing = [c for c in need if c not in df.columns]
    if missing:
        fail(f"Summary CSV missing columns: {missing}")

    out = pd.DataFrame()
    out["Scenario"] = df["scenario"].astype(str).str.replace("_", " ")
    out["Young floor"] = df["young_floor"].astype(int)
    out["Girls floor"] = df["girls_floor"].astype(int)
    out["Young %"] = df["young_share"].map(lambda x: fmt_pct(x, 1))
    out["Girls %"] = df["girls_share"].map(lambda x: fmt_pct(x, 1))
    out["Poorest %"] = df["poorest_share"].map(lambda x: fmt_pct(x, 1))
    out["Objective"] = df["obj_unweighted"].map(lambda x: fmt_num(x, 2))
    out["Obj drop"] = df["obj_drop_vs_raw"].map(lambda x: fmt_num(x, 2))
    out["Obj drop %"] = df["obj_drop_pct_vs_raw"].map(lambda x: fmt_pct(x, 2))
    out["Captured OOS"] = df["observed_oos_captured"].astype(int)
    out["Capture drop"] = df["capture_drop_vs_raw"].astype(int)
    out["Interaction"] = df["interaction_obj_drop"].map(lambda x: fmt_num(x, 3))
    out["Proven optimal"] = df["proven_optimal"].map(
        lambda x: "yes" if str(x).lower() in ("true", "1", "yes") else "no"
    )

    return out


def write_css():
    css = """
:root {
  --text: #1f2933;
  --muted: #52606d;
  --line: #cbd2d9;
  --bg: #ffffff;
  --soft: #f5f7fa;
  --blue: #0072b2;
  --green: #009e73;
  --orange: #d55e00;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--text);
  background: var(--bg);
  line-height: 1.55;
}

header {
  padding: 2.2rem 1.5rem 1.4rem;
  border-bottom: 1px solid var(--line);
  background: var(--soft);
}

main {
  max-width: 1050px;
  margin: 0 auto;
  padding: 1.5rem;
}

h1 {
  margin: 0 0 0.45rem;
  font-size: 1.85rem;
  line-height: 1.2;
}

h2 {
  margin-top: 2rem;
  font-size: 1.35rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.35rem;
}

h3 {
  margin-top: 1.4rem;
  font-size: 1.08rem;
}

p {
  margin: 0.75rem 0;
}

.subtitle,
.meta {
  color: var(--muted);
}

.meta {
  font-size: 0.95rem;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 1rem 0 0;
}

.badge {
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: white;
  font-size: 0.85rem;
  color: var(--muted);
}

.callout {
  border-left: 4px solid var(--blue);
  background: #f0f7ff;
  padding: 0.9rem 1rem;
  margin: 1rem 0;
}

.callout.green {
  border-left-color: var(--green);
  background: #effaf6;
}

.callout.orange {
  border-left-color: var(--orange);
  background: #fff5ee;
}

.figure {
  width: 100%;
  height: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  margin: 1rem 0;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.92rem;
}

.data-table th,
.data-table td {
  border-bottom: 1px solid var(--line);
  padding: 0.48rem 0.55rem;
  text-align: left;
  vertical-align: top;
}

.data-table th {
  background: var(--soft);
  font-weight: 700;
}

ul, ol {
  margin: 0.6rem 0 0.6rem 1.25rem;
}

li {
  margin: 0.28rem 0;
}

a {
  color: var(--blue);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.small {
  font-size: 0.88rem;
  color: var(--muted);
}

.footer {
  margin-top: 2.5rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.88rem;
}

.code {
  font-family: Consolas, Menlo, Monaco, monospace;
  background: var(--soft);
  padding: 0.12rem 0.28rem;
  border-radius: 4px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}

.card {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 1rem;
  background: white;
}

.card h3 {
  margin-top: 0;
}
"""
    (DOCS / "styles.css").write_text(css, encoding="utf-8")


def main():
    require_file(FIG_PNG, "Figure PNG")
    require_file(SUMMARY_CSV, "Summary CSV")
    require_file(CURVES_CSV, "Curves CSV")

    DOCS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    shutil.copy2(FIG_PNG, ASSETS / FIG_PNG.name)

    pdf_exists = FIG_PDF.exists()
    if pdf_exists:
        shutil.copy2(FIG_PDF, ASSETS / FIG_PDF.name)

    shutil.copy2(SUMMARY_CSV, DATA / SUMMARY_CSV.name)
    shutil.copy2(CURVES_CSV, DATA / CURVES_CSV.name)

    (DOCS / ".nojekyll").write_text("", encoding="utf-8")

    write_css()

    metrics = build_metrics_table()
    summary = build_summary_table()

    parts = []

    parts.append("<!doctype html>")
    parts.append("<html lang='en'>")
    parts.append("<head>")
    parts.append("<meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>" + html.escape(TITLE) + "</title>")
    parts.append("<link rel='stylesheet' href='styles.css'>")
    parts.append("<script>")
    parts.append("MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']]}};")
    parts.append("</script>")
    parts.append(
        "<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js' async></script>"
    )
    parts.append("</head>")
    parts.append("<body>")

    parts.append("<header>")
    parts.append("<h1>" + html.escape(TITLE) + "</h1>")
    parts.append("<div class='subtitle'>" + html.escape(SUBTITLE) + "</div>")
    parts.append("<div class='meta'>" + html.escape(AUTHOR) + " &middot; " + html.escape(AFFILIATION) + "</div>")
    parts.append("<div class='meta'>Repository: <a href='" + html.escape(REPO) + "'>" + html.escape(REPO) + "</a></div>")
    parts.append(
        "<div class='badge-row'>"
        "<span class='badge'>MICS 2019</span>"
        "<span class='badge'>Rural Chattogram</span>"
        "<span class='badge'>Ages 6-17</span>"
        "<span class='badge'>Out-of-fold risk score</span>"
        "<span class='badge'>Fairness-constrained MILP</span>"
        "<span class='badge'>Aggregate public outputs only</span>"
        "</div>"
    )
    parts.append("</header>")

    parts.append("<main>")

    parts.append("<h2>Project summary</h2>")
    parts.append(
        "<p>This project builds a reproducible decision-support workflow for allocating limited educational support "
        "among rural children aged 6-17 in Chattogram, Bangladesh. It starts from verified MICS 2019 microdata, "
        "constructs an out-of-fold out-of-school risk score, and then prices the trade-off between predictive reach "
        "and explicit fairness coverage constraints.</p>"
    )
    parts.append(
        "<div class='callout'>"
        "<strong>Main finding:</strong> raw risk ranking captures many out-of-school adolescents but almost excludes "
        "younger children. Age coverage is expensive. Gender parity is comparatively cheap, especially after age "
        "coverage has already been included. The age and gender floors are subadditive in objective cost at the "
        "proportional corner."
        "</div>"
    )

    parts.append("<h2>Verified numbers</h2>")
    parts.append(df_to_html(metrics))

    parts.append("<h2>Figure: price of fairness constraints</h2>")
    parts.append(
        "<img class='figure' src='assets/"
        + html.escape(FIG_PNG.name)
        + "' alt='Fairness price curves'>"
    )
    if pdf_exists:
        parts.append(
            "<p class='small'>Vector figure: <a href='assets/"
            + html.escape(FIG_PDF.name)
            + "'>PDF</a></p>"
        )
    parts.append(
        "<p class='small'>Panels A and B show the cost of requiring coverage for children under 13. "
        "Panels C and D show the cost of requiring coverage for girls, with and without the proportional young floor. "
        "The lower incremental cost of the girls floor under the young floor indicates subadditivity.</p>"
    )

    parts.append("<h2>Policy scenario table</h2>")
    parts.append(
        "<p>The objective is the sum of out-of-fold predicted risk scores. Observed out-of-school capture is an "
        "evaluation metric only; it is not used as the optimization target.</p>"
    )
    parts.append(df_to_html(summary))

    parts.append("<h2>Method in brief</h2>")

    parts.append("<h3>1. Data provenance</h3>")
    parts.append(
        "<p>The public site uses only aggregate outputs. The raw analysis source is the Bangladesh MICS 2019 child "
        "file <span class='code'>fs.sav</span>. The sample is rural Chattogram, ages 6-17, with valid school attendance. "
        "This gives 4,891 children. The modeling-ready sample excludes 30 children because of incomplete interview/zero "
        "weight or missing core features, leaving 4,861 children.</p>"
    )

    parts.append("<h3>2. Risk model</h3>")
    parts.append(
        "<p>A logistic regression model predicts out-of-school status using complete core covariates: age, sex, "
        "wealth quintile, mother's education, ethnic minority status, disability flag, and child work. The score used "
        "for optimization is out-of-fold, produced by 5-fold grouped cross-validation grouped by PSU.</p>"
    )
    parts.append(
        "<p>The model is fitted without sample weights. Survey weights are used for evaluation. The primary child "
        "weight is provisionally treated as <span class='code'>fsweight</span>, while <span class='code'>fshweight</span> "
        "is kept as a normalized sensitivity weight.</p>"
    )

    parts.append("<h3>3. Allocation model</h3>")
    parts.append(
        r"<p>Let \(x_i = 1\) if child \(i\) is selected and \(r_i\) be the out-of-fold risk score. The baseline problem is:"
    )
    parts.append(
        r"\[ \max_x \sum_i r_i x_i \quad \text{subject to} \quad \sum_i x_i = K. \]"
    )
    parts.append(
        r"<p>Fairness versions add lower-bound coverage constraints such as \( \sum_{i: \text{age}_i \leq 12} x_i \geq y_{\min} \) and \( \sum_{i: \text{female}_i = 1} x_i \geq g_{\min} \).</p>"
    )
    parts.append(
        "<p>The integer programs were solved with SciPy's MILP interface using HiGHS. The relative MIP gap was set to "
        "zero, so the reported solutions are proven optimal for the stated models. The young-floor cases were also "
        "cross-checked against an independent exact greedy reference.</p>"
    )

    parts.append("<h2>Important corrections from earlier drafts</h2>")
    parts.append(
        "<div class='callout orange'>"
        "<ul>"
        "<li>The old Phase 2 column named <span class='code'>books_missing</span> was verified to be a nonresponse "
        "indicator for <span class='code'>PR3</span>, not a true no-books variable.</li>"
        "<li>The old Phase 2 column named <span class='code'>FSWEIGHT</span> was verified to contain raw "
        "<span class='code'>fshweight</span>, not raw <span class='code'>fsweight</span>.</li>"
        "<li>The old Phase 2 risk score was not locally reproducible, so Phase 3 rebuilt the score from raw verified data.</li>"
        "<li>The old Phase 2 claim that age and gender floors were superadditive is withdrawn. On the reproducible "
        "Phase 3 score, the interaction is subadditive at the proportional corner.</li>"
        "</ul>"
        "</div>"
    )

    parts.append("<h2>Data access and privacy</h2>")
    parts.append(
        "<p>MICS microdata contain household and child identifiers. Raw <span class='code'>.sav</span> files and "
        "derived child-level CSV files are kept local and are not published in the repository. This website publishes "
        "only aggregate tables, aggregate curves, figures, and code. No child-level records are included.</p>"
    )

    parts.append("<h2>AI use disclosure</h2>")
    parts.append(
        "<p>AI tools were used for code debugging, syntax generation, and structural suggestions. All methodological "
        "decisions, data verification, numerical checks, interpretations, and final wording are the responsibility of "
        "the author.</p>"
    )

    parts.append("<h2>Files used by this page</h2>")
    parts.append(
        "<ul>"
        "<li><a href='data/" + html.escape(SUMMARY_CSV.name) + "'>phase3_fairness_summary.csv</a></li>"
        "<li><a href='data/" + html.escape(CURVES_CSV.name) + "'>phase3_fairness_curves.csv</a></li>"
        "<li><a href='assets/" + html.escape(FIG_PNG.name) + "'>phase3_fairness_curves_publication.png</a></li>"
        + (
            "<li><a href='assets/" + html.escape(FIG_PDF.name) + "'>phase3_fairness_curves_publication.pdf</a></li>"
            if pdf_exists
            else ""
        )
        + "</ul>"
    )

    parts.append("<h2>Next public deliverables</h2>")
    parts.append(
        "<div class='grid'>"
        "<div class='card'><h3>GitHub Pages site</h3><p>Deploy this <span class='code'>docs/</span> folder as the public project page.</p></div>"
        "<div class='card'><h3>Google Colab demo</h3><p>Create a public notebook that reproduces the fairness figure from aggregate CSVs only.</p></div>"
        "<div class='card'><h3>Notion project page</h3><p>Use Notion as a portfolio page linking the repo, site, Colab, and paper draft.</p></div>"
        "<div class='card'><h3>Overleaf paper</h3><p>Convert the verified report skeleton into the research paper later.</p></div>"
        "</div>"
    )

    parts.append("<div class='footer'>Generated from verified aggregate Phase 3 outputs. Do not add child-level microdata to this site.</div>")

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")

    (DOCS / "index.html").write_text("\n".join(parts), encoding="utf-8")

    print("=" * 80)
    print("PUBLIC SITE BUILT")
    print("=" * 80)
    print(f"Docs folder: {DOCS.resolve()}")
    print(f"Index file: {(DOCS / 'index.html').resolve()}")
    print()
    print("Open it with:")
    print("Invoke-Item docs\\index.html")
    print()
    print("Before committing, run:")
    print("git status --short")
    print()
    print("If git status shows data/raw or data/processed files, stop and do not commit.")


if __name__ == "__main__":
    main()