# Fairness-Constrained Allocation of Educational Support in Rural Chattogram

A reproducible decision-support study that estimates out-of-school risk from verified
survey microdata and then prices the trade-off between predictive reach and explicit
fairness coverage constraints, under a fixed support budget.

The public site is live at
<https://rafi-math.github.io/edu-support-optimization-bd/> and is built from aggregate
outputs only. No child-level record, household identifier, or raw survey file is published
in this repository.

## What the project does

The pipeline has three computational stages, all run on the same verified sample:

1. **Risk scoring.** A logistic regression predicts out-of-school status from complete core
   covariates (age, sex, wealth quintile, mother's education, ethnic-minority status,
   disability flag, child work). The score used downstream is out-of-fold, produced by
   5-fold cross-validation grouped by survey cluster (PSU), so no child is scored by a model
   fitted on that child. Logistic regression was selected on out-of-fold ROC-AUC over
   random-forest and MLP alternatives.
2. **Budgeted allocation.** With a fixed number of support packages, select the children that
   maximize total predicted risk, then add lower-bound coverage floors for children under 13
   ("young") and for girls, and measure what each floor costs in reach.
3. **Fairness grid and publication figure.** Sweep the young and girls floors on a 2-D grid,
   solve each cell to proven optimality, and report the price of fairness and the interaction
   between the two floors.

## Data and scope

- Source: Bangladesh MICS 2019, children's file `fs.sav`.
- Area: rural Chattogram Division.
- Ages: 6–17.
- Target: out-of-school status in the current school year.
- Raw rural cohort: 4,891 children. Modeling-ready sample: 4,861 children (30 excluded for
  incomplete interview / zero weight or missing core features).

The MICS 2019 children's file does **not** contain household coordinates, school
identifiers, upazila identifiers, travel distances, center construction costs, or center
capacities. This project therefore does not claim to optimize physical support-center
locations from observed MICS infrastructure data. Any distance, cost, capacity, or budget
value used in the optimization is an analyst- or policy-chosen scenario parameter, not an
observed MICS variable.

## Weight handling

The survey weight is used for evaluation, not for fitting the risk model. An early processed
export labelled its weight column `FSWEIGHT`; that column was later verified to hold raw
`fshweight`, not raw `fsweight`. The rebuilt optimization input uses the correct `fsweight`
as the primary child weight and retains `fshweight` as a normalized sensitivity weight, so
weighted results can be checked under both definitions.

## Verified numbers (anchors)

These are the values the public tables and figure are built from; each is reproducible from
the committed aggregate outputs.

| Quantity | Value |
| --- | --- |
| Modeling-ready sample | 4,861 children |
| Selection budget K | 730 (about 15%) |
| Observed out-of-school children | 641 |
| Unweighted out-of-school rate | 13.19% |
| Provisional child-weighted rate | 14.52% |
| Out-of-fold ROC-AUC (unweighted / weighted) | 0.8398 / 0.8520 |
| Raw-risk observed OOS captured | 332 of 641 (51.8%) |
| Raw-risk selected share aged < 13 | 0.14% (1 child) |
| Raw-risk selected share girls | 29.04% |
| Young + girls interaction, unweighted objective | −5.740 (subadditive) |

Headline result: raw risk ranking reaches many out-of-school adolescents but almost excludes
younger children, so age coverage is expensive; gender parity is comparatively cheap, and the
two floors are subadditive in cost at the proportional corner (imposing one makes the other
cheaper).

## Reproducibility

The integer programs were solved with SciPy's MILP interface (HiGHS) at relative MIP gap
zero, so the reported allocations are proven optimal for the stated models. The young-floor
cases were cross-checked against an independent exact greedy reference. The public site is
generated from aggregate CSVs and figures only.

## Data-audit notes (kept for provenance)

The following were caught and corrected during the work, and are recorded here rather than
hidden, because the correction trail is part of the result:

- A column originally named `books_missing` was verified to be a non-response indicator for
  `PR3`, not a true no-books variable.
- The early `FSWEIGHT` column was verified to contain raw `fshweight`, not raw `fsweight`
  (see Weight handling above).
- The first allocation score was not locally reproducible, so the score was rebuilt from raw
  verified data before any fairness result was published.
- An early claim that the age and gender floors were superadditive is withdrawn; on the
  reproducible score the interaction is subadditive at the proportional corner.

## Public site and maintenance

The live page is served from `docs/` by GitHub Pages. `docs/` is now maintained directly
(the HTML, CSS, and client-side scripts live there); the script `src/build_public_site.py`
produced the initial version and is kept for reference. Editing `docs/` directly is
intentional and does not affect the reproducibility of the science, which rests on the
committed scripts and aggregate outputs under `src/`, `reports/`, and `results/`.

## Privacy

MICS microdata contain household and child identifiers. Raw `.sav` files and derived
child-level CSV files are kept local and are never committed. This repository publishes only
aggregate tables, aggregate curves, figures, code, and documentation. The repository
`.gitignore` enforces this; the public tree contains no path under `data/` and no `.sav`
file.

## Repository layout (representative)

```text
.
├── docs/                      # public site (served by GitHub Pages)
│   ├── index.html
│   ├── phase3_reproduce_demo.ipynb  # public Colab demo (aggregate only)
│   ├── styles.css
│   ├── app.js                 # client-side reveal + reading-zoom, no backend
│   ├── ai-use-log.html        # styled AI-use record
│   ├── AI_USE_LOG.md
│   ├── assets/                # figures (PNG/PDF)
│   ├── data/                  # aggregate CSVs used by the page
│   └── .nojekyll
├── src/                       # analysis and build scripts
│   ├── build_phase3_risk_score.py
│   ├── build_phase3_optimization_input.py
│   ├── run_phase3_fairness_grid.py
│   ├── make_phase3_fairness_figure_publication.py
│   ├── build_phase2_policy_table_from_grid.py
│   ├── build_public_site.py
│   └── ...
├── reports/                   # validation and grid audit logs
├── results/                   # aggregate result tables (no child-level rows)
├── .gitignore
├── requirements.txt
└── README.md
