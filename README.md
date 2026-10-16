# Fairness-Constrained Allocation of Educational Support in Rural Chattogram

A reproducible risk-scoring and optimization project by **Md. Ali Arman Rafi**, Department of Mathematics, University of Chittagong, Bangladesh.

Repository: https://github.com/rafi-math/edu-support-optimization-bd  
Public site source: `docs/`  
Data source: Bangladesh MICS 2019 child file, analyzed locally. Raw microdata are not redistributed in this repository.

---

## Project summary

This project builds a decision-support workflow for allocating limited educational support among rural children aged 6–17 in Chattogram, Bangladesh.

The workflow has three parts:

1. **Verified data provenance** from the Bangladesh MICS 2019 child questionnaire.
2. **Reproducible out-of-fold risk scoring** for out-of-school status.
3. **Fairness-constrained allocation** using an exact integer optimization model.

The main empirical finding is that a raw risk-ranking rule predicts out-of-school status well but concentrates support almost entirely among older adolescents. Adding explicit age-coverage constraints is expensive in terms of predicted reach. Adding gender-coverage constraints is comparatively cheap, especially after age coverage has already been included.

---

## Verified headline numbers

| Item | Value |
|---|---:|
| Analysis sample | Rural Chattogram, ages 6–17, valid school-attendance response: 4,891 children |
| Modeling-ready sample | 4,861 children |
| Selection budget | K = 730 children, about 15% of the modeling-ready sample |
| Observed out-of-school children | 641 |
| Unweighted out-of-school rate | 13.19% |
| Provisional child-weighted out-of-school rate | 14.52% |
| Risk model | Logistic regression with out-of-fold predictions |
| Cross-validation | 5-fold grouped CV by survey PSU |
| Unweighted ROC-AUC | 0.8398 |
| Provisional child-weighted ROC-AUC | 0.8520 |
| Raw top-K observed out-of-school capture | 332 of 641, or 51.8% |
| Raw top-K selected children aged under 13 | 1 child, 0.14% |
| Raw top-K selected girls | 29.0% |
| Raw top-K selected poorest quintile | 41.9% |
| Proportional young-coverage floor | 423 children aged under 13 |
| Cost of proportional young floor | 103.64 objective units, about 31.8% of the raw baseline |
| Girls-parity floor | 367 girls |
| Cost of girls parity alone | 12.91 objective units, about 4.0% of the raw baseline |
| Cost of girls parity on top of proportional young floor | 7.17 objective units |
| Age × gender interaction at proportional corner | -5.74 objective units, subadditive |

The objective is the sum of out-of-fold predicted out-of-school probabilities. Observed out-of-school capture is an evaluation metric, not the optimization target.

---

## Main result

The raw risk ranking is predictive but age-skewed:

- it captures 332 of 641 observed out-of-school children;
- but only 1 selected child is under age 13;
- girls are under-selected relative to both the population share and the observed out-of-school pool;
- poverty targeting is already strong, with 41.9% of selected children in the poorest quintile.

The fairness analysis prices two coverage constraints:

1. **Young-coverage floor:** minimum number of selected children aged under 13.
2. **Girls-coverage floor:** minimum number of selected girls.

The young-coverage floor is the expensive constraint. The girls-coverage floor is much cheaper, especially once the young-coverage floor is already imposed.

At the proportional corner:

```text
cost of young floor alone       = 103.64
cost of girls floor alone       =  12.91
sum of separate costs           = 116.55
cost of both floors together    = 110.81
interaction                     =  -5.74
```

The negative interaction means the two floors are **subadditive**: imposing both together costs less than the sum of imposing each separately.

This replaces the older Phase 2 claim of superadditivity. The Phase 2 result was based on a risk score that was not locally reproducible and is superseded by the Phase 3 pipeline in this repository.

---

## Method

### Data and sample

The analysis uses the Bangladesh MICS 2019 child file, `fs.sav`, locally.

The sample is restricted to:

```text
HH6 = 2       rural residence
HH7 = 20      Chattogram division
6 <= CB3 <= 17 ages 6–17
CB7 in {1,2}  valid school-attendance response
```

This gives 4,891 children.

The modeling-ready sample excludes 30 children:

- 28 children have `FS17 = 4`, indicating an incomplete interview and zero child weight;
- 2 children have a missing work indicator.

The final modeling-ready sample contains 4,861 children.

### Outcome

The outcome is out-of-school status:

```text
Out_of_School = 1 if CB7 = 2
Out_of_School = 0 if CB7 = 1
```

### Risk model

The risk score is the predicted probability that a child is out of school.

The model is a logistic regression with standardized covariates:

- age;
- female;
- wealth quintile;
- mother’s education level;
- ethnic minority status;
- functional difficulty;
- child worked in the past week.

The score used for optimization is an **out-of-fold prediction** from 5-fold grouped cross-validation, grouped by survey PSU. This avoids using in-sample fitted probabilities as if they were unseen predictions.

The model is fitted without sample weights. Survey weights are used for evaluation.

### Survey weights

Two raw weight variables were inspected:

- `fsweight`;
- `fshweight`.

The earlier Phase 2 column named `FSWEIGHT` was verified to contain raw `fshweight`, not raw `fsweight`. Phase 3 therefore keeps both:

- `child_weight_fsweight`, provisionally treated as the child-level population weight;
- `normalized_weight_fshweight`, used as a household-normalized sensitivity weight.

The official MICS codebook confirmation for the primary child weight is still open. Weighted results are therefore reported as provisional. Unweighted results do not depend on this unresolved weight question.

### Allocation model

Let:

```text
x_i = 1 if child i is selected
x_i = 0 otherwise
r_i = out-of-fold risk score for child i
```

The baseline problem is:

```text
maximize sum_i r_i x_i
subject to sum_i x_i = K
x_i binary
```

Fairness versions add coverage floors:

```text
sum_{i: age_i <= 12} x_i >= y_min
sum_{i: female_i = 1} x_i >= g_min
```

The integer programs were solved using SciPy’s MILP interface with HiGHS. The relative MIP gap was set to zero, so the reported solutions are proven optimal for the stated models. The single young-floor cases were also cross-checked against an independent exact greedy reference.

---

## Important corrections from earlier drafts

The current public results use the Phase 3 reproducible pipeline.

The following older Phase 2 items are superseded:

1. The old Phase 2 `risk_score` column was not locally reproducible.
2. The old Phase 2 column named `FSWEIGHT` actually contained raw `fshweight`.
3. The old Phase 2 column named `books_missing` was a nonresponse indicator for `PR3`, not a true “no books at home” variable.
4. The old Phase 2 claim that age and gender floors were superadditive is withdrawn. On the reproducible Phase 3 score, the interaction is subadditive at the proportional corner.

---

## Data privacy and repository policy

Raw MICS `.sav` files and derived child-level CSV files are kept local and are not committed to this repository.

The `.gitignore` file blocks:

```text
/data/
*.sav
*.dta
*.parquet
```

and related local data patterns.

The public repository contains only:

- source code;
- aggregate reports;
- aggregate result tables;
- figures;
- the public site under `docs/`.

No child-level microdata are published.

Users who need the raw survey data should obtain it from the official MICS source and follow the applicable data-access terms.

---

## Repository layout

```text
docs/                 Public project site, aggregate only
reports/              Aggregate audit and analysis logs
results/              Aggregate tables and figures
src/                  Reproducible Python scripts
data/                 Local only, ignored by Git
```

Important public files:

```text
docs/index.html
docs/data/phase3_fairness_summary.csv
docs/data/phase3_fairness_curves.csv
docs/assets/phase3_fairness_curves_publication.png
results/figures/phase3_fairness_curves_publication.pdf
reports/phase3_fairness_summary.txt
reports/phase3_objective_prices.txt
reports/phase3_girls_curve_fine.txt
```

---

## Reproducing the public site

The public site can be rebuilt from committed aggregate files:

```powershell
python src/build_public_site.py
```

Then open:

```powershell
Invoke-Item docs/index.html
```

The fairness figure can be rebuilt from the aggregate curve CSV:

```powershell
python src/make_phase3_fairness_figure_publication.py
```

The full raw-data pipeline requires the local MICS file `data/raw/fs.sav`, which is not published in this repository.

---

## Full pipeline order, using local raw data

If you have the raw MICS child file locally at:

```text
data/raw/fs.sav
```

then the Phase 3 pipeline is:

```powershell
python src/build_phase3_clean_input.py
python src/build_phase3_risk_score.py
python src/build_phase3_optimization_input.py
python src/run_phase3_greedy_young_floor.py
python src/run_phase3_fairness_grid.py
python src/run_phase3_girls_curve_fine.py
python src/summarize_phase3_fairness.py
python src/make_phase3_fairness_figure_publication.py
python src/build_public_site.py
```

Required Python packages include:

```text
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
pyreadstat
```

---

## Limitations

1. The analysis is limited to rural Chattogram children aged 6–17. It is not a national estimate.
2. The risk model uses complete core covariates only. The book-count variable was not used as a substantive predictor because the true no-books indicator is missing for many children.
3. The primary child weight is provisionally treated as `fsweight`, but official MICS codebook confirmation is still pending. Weighted results are sensitivity results.
4. The current budget is a count of service slots, not a monetary budget. Real costs, travel time, facility capacity, and implementation constraints are not yet modeled.
5. Observed out-of-school capture is a survey-based evaluation metric. It does not measure the causal effect of an educational support program.
6. The fairness floors are policy constraints. The project prices the trade-offs; it does not claim that one fairness level is uniquely correct.

---

## AI use disclosure

AI tools were used for code debugging, syntax generation, and structural suggestions. All methodological decisions, data verification, numerical checks, interpretations, and final wording are the responsibility of the author.
