from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


INPUT = Path("data/processed/chattogram_rural_phase3_clean.csv")
OUT_SCORED = Path("data/processed/chattogram_rural_phase3_scored.csv")
REPORT = Path("reports/phase3_risk_scoring.txt")

FEATURES = [
    "age",
    "female",
    "wealth_q",
    "mom_edu",
    "eth_minor",
    "disabled",
    "worked",
]

TARGET = "Out_of_School"
GROUP = "PSU"

WEIGHT_COLS = [
    "child_weight_fsweight",
    "normalized_weight_fshweight",
]

RANDOM_STATE = 2026
N_SPLITS = 5
TOP_FRACTION = 0.15

lines = []


def p(msg=""):
    text = str(msg)
    print(text)
    lines.append(text)


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{label} missing required columns: {missing}")


def metric_table(y, risk, weights_dict):
    rows = []

    rows.append(
        (
            "unweighted",
            len(y),
            roc_auc_score(y, risk),
            average_precision_score(y, risk),
            brier_score_loss(y, risk),
        )
    )

    for name, w in weights_dict.items():
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(w) & (w > 0)

        if int(mask.sum()) == 0:
            auc = np.nan
            ap = np.nan
            brier = np.nan
            n_used = 0
        else:
            yy = y[mask]
            rr = risk[mask]
            ww = w[mask]

            auc = roc_auc_score(yy, rr, sample_weight=ww)
            ap = average_precision_score(yy, rr, sample_weight=ww)
            brier = brier_score_loss(yy, rr, sample_weight=ww)
            n_used = int(mask.sum())

        rows.append((name, n_used, auc, ap, brier))

    return pd.DataFrame(
        rows,
        columns=[
            "evaluation",
            "n_used",
            "roc_auc",
            "average_precision",
            "brier_score",
        ],
    )


def top_summary(y, risk, weight, frac=TOP_FRACTION):
    y = np.asarray(y, dtype=int)
    risk = np.asarray(risk, dtype=float)
    weight = np.asarray(weight, dtype=float)

    n = len(y)
    n_top = int(np.ceil(frac * n))

    order = np.argsort(-risk)
    top_idx = order[:n_top]

    selected = np.zeros(n, dtype=bool)
    selected[top_idx] = True

    total_oos = int(y.sum())
    captured_oos = int(y[selected].sum())

    capture_rate = captured_oos / total_oos if total_oos > 0 else np.nan

    positive = y == 1
    weight_positive_total = float(weight[positive].sum())
    weight_captured_positive = float(weight[selected & positive].sum())

    weighted_capture_rate = (
        weight_captured_positive / weight_positive_total
        if weight_positive_total > 0
        else np.nan
    )

    return {
        "n_selected": n_top,
        "selected_share": n_top / n if n > 0 else np.nan,
        "total_out_of_school": total_oos,
        "captured_out_of_school": captured_oos,
        "unweighted_capture_rate": capture_rate,
        "weighted_capture_rate_child_weight": weighted_capture_rate,
        "mean_risk_selected": float(risk[selected].mean()),
        "mean_risk_not_selected": float(risk[~selected].mean()),
    }


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCORED.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise SystemExit(f"Clean input not found: {INPUT.resolve()}")

    p("=" * 80)
    p("PHASE 3 RISK SCORING")
    p("=" * 80)
    p(f"INPUT: {INPUT.resolve()}")
    p(f"SCORED OUTPUT: {OUT_SCORED.resolve()}")
    p(f"REPORT: {REPORT.resolve()}")

    df_all = pd.read_csv(INPUT, low_memory=False)

    p("")
    p("INPUT CHECK")
    p("-" * 80)
    p(f"Clean input shape: {df_all.shape}")

    require_columns(
        df_all,
        FEATURES + [TARGET, GROUP, "modeling_ready"] + WEIGHT_COLS,
        "Clean input",
    )

    if "modeling_ready" not in df_all.columns:
        raise SystemExit("modeling_ready column missing.")

    df = df_all.loc[df_all["modeling_ready"] == 1].copy().reset_index(drop=True)

    p(f"Total clean rows: {len(df_all)}")
    p(f"Modeling-ready rows: {len(df)}")
    p(f"Excluded rows: {len(df_all) - len(df)}")

    if len(df) == 0:
        raise SystemExit("No modeling-ready rows.")

    p("")
    p("MODELING SAMPLE CHECKS")
    p("-" * 80)

    check_cols = FEATURES + [TARGET, GROUP] + WEIGHT_COLS
    missing = df[check_cols].isna().sum()
    p("Missing values in modeling sample:")
    p(missing.to_string())

    if int(missing.sum()) > 0:
        p("")
        p("VERDICT: FAIL - modeling sample has missing required fields.")
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        p(f"REPORT SAVED: {REPORT.resolve()}")
        return

    y = df[TARGET].astype(int).to_numpy()
    groups = df[GROUP].astype(int).to_numpy()
    X = df[FEATURES].astype(float).to_numpy()

    p("")
    p(f"Target counts:")
    p(pd.Series(y).value_counts(dropna=False).sort_index().to_string())
    p(f"Out-of-school rate: {y.mean():.6f}")

    p("")
    p(f"Unique PSU groups: {len(np.unique(groups))}")
    p("Group size summary:")
    p(
        pd.Series(groups)
        .value_counts()
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        .to_string()
    )

    p("")
    p("MODEL SETUP")
    p("-" * 80)
    p("Model: logistic regression")
    p("Preprocessing: standard scaler inside pipeline")
    p("Validation: out-of-fold predicted probabilities")
    p(f"Folds: {N_SPLITS}")
    p(f"Group variable: {GROUP}")
    p(f"Random state: {RANDOM_STATE}")
    p("Training weights: none")
    p("Evaluation weights: child_weight_fsweight and normalized_weight_fshweight")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    cv_name = "StratifiedGroupKFold"

    try:
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        proba = cross_val_predict(
            pipe,
            X,
            y,
            cv=cv,
            groups=groups,
            method="predict_proba",
            n_jobs=None,
        )

    except Exception as exc:
        p("")
        p(f"StratifiedGroupKFold failed: {type(exc).__name__}: {exc}")
        p("Falling back to GroupKFold without stratification.")

        from sklearn.model_selection import GroupKFold

        cv_name = "GroupKFold fallback"
        cv = GroupKFold(n_splits=N_SPLITS)

        proba = cross_val_predict(
            pipe,
            X,
            y,
            cv=cv,
            groups=groups,
            method="predict_proba",
            n_jobs=None,
        )

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise SystemExit(f"Unexpected predict_proba output shape: {proba.shape}")

    risk = proba[:, 1]

    p("")
    p(f"Cross-validation used: {cv_name}")
    p(f"Out-of-fold risk score shape: {risk.shape}")
    p(f"Finite risk scores: {int(np.isfinite(risk).sum())}/{len(risk)}")
    p(f"Risk score min: {risk.min():.6f}")
    p(f"Risk score max: {risk.max():.6f}")
    p(f"Risk score mean: {risk.mean():.6f}")

    p("")
    p("RISK SCORE DISTRIBUTION")
    p("-" * 80)
    p(
        pd.Series(risk)
        .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        .to_string()
    )

    p("")
    p("EVALUATION METRICS")
    p("-" * 80)

    weights_dict = {
        name: df[name].astype(float).to_numpy()
        for name in WEIGHT_COLS
    }

    metrics = metric_table(y, risk, weights_dict)
    p(metrics.to_string(index=False))

    p("")
    p("TOP 15% BY RISK SCORE")
    p("-" * 80)

    child_w = df["child_weight_fsweight"].astype(float).to_numpy()
    top = top_summary(y, risk, child_w)

    for key, value in top.items():
        if isinstance(value, float):
            p(f"{key}: {value:.6f}")
        else:
            p(f"{key}: {value}")

    p("")
    p("FINAL FIT COEFFICIENTS FOR INTERPRETATION")
    p("-" * 80)
    p("These coefficients come from a final fit on all modeling-ready rows.")
    p("They are for interpretation only. The saved risk_score_oof is out-of-fold.")

    final_pipe = pipe.fit(X, y)
    coef = pd.Series(
        final_pipe.named_steps["logreg"].coef_[0],
        index=FEATURES,
    )

    coef_df = pd.DataFrame(
        {
            "feature": coef.index,
            "standardized_log_odds": coef.values,
            "odds_ratio_per_one_sd": np.exp(coef.values),
        }
    )

    p(coef_df.to_string(index=False))

    p("")
    p("SAVE SCORED FILE")
    p("-" * 80)

    scored = df.copy()
    scored["risk_score_oof"] = risk
    scored["risk_model"] = (
        "logistic_regression_standard_scaler_"
        f"{cv_name.lower()}_{N_SPLITS}fold_unweighted_fit"
    )

    scored.to_csv(OUT_SCORED, index=False)

    p(f"Saved scored file: {OUT_SCORED.resolve()}")
    p(f"Scored shape: {scored.shape}")
    p("Do not commit this file. It contains child-level survey data and predicted scores.")

    p("")
    p("VERDICT")
    p("-" * 80)

    if int(np.isfinite(risk).sum()) == len(risk):
        p("RISK SCORING STATUS: PASS")
        p("All modeling-ready rows received a finite out-of-fold risk score.")
    else:
        p("RISK SCORING STATUS: FAIL")
        p("Some risk scores are not finite.")

    p("")
    p("IMPORTANT LIMITATIONS")
    p("-" * 80)
    p("This is a first reproducible risk model, not the final optimized model.")
    p("It uses only complete core features: age, female, wealth_q, mom_edu, eth_minor, disabled, worked.")
    p("It does not use no_books because that variable has 1,621 missing values.")
    p("It does not use books_missing_data as a substantive feature because it is a nonresponse indicator.")
    p("The model is fitted without sample weights.")
    p("Survey weights are used only to evaluate population-weighted performance.")
    p("The primary child weight is still provisional until MICS codebook confirmation.")

    p("")
    p("NEXT STEP AFTER THIS REPORT")
    p("-" * 80)
    p("1. Send reports/phase3_risk_scoring.txt.")
    p("2. If PASS, rebuild the optimization input using risk_score_oof, original IDs, PSU, and child_weight_fsweight.")
    p("3. Run the budgeted selection MILP using the verified score.")
    p("4. Run fairness constraints and sensitivity checks using both weight columns.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    p("")
    p(f"REPORT SAVED: {REPORT.resolve()}")


if __name__ == "__main__":
    main()