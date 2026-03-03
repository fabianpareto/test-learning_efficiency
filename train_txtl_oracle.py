#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "Mg_glutamate_mM",
    "K_glutamate_mM",
    "DTT_mM",
    "NTP_multiplier",
    "PEG_8000_percent",
    "temperature_C",
    "plasmid_concentration_nM",
]

CATEGORICAL_FEATURES = [
    "energy_source",
    "extract_type",
    "chaperones",
    "reaction_mode",
]


def _normalize_categorical_value(value, feature_name):
    if pd.isna(value):
        return "missing"
    text = str(value).strip().lower()
    if text == "" or text == "nan":
        return "missing"

    if feature_name == "energy_source":
        if "maltodextrin" in text:
            return "maltodextrin"
        if "maltose" in text:
            return "maltose"
        if "3-pga" in text or "3 pga" in text:
            return "3-pga"
        if "pep" in text:
            return "pep"
        if "pyruvate" in text:
            return "pyruvate"
        return text

    if feature_name == "extract_type":
        if "kc6" in text:
            return "kc6"
        if "wild" in text or "wt" in text:
            return "wild-type"
        if "bl21" in text:
            return "bl21"
        if "a19" in text:
            return "a19"
        return text

    if feature_name == "chaperones":
        if text in ("none", "no", "null"):
            return "none"
        if "combined" in text:
            return "combined"
        if "dnak" in text:
            return "dnak/dnaj/grpe"
        if "groes" in text or "groel" in text:
            return "groes/el"
        return text

    if feature_name == "reaction_mode":
        if "semi" in text:
            return "semi-continuous"
        if "micro" in text:
            return "microfluidic"
        if "dialysis" in text or "two-stage" in text:
            return "dialysis/two-stage"
        if "batch" in text:
            return "batch"
        return text

    return text


def normalize_categorical_columns(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _normalize_categorical_value(v, col))
    return df


def parse_target_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", text)
    if match:
        lo = float(match.group(1))
        hi = float(match.group(2))
        return (lo + hi) / 2.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_target(df):
    est = df["estimated_productivity_mg_mL_hr"].map(parse_target_value)
    calc = pd.to_numeric(df["productivity_calc_mg_mL_hr"], errors="coerce")
    return calc.where(calc.notna(), est)


def fit_preprocessor(train_df):
    numeric_impute = {}
    numeric_mean = {}
    numeric_std = {}

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(train_df[col], errors="coerce")
        med = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(med)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if std == 0.0:
            std = 1.0

        numeric_impute[col] = med
        numeric_mean[col] = mean
        numeric_std[col] = std

    categories = {}
    for col in CATEGORICAL_FEATURES:
        series = train_df[col].fillna("missing").astype(str)
        categories[col] = sorted(series.unique().tolist())

    return {
        "numeric_impute": numeric_impute,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "categories": categories,
    }


def transform(df, prep):
    blocks = []

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").fillna(prep["numeric_impute"][col])
        z = (series - prep["numeric_mean"][col]) / prep["numeric_std"][col]
        blocks.append(z.to_numpy(dtype=float).reshape(-1, 1))

    for col in CATEGORICAL_FEATURES:
        series = df[col].fillna("missing").astype(str)
        cats = prep["categories"][col]
        mat = np.zeros((len(df), len(cats)), dtype=float)
        index = {c: i for i, c in enumerate(cats)}
        for row_idx, value in enumerate(series):
            if value in index:
                mat[row_idx, index[value]] = 1.0
        blocks.append(mat)

    return np.hstack(blocks)


def fit_ridge(X, y, alpha):
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    reg = np.eye(Xb.shape[1])
    reg[0, 0] = 0.0
    # Solve ridge via augmented least squares for numerical stability.
    A = np.vstack([Xb, np.sqrt(alpha) * reg])
    b = np.concatenate([y, np.zeros(Xb.shape[1])])
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coeffs[0]), coeffs[1:]


def fit_bagged_ridge(X, y, alpha, n_bags=5, seed=42):
    """Fit ridge regression with bootstrap aggregating."""
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_bags):
        boot_idx = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb = X[boot_idx], y[boot_idx]
        intercept, coeffs = fit_ridge(Xb, yb, alpha=alpha)
        models.append((intercept, coeffs))
    return models


def predict_bagged(X, models):
    """Average predictions from bagged models."""
    preds = np.zeros(len(X))
    for intercept, coeffs in models:
        preds += predict(X, intercept, coeffs)
    return preds / len(models)


def select_alpha_loo(df, target_col, alphas, use_bagging=False, n_bags=5):
    n = len(df)
    rows = []
    for alpha in alphas:
        errors = []
        for i in range(n):
            train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
            fold_train = df.iloc[train_idx].reset_index(drop=True)
            fold_val = df.iloc[[i]].reset_index(drop=True)

            prep = fit_preprocessor(fold_train)
            X_train = transform(fold_train, prep)
            y_train = fold_train[target_col].to_numpy(dtype=float)
            X_val = transform(fold_val, prep)
            y_val = fold_val[target_col].to_numpy(dtype=float)

            if use_bagging:
                models = fit_bagged_ridge(X_train, y_train, alpha=alpha, n_bags=n_bags)
                pred = predict_bagged(X_val, models)
            else:
                intercept, coeffs = fit_ridge(X_train, y_train, alpha=alpha)
                pred = predict(X_val, intercept, coeffs)
            errors.append(float(np.abs(y_val[0] - pred[0])))
        loo_mae = float(np.mean(errors))
        rows.append({"alpha": float(alpha), "loo_mae": loo_mae})

    rows.sort(key=lambda r: r["loo_mae"])
    best_alpha = rows[0]["alpha"]
    return best_alpha, rows


def predict(X, intercept, coeffs):
    return intercept + X @ coeffs


def apply_prediction_calibration(preds, calibration):
    if not calibration:
        return preds
    method = calibration.get("method")
    if method != "upper_tail_uplift":
        return preds
    factor = float(calibration.get("factor", 0.0))
    threshold = float(calibration.get("threshold", 0.0))
    return preds + factor * np.maximum(0.0, preds - threshold)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser(description="Train a TX-TL productivity oracle")
    parser.add_argument("--data", default="txtl_optimization_experiments_verified_expanded.csv")
    parser.add_argument("--outdir", default="oracle_artifacts")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--alpha-grid", default="0.1,0.2,0.3,0.5,1,3,10,30,100")
    parser.add_argument(
        "--auto-alpha",
        action="store_true",
        help="Select alpha from --alpha-grid via LOO cross-validation.",
    )
    parser.add_argument("--bagging", action="store_true", help="Use bootstrap aggregating (bagging)")
    parser.add_argument("--n-bags", type=int, default=5, help="Number of bootstrap samples for bagging")
    parser.add_argument("--uplift-factor", type=float, default=0.3)
    parser.add_argument("--uplift-quantile", type=float, default=0.75)
    parser.add_argument("--no-uplift", action="store_true")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Use only rows where parameters_verified starts with 'Yes'.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.copy()
    df = normalize_categorical_columns(df)
    df["target_productivity_mg_mL_hr"] = build_target(df)

    if args.verified_only:
        verified = df["parameters_verified"].fillna("").astype(str).str.startswith("Yes")
        df = df.loc[verified].copy()

    df = df.dropna(subset=["target_productivity_mg_mL_hr"]).reset_index(drop=True)

    n = len(df)
    if n < 3:
        raise ValueError("Not enough rows for leave-one-out validation.")

    y_all = df["target_productivity_mg_mL_hr"].to_numpy(dtype=float)

    # Alpha selection via LOO
    alpha_candidates = [float(x) for x in args.alpha_grid.split(",") if str(x).strip()]
    cv_rows = []
    if args.auto_alpha:
        best_alpha, cv_rows = select_alpha_loo(
            df, "target_productivity_mg_mL_hr", alpha_candidates,
            use_bagging=args.bagging, n_bags=args.n_bags
        )
    else:
        best_alpha = float(args.alpha)

    # Leave-one-out cross-validation
    loo_preds_raw = np.zeros(n)
    for i in range(n):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[[i]].reset_index(drop=True)

        prep = fit_preprocessor(fold_train)
        X_train = transform(fold_train, prep)
        y_train = fold_train["target_productivity_mg_mL_hr"].to_numpy(dtype=float)
        X_val = transform(fold_val, prep)

        if args.bagging:
            models = fit_bagged_ridge(X_train, y_train, alpha=best_alpha, n_bags=args.n_bags)
            loo_preds_raw[i] = predict_bagged(X_val, models)[0]
        else:
            intercept, coeffs = fit_ridge(X_train, y_train, alpha=best_alpha)
            loo_preds_raw[i] = predict(X_val, intercept, coeffs)[0]

    # Train final model on all data
    prep = fit_preprocessor(df)
    X_all = transform(df, prep)
    if args.bagging:
        models = fit_bagged_ridge(X_all, y_all, alpha=best_alpha, n_bags=args.n_bags)
        # For bagged model, save averaged coefficients
        intercept = float(np.mean([m[0] for m in models]))
        coeffs = np.mean([m[1] for m in models], axis=0)
    else:
        intercept, coeffs = fit_ridge(X_all, y_all, alpha=best_alpha)

    # Calibration (based on final model's predictions)
    yhat_all_raw = predict(X_all, intercept, coeffs)
    calibration = None
    if not args.no_uplift:
        threshold = float(np.quantile(yhat_all_raw, args.uplift_quantile))
        calibration = {
            "method": "upper_tail_uplift",
            "factor": float(args.uplift_factor),
            "quantile": float(args.uplift_quantile),
            "threshold": threshold,
        }

    # Apply calibration to LOO predictions for scoring
    loo_preds = apply_prediction_calibration(loo_preds_raw, calibration)
    loo_metrics = metrics(y_all, loo_preds)

    yhat_all = apply_prediction_calibration(yhat_all_raw, calibration)
    train_metrics = metrics(y_all, yhat_all)

    # Save artifacts
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_names = []
    feature_names.extend(NUMERIC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        feature_names.extend([f"{col}=={cat}" for cat in prep["categories"][col]])

    model_payload = {
        "model_type": "bagged_ridge_regression" if args.bagging else "ridge_regression",
        "target": "productivity_mg_mL_hr",
        "alpha": best_alpha,
        "alpha_input": args.alpha,
        "auto_alpha": args.auto_alpha,
        "cv_alpha_results": cv_rows,
        "bagging": args.bagging,
        "n_bags": args.n_bags if args.bagging else None,
        "trained_rows": n,
        "verified_only": args.verified_only,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessor": prep,
        "feature_names": feature_names,
        "intercept": intercept,
        "coefficients": coeffs.tolist(),
        "prediction_calibration": calibration,
        "categorical_normalization": "rule_v1",
        "metrics_train": train_metrics,
        "metrics_loo": loo_metrics,
    }

    with (outdir / "txtl_oracle_model.json").open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    loo_predictions = df[["experiment_id", "target_productivity_mg_mL_hr"]].copy()
    loo_predictions["predicted_productivity_raw_mg_mL_hr"] = loo_preds_raw
    loo_predictions["predicted_productivity_mg_mL_hr"] = loo_preds
    loo_predictions["abs_error"] = np.abs(
        loo_predictions["target_productivity_mg_mL_hr"]
        - loo_predictions["predicted_productivity_mg_mL_hr"]
    )
    loo_predictions.to_csv(outdir / "loo_predictions.csv", index=False)

    print("Oracle trained with leave-one-out validation.")
    print(f"Rows: {n}")
    print(f"Model: {'Bagged ridge (n_bags=' + str(args.n_bags) + ')' if args.bagging else 'Ridge regression'}")
    print(f"Alpha: {best_alpha:.4g} (auto_alpha={args.auto_alpha})")
    print(
        "Prediction calibration: "
        f"{'upper_tail_uplift' if calibration else 'none'}"
    )
    if calibration:
        print(
            f"  factor={calibration['factor']}, "
            f"quantile={calibration['quantile']}, "
            f"threshold={calibration['threshold']:.4f}"
        )
    print(
        "LOO metrics:   "
        f"MAE={loo_metrics['mae']:.4f}, "
        f"RMSE={loo_metrics['rmse']:.4f}, "
        f"R2={loo_metrics['r2']:.4f}"
    )
    print(
        "Train metrics: "
        f"MAE={train_metrics['mae']:.4f}, "
        f"RMSE={train_metrics['rmse']:.4f}, "
        f"R2={train_metrics['r2']:.4f}"
    )
    print(f"LOO predictions saved to: {outdir / 'loo_predictions.csv'}")
    print(f"Model saved to: {outdir / 'txtl_oracle_model.json'}")


if __name__ == "__main__":
    main()
