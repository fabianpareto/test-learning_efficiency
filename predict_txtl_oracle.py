#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


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


def transform(df, model):
    prep = model["preprocessor"]
    numeric_features = model["numeric_features"]
    categorical_features = model["categorical_features"]

    blocks = []

    for col in numeric_features:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
        else:
            series = pd.Series([np.nan] * len(df))
        series = series.fillna(prep["numeric_impute"][col])
        z = (series - prep["numeric_mean"][col]) / prep["numeric_std"][col]
        blocks.append(z.to_numpy(dtype=float).reshape(-1, 1))

    for col in categorical_features:
        series = df.get(col)
        if series is None:
            series = pd.Series(["missing"] * len(df))
        else:
            series = series.map(lambda v: _normalize_categorical_value(v, col))

        cats = prep["categories"][col]
        mat = np.zeros((len(df), len(cats)), dtype=float)
        index = {c: i for i, c in enumerate(cats)}
        for row_idx, value in enumerate(series):
            if value in index:
                mat[row_idx, index[value]] = 1.0
        blocks.append(mat)

    return np.hstack(blocks)


def parse_target_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", text)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_target(df):
    est = df["estimated_productivity_mg_mL_hr"].map(parse_target_value) if "estimated_productivity_mg_mL_hr" in df.columns else pd.Series([np.nan] * len(df))
    calc = pd.to_numeric(df["productivity_calc_mg_mL_hr"], errors="coerce") if "productivity_calc_mg_mL_hr" in df.columns else pd.Series([np.nan] * len(df))
    return calc.where(calc.notna(), est)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def apply_prediction_calibration(preds, model):
    calibration = model.get("prediction_calibration")
    if not calibration:
        return preds
    if calibration.get("method") != "upper_tail_uplift":
        return preds
    factor = float(calibration.get("factor", 0.0))
    threshold = float(calibration.get("threshold", 0.0))
    return preds + factor * np.maximum(0.0, preds - threshold)


def main():
    parser = argparse.ArgumentParser(description="Predict TX-TL productivity with a trained oracle")
    parser.add_argument("--model", default="oracle_artifacts/txtl_oracle_model.json")
    parser.add_argument("--input", required=True, help="CSV with parameter columns")
    parser.add_argument("--output", default="oracle_predictions.csv")
    args = parser.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)

    df = pd.read_csv(args.input)
    X = transform(df, model)

    coeffs = np.asarray(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    preds_raw = intercept + X @ coeffs
    preds = apply_prediction_calibration(preds_raw, model)

    out = df.copy()
    out["predicted_productivity_raw_mg_mL_hr"] = preds_raw
    out["predicted_productivity_mg_mL_hr"] = preds

    # Compute and print metrics if ground truth is available
    target = build_target(df)
    if "target_productivity_mg_mL_hr" in df.columns:
        target = pd.to_numeric(df["target_productivity_mg_mL_hr"], errors="coerce").where(
            lambda s: s.notna(), target
        )
    valid = target.notna()
    if valid.any():
        y_true = target[valid].to_numpy(dtype=float)
        y_pred = preds[valid.to_numpy()]
        m = metrics(y_true, y_pred)
        out["target_productivity_mg_mL_hr"] = target
        out["abs_error"] = np.where(valid, np.abs(target - preds), np.nan)
        print(
            f"Prediction metrics ({int(valid.sum())}/{len(df)} rows with ground truth): "
            f"MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}"
        )

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {Path(args.output)}")


if __name__ == "__main__":
    main()
