import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = "txtl_optimization_experiments_verified_expanded.csv"
ORACLE_MODEL_PATH = "oracle_artifacts/txtl_oracle_model.json"

PARAM_TO_CSV_COL = {
    "Mg_glutamate_mM": "Mg_glutamate_mM",
    "K_glutamate_mM": "K_glutamate_mM",
    "DTT_mM": "DTT_mM",
    "NTP_multiplier": "NTP_multiplier",
    "energy_source": "energy_source",
    "PEG_8000_percent": "PEG_8000_percent",
    "extract_type": "extract_type",
    "temperature_C": "temperature_C",
    "chaperones": "chaperones",
    "plasmid_concentration_nM": "plasmid_concentration_nM",
    "reaction_mode": "reaction_mode",
}

_df_cache = None
_oracle_cache = None


def load_experiments(path=DEFAULT_DATA_PATH):
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(path)
    return _df_cache


def parse_productivity_range(prod_str):
    if pd.isna(prod_str):
        return None
    prod_str = str(prod_str).strip()
    if "-" in prod_str:
        parts = prod_str.split("-")
        try:
            low, high = float(parts[0]), float(parts[1])
            return (low + high) / 2
        except ValueError:
            return None
    try:
        return float(prod_str)
    except ValueError:
        return None


def _load_oracle_model(path=ORACLE_MODEL_PATH):
    global _oracle_cache
    if _oracle_cache is not None:
        return _oracle_cache
    model_path = Path(path)
    if not model_path.exists():
        _oracle_cache = None
        return None
    with model_path.open("r", encoding="utf-8") as f:
        _oracle_cache = json.load(f)
    return _oracle_cache


def _oracle_transform_row(hypothesis, model):
    prep = model["preprocessor"]
    numeric_features = model["numeric_features"]
    categorical_features = model["categorical_features"]

    x = []
    for col in numeric_features:
        value = hypothesis.get(col)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = prep["numeric_impute"][col]
        z = (value - prep["numeric_mean"][col]) / prep["numeric_std"][col]
        x.append(float(z))

    for col in categorical_features:
        value = hypothesis.get(col)
        if value is None:
            value = "missing"
        value = str(value)
        for cat in prep["categories"][col]:
            x.append(1.0 if value == cat else 0.0)

    return np.array(x, dtype=float)


def oracle_predict_hypothesis(hypothesis, model=None):
    if model is None:
        model = _load_oracle_model()
    if model is None:
        return None

    x = _oracle_transform_row(hypothesis, model)
    coeffs = np.array(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    pred_raw = intercept + float(np.dot(x, coeffs))
    calibration = model.get("prediction_calibration")
    pred = pred_raw
    if calibration and calibration.get("method") == "upper_tail_uplift":
        factor = float(calibration.get("factor", 0.0))
        threshold = float(calibration.get("threshold", 0.0))
        pred = pred_raw + factor * max(0.0, pred_raw - threshold)

    return {
        "available": True,
        "predicted_productivity_raw_mg_mL_hr": pred_raw,
        "predicted_productivity_mg_mL_hr": pred,
        "model_type": model.get("model_type", "oracle"),
        "target": model.get("target", "productivity_mg_mL_hr"),
        "verified_only": model.get("verified_only", True),
    }


def _match_param_value(hyp_value, csv_value, param_name):
    if pd.isna(csv_value) or csv_value == "":
        return False

    numeric_tolerance = {
        "Mg_glutamate_mM": 2.0,
        "K_glutamate_mM": 20.0,
        "DTT_mM": 1.0,
        "NTP_multiplier": 0.5,
        "PEG_8000_percent": 1.0,
        "temperature_C": 3.0,
        "plasmid_concentration_nM": 5.0,
    }

    if param_name in numeric_tolerance:
        try:
            return abs(float(hyp_value) - float(csv_value)) <= numeric_tolerance[param_name]
        except (ValueError, TypeError):
            return False

    h_str = str(hyp_value).strip().lower()
    c_str = str(csv_value).strip().lower()

    if param_name == "energy_source":
        if "maltodextrin" in h_str and "maltodextrin" in c_str:
            return True
        if "maltose" in h_str and "maltose" in c_str and "maltodextrin" not in c_str:
            return True
        if h_str in ("3-pga", "3-pga 30mm") and "3-pga" in c_str and "maltose" not in c_str and "maltodextrin" not in c_str:
            return True
        if "pep" in h_str and "pep" in c_str:
            return True
        if "pyruvate" in h_str and "pyruvate" in c_str:
            return True
        return False

    if param_name == "extract_type":
        if "kc6" in h_str and "kc6" in c_str:
            return True
        if "wild" in h_str and ("wild" in c_str or "wt" in c_str):
            return True
        if "bl21" in h_str and "bl21" in c_str:
            return True
        if "a19" in h_str and "a19" in c_str:
            return True
        for term in ["spea", "tnaa", "sdaa"]:
            if term in h_str and term in c_str:
                return True
        return False

    if param_name == "chaperones":
        if h_str == "none" and c_str == "none":
            return True
        if "combined" in h_str and "combined" in c_str:
            return True
        if "dnak" in h_str and "dnak" in c_str:
            return True
        if "groes" in h_str and "groes" in c_str:
            return True
        return False

    if param_name == "reaction_mode":
        if "semi" in h_str and "semi" in c_str:
            return True
        if "micro" in h_str and "micro" in c_str:
            return True
        if "dialysis" in h_str and "dialysis" in c_str:
            return True
        if h_str == "batch" and c_str == "batch":
            return True
        return False

    return h_str == c_str


def find_matching_experiments(hypothesis, df=None):
    if df is None:
        df = load_experiments()

    matching_idxs = []
    hyp_params_found = [p for p in PARAM_TO_CSV_COL if hypothesis.get(p) is not None]

    for idx, row in df.iterrows():
        mismatched_params = []
        params_checked = 0

        for param, csv_col in PARAM_TO_CSV_COL.items():
            hyp_value = hypothesis.get(param)
            if hyp_value is None or csv_col not in df.columns:
                continue

            csv_value = row.get(csv_col)
            if pd.isna(csv_value) or csv_value == "":
                params_checked += 1
                continue

            params_checked += 1
            if not _match_param_value(hyp_value, csv_value, param):
                mismatched_params.append((param, hyp_value, csv_value))

        if params_checked == 0:
            continue

        if not mismatched_params:
            matching_idxs.append(idx)
    return matching_idxs, hyp_params_found


def get_experimental_productivity(matching_idxs, df=None):
    if df is None:
        df = load_experiments()

    if not matching_idxs:
        return {
            "has_data": False,
            "num_matches": 0,
            "matched_experiments": [],
            "min_productivity": None,
            "max_productivity": None,
            "avg_productivity": None,
        }

    prods = []
    exp_ids = []
    for idx in matching_idxs:
        prod = parse_productivity_range(df.loc[idx, "estimated_productivity_mg_mL_hr"])
        exp_ids.append(df.loc[idx, "experiment_id"])
        if prod is not None:
            prods.append(prod)

    if not prods:
        return {
            "has_data": False,
            "num_matches": len(matching_idxs),
            "matched_experiments": exp_ids,
            "min_productivity": None,
            "max_productivity": None,
            "avg_productivity": None,
        }

    return {
        "has_data": True,
        "num_matches": len(matching_idxs),
        "matched_experiments": exp_ids,
        "min_productivity": min(prods),
        "max_productivity": max(prods),
        "avg_productivity": sum(prods) / len(prods),
    }


def get_best_experimental_hypothesis(hypotheses, cross_check_results):
    best_idx = None
    best_prod = None
    hypotheses_with_data = 0

    for i, (_, result) in enumerate(zip(hypotheses, cross_check_results)):
        exp_data = result.get("experimental_productivity", {})
        if exp_data.get("has_data"):
            hypotheses_with_data += 1
            min_prod = exp_data["min_productivity"]
            if best_prod is None or (min_prod is not None and min_prod > best_prod):
                best_prod = min_prod
                best_idx = i

    return best_idx, best_prod, hypotheses_with_data


def cross_check_hypotheses(
    hypotheses,
    df=None,
    verbose=True,
    use_oracle_fallback=True,
    oracle_model_path=ORACLE_MODEL_PATH,
):
    if df is None:
        df = load_experiments()

    results = []
    oracle_model = _load_oracle_model(oracle_model_path) if use_oracle_fallback else None

    for i, hyp in enumerate(hypotheses):
        matching_idxs, hyp_params_found = find_matching_experiments(hyp, df)
        exp_data = get_experimental_productivity(matching_idxs, df)

        oracle_prediction = None
        if use_oracle_fallback:
            oracle_prediction = oracle_predict_hypothesis(hyp, oracle_model)

        result = {
            "hypothesis_name": hyp.get("hypothesis_name", f"Hypothesis {i+1}"),
            "hypothesis": hyp,
            "matching_idxs": matching_idxs,
            "near_matches": [],
            "experimental_productivity": exp_data,
            "oracle_prediction": oracle_prediction,
        }
        results.append(result)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Hypothesis: {result['hypothesis_name']}")
            print(f"  Hypothesis params found: {len(hyp_params_found)}/11 - {hyp_params_found}")
            if exp_data["has_data"]:
                print(f"  MATCHED {exp_data['num_matches']} experiment(s):")
                print(f"    Experiments: {exp_data['matched_experiments']}")
                print(f"    Experimental productivity: {exp_data['min_productivity']:.3f} mg/mL/hr")
            else:
                print("  NO FULL MATCH in experimental dataset.")
            if oracle_prediction is not None:
                print(
                    "  ORACLE prediction: "
                    f"{oracle_prediction['predicted_productivity_mg_mL_hr']:.3f} mg/mL/hr"
                )
            elif use_oracle_fallback:
                print("  ORACLE unavailable (model file not found).")

    return results


def summarize_evidence(cross_check_result, df=None):
    if df is None:
        df = load_experiments()

    rows = []
    for idx in cross_check_result.get("matching_idxs", []):
        exp = df.loc[idx]
        rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "study": exp["study"],
                "productivity": exp["estimated_productivity_mg_mL_hr"],
                "total_yield": exp.get("total_yield_mg_mL", ""),
                "notes": exp.get("notes", ""),
            }
        )

    return pd.DataFrame(rows)
