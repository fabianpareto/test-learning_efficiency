"""
GP-UCB Bayesian Optimization for TX-TL Productivity.

Provides the parameter space, encoding/decoding, GP-UCB acquisition loop,
multi-run orchestration, and behavioral-analysis utilities used by the
GP baseline notebook.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from cross_check import oracle_predict_hypothesis

# ── Parameter space (matches the LLM hypothesis-generator vocabulary) ─────

NUMERIC_PARAMS: list[str] = [
    "Mg_glutamate_mM",
    "K_glutamate_mM",
    "DTT_mM",
    "NTP_multiplier",
    "PEG_8000_percent",
    "temperature_C",
    "plasmid_concentration_nM",
]

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Mg_glutamate_mM": (4.0, 6.0),
    "K_glutamate_mM": (60.0, 80.0),
    "DTT_mM": (0.0, 3.0),
    "NTP_multiplier": (1.0, 3.0),
    "PEG_8000_percent": (0.0, 5.0),
    "temperature_C": (25.0, 37.0),
    "plasmid_concentration_nM": (5.0, 10.0),
}

DISCRETE_TEMPS: list[int] = [25, 30, 37]

CATEGORICAL_PARAMS: list[str] = [
    "energy_source",
    "extract_type",
    "chaperones",
    "reaction_mode",
]

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "energy_source": ["3-PGA", "3-PGA + maltose", "3-PGA + maltodextrin"],
    "extract_type": ["wild-type", "KC6"],
    "chaperones": ["none", "GroES/EL", "DnaK/DnaJ/GrpE", "combined"],
    "reaction_mode": ["batch", "semi-continuous", "microfluidic"],
}

N_NUMERIC = len(NUMERIC_PARAMS)
N_ONEHOT = sum(len(v) for v in CATEGORICAL_OPTIONS.values())
N_FEATURES = N_NUMERIC + N_ONEHOT


# ── Encoding / decoding ──────────────────────────────────────────────────

def encode_hypothesis(hyp: dict) -> np.ndarray:
    """Hypothesis dict -> [0,1]-normalised numeric + one-hot feature vector."""
    x: list[float] = []
    for col in NUMERIC_PARAMS:
        lo, hi = NUMERIC_BOUNDS[col]
        x.append((float(hyp[col]) - lo) / (hi - lo) if hi > lo else 0.0)
    for col in CATEGORICAL_PARAMS:
        val = hyp[col]
        for option in CATEGORICAL_OPTIONS[col]:
            x.append(1.0 if val == option else 0.0)
    return np.array(x)


def decode_feature_vector(x: np.ndarray) -> dict:
    """Inverse of *encode_hypothesis* (for inspection)."""
    hyp: dict[str, Any] = {}
    for i, col in enumerate(NUMERIC_PARAMS):
        lo, hi = NUMERIC_BOUNDS[col]
        hyp[col] = round(lo + x[i] * (hi - lo), 2)
    offset = N_NUMERIC
    for col in CATEGORICAL_PARAMS:
        opts = CATEGORICAL_OPTIONS[col]
        onehot = x[offset : offset + len(opts)]
        hyp[col] = opts[int(np.argmax(onehot))]
        offset += len(opts)
    return hyp


# ── Candidate generation & oracle ────────────────────────────────────────

def random_hypothesis(rng: np.random.Generator) -> dict:
    """Sample a hypothesis uniformly from the parameter space."""
    hyp: dict[str, Any] = {}
    for col in NUMERIC_PARAMS:
        if col == "temperature_C":
            hyp[col] = float(rng.choice(DISCRETE_TEMPS))
        else:
            lo, hi = NUMERIC_BOUNDS[col]
            hyp[col] = round(float(rng.uniform(lo, hi)), 2)
    for col in CATEGORICAL_PARAMS:
        hyp[col] = str(rng.choice(CATEGORICAL_OPTIONS[col]))
    return hyp


def oracle_score(hyp: dict) -> float:
    """Query the oracle model and return predicted productivity (mg/mL/hr)."""
    result = oracle_predict_hypothesis(hyp)
    assert result is not None, "Oracle model not found"
    return result["predicted_productivity_mg_mL_hr"]


# ── GP-UCB optimisation loop ─────────────────────────────────────────────

def run_gp_optimization(
    n_total: int = 30,
    n_seed: int = 1,
    kappa: float = 2.0,
    n_candidates: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Single run of Bayesian optimisation with GP-UCB for TX-TL productivity.

    Parameters
    ----------
    n_total       : total oracle queries (budget)
    n_seed        : random starting points
    kappa         : UCB exploration weight  a(x) = mu(x) + kappa * sigma(x)
    n_candidates  : random candidates per acquisition step
    random_state  : RNG seed
    verbose       : print per-iteration progress
    """
    rng = np.random.default_rng(random_state)

    X_data: list[np.ndarray] = []
    y_data: list[float] = []
    hypotheses: list[dict] = []

    if verbose:
        print(f"{'=' * 60}")
        print(f"GP-UCB  seed={random_state}  kappa={kappa}")
        print(f"  {n_seed} seed + {n_total - n_seed} GP = {n_total} oracle queries")
        print(f"{'=' * 60}")

    # Phase 1 – random seed points
    for i in range(n_seed):
        hyp = random_hypothesis(rng)
        y = oracle_score(hyp)
        X_data.append(encode_hypothesis(hyp))
        y_data.append(y)
        hypotheses.append(hyp)
        if verbose:
            print(f"  [Seed {i + 1}/{n_seed}] oracle={y:.3f} mg/mL/hr")

    # Phase 2 – GP-guided acquisition
    for i in range(n_total - n_seed):
        X = np.array(X_data)
        y = np.array(y_data)

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=int(rng.integers(0, 2**31)),
        )
        gp.fit(X, y)

        cand_hyps = [random_hypothesis(rng) for _ in range(n_candidates)]
        cand_X = np.array([encode_hypothesis(h) for h in cand_hyps])
        mu, sigma = gp.predict(cand_X, return_std=True)
        ucb = mu + kappa * sigma

        best_idx = int(np.argmax(ucb))
        best_hyp = cand_hyps[best_idx]
        best_y = oracle_score(best_hyp)

        X_data.append(cand_X[best_idx])
        y_data.append(best_y)
        hypotheses.append(best_hyp)

        if verbose:
            step = n_seed + i + 1
            print(
                f"  [GP {step}/{n_total}] oracle={best_y:.3f}  "
                f"best_so_far={max(y_data):.3f}  "
                f"ucb={ucb[best_idx]:.3f}  "
                f"mu={mu[best_idx]:.3f}  sigma={sigma[best_idx]:.3f}"
            )

    best_overall_idx = int(np.argmax(y_data))
    return {
        "y_history": y_data,
        "hypotheses": hypotheses,
        "best_y": max(y_data),
        "best_hypothesis": hypotheses[best_overall_idx],
        "random_state": random_state,
        "kappa": kappa,
        "n_seed": n_seed,
        "n_total": n_total,
    }


def run_multiple(
    n_runs: int = 100,
    n_total: int = 30,
    n_seed: int = 1,
    kappa: float = 2.0,
    n_candidates: int = 100,
    base_seed: int = 42,
    print_every: int = 50,
) -> list[dict]:
    """Run *n_runs* independent GP-UCB optimisations, printing progress."""
    results: list[dict] = []
    for run in range(n_runs):
        seed = base_seed + run * 111
        res = run_gp_optimization(
            n_total=n_total,
            n_seed=n_seed,
            kappa=kappa,
            n_candidates=n_candidates,
            random_state=seed,
            verbose=False,
        )
        results.append(res)
        if (run + 1) % print_every == 0 or run == 0:
            print(f"  Completed {run + 1}/{n_runs} runs...")
    return results


# ── Stepwise statistics ──────────────────────────────────────────────────

def stepwise_stats(
    results: list[dict],
    cumulative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (step_mean, step_se, ci95_lo, ci95_hi).

    If *cumulative* is True (default), each run's curve is the cumulative max
    (best-so-far).  If False, raw per-step oracle scores are used.
    """
    n_runs = len(results)
    if cumulative:
        matrix = np.array(
            [np.maximum.accumulate(r["y_history"]) for r in results]
        )
    else:
        matrix = np.array([r["y_history"] for r in results])
    step_mean = matrix.mean(axis=0)
    step_se = matrix.std(axis=0) / np.sqrt(n_runs)
    return step_mean, step_se, step_mean - 1.96 * step_se, step_mean + 1.96 * step_se


# ── Behavioral analysis ─────────────────────────────────────────────────

def load_oracle_model(
    path: str | Path = "oracle_artifacts/txtl_oracle_model.json",
) -> dict:
    """Load oracle model JSON and derive coefficient metadata."""
    with open(path) as f:
        model = json.load(f)

    coeffs = np.array(model["coefficients"])
    numeric_features = model["numeric_features"]
    categorical_features = model["categorical_features"]
    categories = model["preprocessor"]["categories"]

    coeff_sign = {
        col: float(coeffs[i]) for i, col in enumerate(numeric_features)
    }

    numeric_optimal = {
        "Mg_glutamate_mM": 6.0,
        "K_glutamate_mM": 80.0,
        "DTT_mM": 0.0,
        "NTP_multiplier": 3.0,
        "PEG_8000_percent": 5.0,
        "temperature_C": 30,
        "plasmid_concentration_nM": 10.0,
    }

    hyp_gen_vocab = {
        "energy_source": ["3-PGA", "3-PGA + maltose", "3-PGA + maltodextrin"],
        "extract_type": ["wild-type", "KC6"],
        "chaperones": ["none", "GroES/EL", "DnaK/DnaJ/GrpE", "combined"],
        "reaction_mode": ["batch", "semi-continuous", "microfluidic"],
    }

    idx = len(numeric_features)
    cat_coeff_map: dict[str, dict[str, float]] = {}
    for feat in categorical_features:
        cats = categories[feat]
        cat_coeffs = coeffs[idx : idx + len(cats)]
        cat_coeff_map[feat] = {
            c.lower(): float(v) for c, v in zip(cats, cat_coeffs)
        }
        idx += len(cats)

    cat_optimals: dict[str, str] = {}
    cat_coeff_tables: dict[str, list[tuple[str, float]]] = {}
    for feat, vocab in hyp_gen_vocab.items():
        scored = [
            (v, cat_coeff_map.get(feat, {}).get(v.lower()))
            for v in vocab
            if cat_coeff_map.get(feat, {}).get(v.lower()) is not None
        ]
        if scored:
            cat_optimals[feat] = max(scored, key=lambda x: x[1])[0]
        cat_coeff_tables[feat] = scored

    return {
        "model": model,
        "coeff_sign": coeff_sign,
        "numeric_optimal": numeric_optimal,
        "cat_optimals": cat_optimals,
        "cat_coeff_tables": cat_coeff_tables,
    }


def _compute_exploitation(
    hyp: dict,
    prev_best_hyp: dict,
    coeff_sign: dict[str, float],
    numeric_optimal: dict[str, float],
    cat_optimals: dict[str, str],
) -> float:
    n_changed, n_constructive = 0, 0
    for param in numeric_optimal:
        try:
            curr, prev = float(hyp[param]), float(prev_best_hyp[param])
        except (TypeError, ValueError, KeyError):
            continue
        delta = curr - prev
        if abs(delta) < 1e-9:
            continue
        n_changed += 1
        n_constructive += (coeff_sign.get(param, 0) * delta) > 0
    for param, optimal in cat_optimals.items():
        prev_opt = str(prev_best_hyp.get(param, "")).lower() == optimal.lower()
        curr_opt = str(hyp.get(param, "")).lower() == optimal.lower()
        delta_01 = int(curr_opt) - int(prev_opt)
        if delta_01 != 0:
            n_changed += 1
            n_constructive += delta_01 > 0
    return n_constructive / n_changed if n_changed > 0 else np.nan


def extract_behavioral_data(
    results: list[dict],
    oracle_meta: dict | None = None,
) -> pd.DataFrame:
    """
    Build a per-query DataFrame with exploitation scores and parameter
    distances, analogous to the LLM rationale analysis.
    """
    if oracle_meta is None:
        oracle_meta = load_oracle_model()

    coeff_sign = oracle_meta["coeff_sign"]
    numeric_optimal = oracle_meta["numeric_optimal"]
    cat_optimals = oracle_meta["cat_optimals"]

    rows: list[dict] = []
    for run_idx, res in enumerate(results):
        run_label = f'GP run {run_idx + 1} (seed={res["random_state"]})'
        hyps, ys = res["hypotheses"], res["y_history"]
        for q, (hyp, y) in enumerate(zip(hyps, ys)):
            phase = "seed" if q < res["n_seed"] else "GP-guided"
            exploitation = np.nan
            score_imp = np.nan
            prev_best_score = None
            if q > 0:
                best_idx = int(np.argmax(ys[:q]))
                prev_best_hyp = hyps[best_idx]
                prev_best_score = ys[best_idx]
                exploitation = _compute_exploitation(
                    hyp, prev_best_hyp, coeff_sign, numeric_optimal, cat_optimals
                )
                score_imp = y - prev_best_score

            param_dists: dict[str, float] = {}
            for param, opt in numeric_optimal.items():
                try:
                    param_dists[f"{param}_distance"] = abs(
                        float(hyp[param]) - float(opt)
                    )
                except (TypeError, ValueError, KeyError):
                    param_dists[f"{param}_distance"] = np.nan

            rows.append(
                {
                    "run": run_label,
                    "query": q + 1,
                    "phase": phase,
                    "oracle_prediction": y,
                    "prev_best_score": prev_best_score,
                    "exploitation_score": exploitation,
                    "score_improvement_vs_prior_best": score_imp,
                    "avg_param_distance": np.nanmean(list(param_dists.values())),
                    **param_dists,
                }
            )
    return pd.DataFrame(rows)
