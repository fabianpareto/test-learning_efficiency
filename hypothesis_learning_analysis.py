#!/usr/bin/env python3
"""
Analyze generated hypotheses to detect learning patterns from oracle feedback.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def extract_all_hypotheses(all_iterators, all_cross_checks):
    """Extract all hypotheses with oracle predictions into a single DataFrame."""
    rows = []

    for model_name, iterator in all_iterators.items():
        cross_checks = all_cross_checks[model_name]

        for iter_idx, (hyp, results) in enumerate(zip(iterator.all_hypotheses,
                                                       [r for iter_results in cross_checks
                                                        for r in iter_results])):
            iteration = iter_idx // 3 + 1  # 3 hypotheses per iteration

            oracle_pred = results.get('oracle_prediction')
            oracle_score = oracle_pred.get('predicted_productivity_mg_mL_hr') if oracle_pred else None

            row = {
                'model': model_name,
                'iteration': iteration,
                'hypothesis_name': hyp.get('hypothesis_name', ''),
                'oracle_prediction': oracle_score,
                **{k: hyp.get(k) for k in [
                    'Mg_glutamate_mM', 'K_glutamate_mM', 'DTT_mM', 'NTP_multiplier',
                    'PEG_8000_percent', 'temperature_C', 'plasmid_concentration_nM',
                    'energy_source', 'extract_type', 'chaperones', 'reaction_mode'
                ]}
            }
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_learning_trends(df):
    """Analyze if models improve oracle predictions over iterations."""
    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()

        # Calculate iteration statistics
        iter_stats = model_df.groupby('iteration')['oracle_prediction'].agg([
            'mean', 'max', 'min', 'std'
        ]).reset_index()

        # Linear regression: oracle_prediction vs iteration
        valid = model_df['oracle_prediction'].notna()
        if valid.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                model_df.loc[valid, 'iteration'],
                model_df.loc[valid, 'oracle_prediction']
            )
        else:
            slope, r_value, p_value = np.nan, np.nan, np.nan

        # Best score in first vs last 3 iterations
        first_3_best = model_df[model_df['iteration'] <= 3]['oracle_prediction'].max()
        last_3_best = model_df[model_df['iteration'] >= model_df['iteration'].max() - 2]['oracle_prediction'].max()

        results.append({
            'model': model,
            'total_hypotheses': len(model_df),
            'iterations': model_df['iteration'].max(),
            'best_overall': model_df['oracle_prediction'].max(),
            'first_3_iter_best': first_3_best,
            'last_3_iter_best': last_3_best,
            'improvement': last_3_best - first_3_best if pd.notna([first_3_best, last_3_best]).all() else np.nan,
            'trend_slope': slope,
            'trend_r2': r_value**2 if not np.isnan(r_value) else np.nan,
            'trend_p_value': p_value,
        })

    return pd.DataFrame(results).sort_values('improvement', ascending=False)


def analyze_parameter_convergence(df, oracle_optimal_params):
    """Check if parameters converge toward oracle optimal over iterations."""
    numeric_params = ['Mg_glutamate_mM', 'K_glutamate_mM', 'DTT_mM', 'NTP_multiplier',
                     'PEG_8000_percent', 'temperature_C', 'plasmid_concentration_nM']

    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()

        for param in numeric_params:
            if param not in oracle_optimal_params:
                continue

            optimal_val = oracle_optimal_params[param]

            # Calculate distance to optimal over iterations
            model_df[f'{param}_distance'] = np.abs(
                pd.to_numeric(model_df[param], errors='coerce') - optimal_val
            )

            # Mean distance in first vs last 3 iterations
            first_3 = model_df[model_df['iteration'] <= 3][f'{param}_distance'].mean()
            last_3 = model_df[model_df['iteration'] >= model_df['iteration'].max() - 2][f'{param}_distance'].mean()

            results.append({
                'model': model,
                'parameter': param,
                'first_3_iter_distance': first_3,
                'last_3_iter_distance': last_3,
                'convergence': first_3 - last_3,  # positive = moving toward optimal
            })

    return pd.DataFrame(results)


def analyze_diversity_vs_exploitation(df):
    """Analyze diversity of hypotheses over iterations."""
    numeric_params = ['Mg_glutamate_mM', 'K_glutamate_mM', 'DTT_mM', 'NTP_multiplier',
                     'PEG_8000_percent', 'temperature_C', 'plasmid_concentration_nM']

    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model].copy()

        for iteration in sorted(model_df['iteration'].unique()):
            iter_df = model_df[model_df['iteration'] == iteration]

            # Calculate coefficient of variation for numeric parameters
            diversity_scores = []
            for param in numeric_params:
                vals = pd.to_numeric(iter_df[param], errors='coerce')
                if vals.std() > 0 and vals.mean() > 0:
                    cv = vals.std() / vals.mean()
                    diversity_scores.append(cv)

            avg_diversity = np.mean(diversity_scores) if diversity_scores else np.nan

            # Count unique categorical values
            categorical_diversity = sum([
                iter_df['energy_source'].nunique(),
                iter_df['extract_type'].nunique(),
                iter_df['chaperones'].nunique(),
                iter_df['reaction_mode'].nunique(),
            ]) / 4.0  # Average unique values per categorical

            results.append({
                'model': model,
                'iteration': iteration,
                'numeric_diversity_cv': avg_diversity,
                'categorical_diversity': categorical_diversity,
                'mean_oracle_pred': iter_df['oracle_prediction'].mean(),
                'max_oracle_pred': iter_df['oracle_prediction'].max(),
            })

    return pd.DataFrame(results)


def plot_learning_curves(df, oracle_optimal=0.5184):
    """Plot learning curves for all models."""
    models = df['model'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        if idx >= len(axes):
            break

        ax = axes[idx]
        model_df = df[df['model'] == model].copy()

        # Plot all hypotheses
        for iteration in sorted(model_df['iteration'].unique()):
            iter_df = model_df[model_df['iteration'] == iteration]
            ax.scatter([iteration] * len(iter_df), iter_df['oracle_prediction'],
                      alpha=0.5, s=30, color='blue')

        # Plot best per iteration
        iter_best = model_df.groupby('iteration')['oracle_prediction'].max()
        ax.plot(iter_best.index, iter_best.values, 'o-', color='black',
               linewidth=2, markersize=8, label='Best per iteration')

        # Plot mean per iteration
        iter_mean = model_df.groupby('iteration')['oracle_prediction'].mean()
        ax.plot(iter_mean.index, iter_mean.values, 's--', color='orange',
               linewidth=1.5, markersize=6, label='Mean per iteration', alpha=0.7)

        # Oracle optimal line
        ax.axhline(oracle_optimal, color='green', linestyle='--', linewidth=2,
                  label=f'Oracle optimal ({oracle_optimal:.3f})', alpha=0.7)

        # Trend line
        valid = model_df['oracle_prediction'].notna()
        if valid.sum() > 2:
            z = np.polyfit(model_df.loc[valid, 'iteration'],
                          model_df.loc[valid, 'oracle_prediction'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(model_df['iteration'].min(), model_df['iteration'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r--', linewidth=1.5, alpha=0.5, label='Trend')

        ax.set_title(model, fontsize=10)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Oracle Prediction (mg/mL/hr)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    # Hide unused axes
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def plot_diversity_trends(diversity_df):
    """Plot diversity trends over iterations."""
    models = diversity_df['model'].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model in models:
        model_df = diversity_df[diversity_df['model'] == model]

        # Numeric diversity
        axes[0].plot(model_df['iteration'], model_df['numeric_diversity_cv'],
                    marker='o', label=model, linewidth=2)

        # Categorical diversity
        axes[1].plot(model_df['iteration'], model_df['categorical_diversity'],
                    marker='s', label=model, linewidth=2)

    axes[0].set_title('Numeric Parameter Diversity (Coefficient of Variation)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Average CV across numeric parameters')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title('Categorical Parameter Diversity')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Average unique values per categorical param')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def summarize_learning_insights(learning_df, convergence_df, diversity_df):
    """Generate text summary of learning insights."""
    print("=" * 80)
    print("HYPOTHESIS LEARNING ANALYSIS")
    print("=" * 80)
    print()

    # Overall learning trends
    print("1. LEARNING TRENDS (improvement from first 3 to last 3 iterations)")
    print("-" * 80)
    for _, row in learning_df.iterrows():
        improvement = row['improvement']
        status = "✓ IMPROVING" if improvement > 0 else "✗ NOT IMPROVING"
        print(f"{row['model']:20s} | {status:15s} | "
              f"Δ={improvement:+.4f} | "
              f"Best={row['best_overall']:.4f} | "
              f"Trend slope={row['trend_slope']:+.5f} (R²={row['trend_r2']:.3f}, p={row['trend_p_value']:.3f})")
    print()

    # Parameter convergence
    print("2. PARAMETER CONVERGENCE (moving toward oracle optimal)")
    print("-" * 80)
    conv_summary = convergence_df.groupby('model')['convergence'].agg(['mean', 'count'])
    conv_summary['pct_converging'] = (convergence_df[convergence_df['convergence'] > 0]
                                      .groupby('model').size() / conv_summary['count'] * 100)
    for model, row in conv_summary.iterrows():
        converging = row['pct_converging']
        status = "✓ CONVERGING" if converging > 50 else "~ MIXED" if converging > 30 else "✗ DIVERGING"
        print(f"{model:20s} | {status:15s} | "
              f"{converging:.1f}% params converging | "
              f"Avg convergence={row['mean']:+.3f}")
    print()

    # Diversity analysis
    print("3. DIVERSITY PATTERNS")
    print("-" * 80)
    for model in diversity_df['model'].unique():
        model_df = diversity_df[diversity_df['model'] == model]

        # Compare early vs late diversity
        early = model_df[model_df['iteration'] <= 3]['numeric_diversity_cv'].mean()
        late = model_df[model_df['iteration'] >= model_df['iteration'].max() - 2]['numeric_diversity_cv'].mean()

        change = late - early
        pattern = "MAINTAINING" if abs(change) < 0.05 else "EXPLOITING" if change < -0.05 else "EXPLORING"

        print(f"{model:20s} | {pattern:12s} | "
              f"Early CV={early:.3f}, Late CV={late:.3f}, Δ={change:+.3f}")
    print()

    # Top hypotheses across all models
    print("4. TOP 10 HYPOTHESES ACROSS ALL MODELS")
    print("-" * 80)
    all_hyp = []
    for model in learning_df['model']:
        model_df = convergence_df[convergence_df['model'] == model]
        # This won't work directly - need the original df
    print("(See full DataFrame for details)")
    print()

    return learning_df, conv_summary
