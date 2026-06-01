import sys
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_processor.DataProcessor import DataProcessing


# ── config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
BREED = 'Limousine'
ANIMAL_COL = 'cow_id'
DATE_COL = 'pred_date'
WEIGHT_COL = 'weight'

MODEL_VARIANT = 'energy'
# allowed: 'energy', 'molecular'

COMMON_STAN_KEYS = [
    'N',
    'n_animals',
    'animal_id',
    'true_weight',
    'days_gap',
    'next_gap',
    'is_first',
    'is_last',
    'next_idx',
    'hasBEF_dt',
    'hormone_effect_dt',
    'gotDewormed_dt',
    'gotHNMVaccination_dt',
]

MAINTENANCE_COLS = {
    'hasBEF_dt': 'hasBEF_dt',
    'hormone_effect_dt': 'hormone_effect_dt',
    'gotDewormed_dt': 'gotDewormed_dt',
    'gotHNMVaccination_dt': 'gotHNMVaccination_dt',
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    stan_file: str
    nutrition_cols: Dict[str, str]
    stan_data_keys: List[str]
    summary_vars: List[str]
    title: str
    plot_subtitle: str


MODEL_CONFIGS = {
    'molecular': ModelConfig(
        name='molecular',
        stan_file=os.path.join(os.path.dirname(__file__), 'stage2_molecular.stan'),
        nutrition_cols={
            'obs_total_cp':   'total_cp_dt',
            'obs_total_cf':   'total_cf_dt',
            'obs_total_fat':  'total_fat_dt',
            'obs_total_betn': 'total_betn_dt',
            'obs_total_dmi':  'total_dmi',
        },
        stan_data_keys=COMMON_STAN_KEYS + [
            'obs_total_cp',
            'obs_total_cf',
            'obs_total_fat',
            'obs_total_betn',
            'obs_total_dmi',
        ],
        summary_vars=[
            'sigma_adg',
            'mcal_per_kg_betn',
            'mcal_per_kg_cp',
            'mcal_per_kg_fat',
            'beta_cf_ratio',
            'const_DE_mul',
            'beta_metabolic_weight',
            'beta_BEF',
            'beta_hormone',
            'beta_dewormed',
            'beta_vaccination',
            'alpha_partition',
            'gamma_hormone',
            'gamma_NEg',
            'NEg_half',
            'gamma_cp_partition',
            'energy_per_kg_muscle',
            'energy_per_kg_fat',
        ],
        title='STAGE 2 — Molecular Energy Model (Kalman-denoised weights)',
        plot_subtitle='Energy Balance — CP / CF / Fat / BETN direct coefficients',
    ),
    'energy': ModelConfig(
        name='energy',
        stan_file=os.path.join(os.path.dirname(__file__), 'stage2_energy.stan'),
        nutrition_cols={
            'obs_tdn_silage': 'tdn_silage_dt',
            'obs_tdn_tahu': 'tdn_tahu_dt',
            'obs_tdn_SP2A': 'tdn_SP2A_dt',
            'obs_tdn_SP2B': 'tdn_SP2B_dt',
            'obs_tdn_SMG': 'tdn_SMG_dt',
            'obs_tdn_rumput': 'tdn_rumput_dt',
        },
        stan_data_keys=COMMON_STAN_KEYS + [
            'obs_tdn_silage',
            'obs_tdn_tahu',
            'obs_tdn_SP2A',
            'obs_tdn_SP2B',
            'obs_tdn_SMG',
            'obs_tdn_rumput',
        ],
        summary_vars=[
            'sigma_adg',
            'mcal_per_kg_silage',
            'mcal_per_kg_tahu',
            'mcal_per_kg_SP2A',
            'mcal_per_kg_SP2B',
            'mcal_per_kg_SMG',
            'mcal_per_kg_rumput',
            'gamma_dmi',
            'beta_metabolic_weight',
            'beta_BEF',
            'beta_hormone',
            'beta_dewormed',
            'beta_vaccination',
            'alpha_partition',
            'gamma_hormone',
            'gamma_NEg',
            'NEg_half',
            'energy_per_kg_muscle',
            'energy_per_kg_fat',
            'day_diff_beta',
        ],
        title='STAGE 2 — Feed-Source Energy Model (Kalman-denoised weights)',
        plot_subtitle='Energy Balance — Per-feed TDN inputs',
    ),
}
# ─────────────────────────────────────────────────────────────────────────────


def get_model_config(model_variant: str) -> ModelConfig:
    if model_variant not in MODEL_CONFIGS:
        valid = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model_variant='{model_variant}'. Choose one of: {valid}")
    return MODEL_CONFIGS[model_variant]


def analyze_large_residual_events(
    df: pd.DataFrame,
    adg_obs_clean: np.ndarray,
    adg_pred_clean: np.ndarray,
    mask: np.ndarray,
    rmse: float,
    results_dir: str,
    breed: str,
    model_name: str,
) -> pd.DataFrame:
    analysis_df = df.copy().reset_index(drop=True)

    valid_idx = np.where(mask)[0]
    residual = adg_obs_clean - adg_pred_clean
    abs_residual = np.abs(residual)
    threshold = 1.5 * rmse
    is_large_error = abs_residual > threshold

    flagged = analysis_df.iloc[valid_idx].copy()
    flagged["observed_adg"] = adg_obs_clean
    flagged["predicted_adg"] = adg_pred_clean
    flagged["residual"] = residual
    flagged["abs_residual"] = abs_residual
    flagged["large_error"] = is_large_error
    flagged["error_direction"] = np.where(
        flagged["residual"] > 0,
        "underpredicted",
        "overpredicted",
    )

    outlier_path = os.path.join(results_dir, f"{breed}_{model_name}_large_residual_events.csv")
    flagged.sort_values("abs_residual", ascending=False).to_csv(outlier_path, index=False)

    print("\n── Large Residual Event Analysis ─────────────────")
    print(f"Threshold: |residual| > 1.5 * RMSE = {threshold:.3f} kg/day")
    print(f"Flagged events: {int(is_large_error.sum())} / {len(flagged)}")

    if is_large_error.sum() == 0:
        print("No events exceeded the threshold.")
        print(f"Saved event table to: {outlier_path}")
        return flagged

    flagged_only = flagged[flagged["large_error"]].copy()
    normal_only = flagged[~flagged["large_error"]].copy()

    print("\nTop 10 worst-missed events:")
    cols_to_show = [
        ANIMAL_COL,
        DATE_COL,
        WEIGHT_COL,
        "observed_adg",
        "predicted_adg",
        "residual",
        "abs_residual",
        "error_direction",
    ]

    extra_cols = [c for c in [
        "day_diff",
        "hasBEF_dt",
        "hormone_effect_dt",
        "gotDewormed_dt",
        "gotHNMVaccination_dt",
        "total_tdn_kg",
        "total_cp_kg",
        "total_cf_kg",
        "total_fat_kg",
        "tdn_silage_dt",
        "tdn_tahu_dt",
        "tdn_SP2A_dt",
        "tdn_SMG_dt",
        "tdn_rumput_dt",
    ] if c in flagged_only.columns]

    print(flagged_only[cols_to_show + extra_cols].head(10).to_string(index=False))

    binary_cols = [c for c in [
        "hasBEF_dt",
        "hormone_effect_dt",
        "gotDewormed_dt",
        "gotHNMVaccination_dt",
    ] if c in flagged.columns]

    if binary_cols:
        print("\nEvent prevalence among flagged vs normal rows:")
        rows = []
        for col in binary_cols:
            flagged_rate = flagged_only[col].fillna(0).mean()
            normal_rate = normal_only[col].fillna(0).mean() if len(normal_only) > 0 else np.nan
            diff = flagged_rate - normal_rate if pd.notna(normal_rate) else np.nan
            rows.append({
                "feature": col,
                "flagged_rate": flagged_rate,
                "normal_rate": normal_rate,
                "diff": diff,
            })
        print(
            pd.DataFrame(rows)
            .sort_values("diff", ascending=False)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    numeric_candidates = [
        WEIGHT_COL,
        "day_diff",
        "total_tdn_kg",
        "total_cp_kg",
        "total_cf_kg",
        "total_fat_kg",
        "tdn_silage_dt",
        "tdn_tahu_dt",
        "tdn_SP2A_dt",
        "tdn_SMG_dt",
        "tdn_rumput_dt",
    ]
    numeric_cols = [c for c in numeric_candidates if c in flagged.columns]

    if numeric_cols:
        print("\nNumeric feature means among flagged vs normal rows:")
        rows = []
        for col in numeric_cols:
            flagged_mean = flagged_only[col].mean()
            normal_mean = normal_only[col].mean() if len(normal_only) > 0 else np.nan
            diff = flagged_mean - normal_mean if pd.notna(normal_mean) else np.nan
            rows.append({
                "feature": col,
                "flagged_mean": flagged_mean,
                "normal_mean": normal_mean,
                "diff": diff,
            })

        numeric_summary = pd.DataFrame(rows)
        numeric_summary["abs_diff"] = numeric_summary["diff"].abs()
        print(
            numeric_summary
            .sort_values("abs_diff", ascending=False)
            .drop(columns=["abs_diff"])
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    underpred = flagged_only[flagged_only["error_direction"] == "underpredicted"]
    overpred = flagged_only[flagged_only["error_direction"] == "overpredicted"]

    print("\nDirection split:")
    print(f"  Underpredicted badly (actual > predicted): {len(underpred)}")
    print(f"  Overpredicted badly  (actual < predicted): {len(overpred)}")
    print(f"\nSaved flagged event table to: {outlier_path}")

    return flagged


def _print_sp2a_stats(df: pd.DataFrame, label: str):
    col = 'tdn_SP2A_dt'
    if col in df.columns:
        non_zero = (df[col] != 0).sum()
        total = len(df)
        print(f"  [SP2A check @ {label}] non-zero: {non_zero} / {total} ({non_zero/total:.1%})")
    else:
        print(f"  [SP2A check @ {label}] column missing")

def load_data(breed: str, model_config: ModelConfig) -> pd.DataFrame:
    processor = DataProcessing(main_folder='../data')
    dfs = processor.get_dfs(n_weighings=[1], apply_smoothing=True)
    df = list(dfs.values())[0]
    df = df[df['breed'].isin([breed])].copy()
    df = df.sort_values([ANIMAL_COL, DATE_COL]).reset_index(drop=True)

    if WEIGHT_COL not in df.columns:
        raise ValueError(f"Column '{WEIGHT_COL}' not found — is apply_smoothing=True?")

    missing = df[WEIGHT_COL].isna().sum()
    if missing > 0:
        print(f"  Warning: {missing} missing smoothed weights — forward-filling")
        df[WEIGHT_COL] = df.groupby(ANIMAL_COL)[WEIGHT_COL].ffill()

    print(f"Loaded {len(df)} observations for {len(df[ANIMAL_COL].unique())} {breed} animals")
    print(f"Smoothed weight range: {df[WEIGHT_COL].min():.1f} – {df[WEIGHT_COL].max():.1f} kg")

    print("\nNutrition summary (mean per observation):")
    for stan_key, col in model_config.nutrition_cols.items():
        if col in df.columns:
            print(
                f"  {col:<16}: {df[col].mean():.2f} "
                f"(min {df[col].min():.2f} / max {df[col].max():.2f} / zeros {int((df[col] == 0).sum())})"
            )
        else:
            print(f"  {col:<16}: MISSING")

    required = list(model_config.nutrition_cols.values())
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' missing from dataframe for model '{model_config.name}'. "
                f"Check that your nutrition lookup populates it."
            )
        zero_frac = (df[col] == 0).mean()
        if zero_frac > 0.5:
            print(
                f"  ⚠️ Warning: '{col}' is zero in {zero_frac:.0%} of rows — "
                f"ingredient data may be missing or sparse"
            )

    return df


def build_stan_data(df: pd.DataFrame, model_config: ModelConfig) -> dict:
    df = df.sort_values([ANIMAL_COL, DATE_COL]).reset_index(drop=True)

    animals = df[ANIMAL_COL].unique()
    animal_map = {a: i + 1 for i, a in enumerate(animals)}
    animal_id = df[ANIMAL_COL].map(animal_map).values.tolist()

    days_gap = []
    is_first = []
    is_last = []
    prev_idx = []
    next_idx = []
    next_gap = []

    for i in range(len(df)):
        animal = df.loc[i, ANIMAL_COL]
        date = df.loc[i, DATE_COL]

        prev_mask = (df[ANIMAL_COL] == animal) & (df[DATE_COL] < date)
        next_mask = (df[ANIMAL_COL] == animal) & (df[DATE_COL] > date)
        prev_indices = df[prev_mask].index
        next_indices = df[next_mask].index

        day_diff = df.loc[i, 'day_diff'] if 'day_diff' in df.columns else 1.0

        if len(prev_indices) == 0:
            is_first.append(1)
            days_gap.append(day_diff)
            prev_idx.append(1)
        else:
            is_first.append(0)
            days_gap.append(day_diff)
            prev_idx.append(int(prev_indices[-1] + 1))

        if len(next_indices) == 0:
            is_last.append(1)
            next_idx.append(1)
            next_gap.append(1.0)
        else:
            q = next_indices[0]
            is_last.append(0)
            next_idx.append(int(q + 1))
            next_gap.append(day_diff)

    nutrition_data = {}
    for stan_key, col in model_config.nutrition_cols.items():
        if col in df.columns:
            nutrition_data[stan_key] = df[col].fillna(0.0).values.tolist()
        else:
            print(f"  Warning: nutrition column '{col}' not found — filling with zeros")
            nutrition_data[stan_key] = [0.0] * len(df)

    maintenance_data = {}
    for stan_key, col in MAINTENANCE_COLS.items():
        if col in df.columns:
            maintenance_data[stan_key] = df[col].fillna(0.0).values.tolist()
        else:
            print(f"  Warning: maintenance column '{col}' not found — filling with zeros")
            maintenance_data[stan_key] = [0.0] * len(df)

    smoothed_adg = []
    for i in range(len(df)):
        if is_last[i] == 0:
            w_now = df.loc[i, WEIGHT_COL]
            w_next = df.loc[next_idx[i] - 1, WEIGHT_COL]
            gap = next_gap[i]
            smoothed_adg.append((w_next - w_now) / gap)

    print("\nSmoothed ADG distribution (what stage 2 is fitting):")
    print(f"  mean:  {np.mean(smoothed_adg):.3f} kg/day")
    print(f"  std:   {np.std(smoothed_adg):.3f} kg/day")
    print(f"  range: [{np.min(smoothed_adg):.3f}, {np.max(smoothed_adg):.3f}]")
    print(f"  n_obs: {len(smoothed_adg)}")

    return {
        'N': len(df),
        'n_animals': len(animals),
        'animal_id': animal_id,
        'true_weight': df[WEIGHT_COL].values.tolist(),
        'days_gap': days_gap,
        'next_gap': next_gap,
        'is_first': is_first,
        'is_last': is_last,
        'prev_idx': prev_idx,
        'next_idx': next_idx,
        **nutrition_data,
        **maintenance_data,
    }


def build_model_data(stan_data: dict, model_config: ModelConfig) -> dict:
    missing_keys = [k for k in model_config.stan_data_keys if k not in stan_data]
    if missing_keys:
        raise KeyError(
            f"Stan data missing required keys for model '{model_config.name}': {missing_keys}"
        )
    return {k: stan_data[k] for k in model_config.stan_data_keys}


def plot_actual_vs_predicted(
    idata,
    summary: pd.DataFrame,
    breed: str,
    model_config: ModelConfig,
    results_dir: str,
):
    import matplotlib.pyplot as plt
    from scipy import stats

    print("\n── Generating Actual vs Prediction Plot ────────")

    try:
        if 'adg_predicted_save' in idata.posterior.data_vars:
            adg_pred = idata.posterior['adg_predicted_save'].mean(dim=['chain', 'draw']).values
        elif 'adg_predicted' in idata.posterior.data_vars:
            adg_pred = idata.posterior['adg_predicted'].mean(dim=['chain', 'draw']).values
        else:
            raise KeyError("adg_predicted not found in posterior")

        if 'adg' in idata.posterior.data_vars:
            adg_obs = idata.posterior['adg'].mean(dim=['chain', 'draw']).values
        else:
            raise KeyError("adg not found in posterior")

    except KeyError as e:
        print(f"  ✗ {e}")
        print("  Skipping plot generation")
        return None

    mask = ~(np.isnan(adg_obs) | np.isnan(adg_pred))
    adg_pred_clean = adg_pred[mask]
    adg_obs_clean = adg_obs[mask]

    print(f"  Valid observations: {len(adg_pred_clean)}")
    print(f"  ADG range: [{adg_obs_clean.min():.3f}, {adg_obs_clean.max():.3f}] kg/day")

    r_squared = stats.pearsonr(adg_obs_clean, adg_pred_clean)[0] ** 2
    rmse = np.sqrt(np.mean((adg_obs_clean - adg_pred_clean) ** 2))
    mae = np.mean(np.abs(adg_obs_clean - adg_pred_clean))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        adg_pred_clean,
        adg_obs_clean,
        alpha=0.6,
        s=30,
        color='steelblue',
        edgecolor='white',
        linewidth=0.5,
    )

    min_val = min(adg_pred_clean.min(), adg_obs_clean.min())
    max_val = max(adg_pred_clean.max(), adg_obs_clean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect (1:1)')

    slope, intercept = np.polyfit(adg_pred_clean, adg_obs_clean, 1)
    line_x = np.array([min_val, max_val])
    ax.plot(line_x, slope * line_x + intercept, 'orange', linewidth=2, label=f'Fitted (slope={slope:.3f})')

    ax.set_xlabel('Predicted ADG (kg/day)', fontsize=12)
    ax.set_ylabel('Observed ADG (kg/day)', fontsize=12)
    ax.set_title(
        f'{breed} Cattle: Actual vs Predicted ADG\n({model_config.plot_subtitle})',
        fontsize=14,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    sigma_adg_mean = summary.loc["sigma_adg", "mean"] if "sigma_adg" in summary.index else np.nan
    metrics_text = (
        f'R² = {r_squared:.3f}\n'
        f'RMSE = {rmse:.3f} kg/day\n'
        f'MAE = {mae:.3f} kg/day\n'
        f'σ_adg = {sigma_adg_mean:.3f} kg/day\n'
        f'n = {len(adg_pred_clean)} obs'
    )
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    plot_path = os.path.join(results_dir, f'{breed}_{model_config.name}_actual_vs_predicted.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  Plot saved to: {plot_path}")
    print(f"  R² = {r_squared:.3f}, RMSE = {rmse:.3f} kg/day")

    return {
        "mask": mask,
        "adg_obs_clean": adg_obs_clean,
        "adg_pred_clean": adg_pred_clean,
        "rmse": rmse,
    }


def save_animal_intercepts(df: pd.DataFrame, idata, breed: str, model_name: str, results_dir: str):
    animals = df[ANIMAL_COL].unique()

    def _summarise_and_save(var_name: str, label: str):
        if var_name not in idata.posterior.data_vars:
            print(f"\n{var_name} not found in posterior — skipping {label} summary")
            return
        print(f"\n── Per-animal intercepts ({var_name}) ─────────")
        posterior = idata.posterior[var_name]
        means = posterior.mean(dim=['chain', 'draw']).values
        sds   = posterior.std(dim=['chain', 'draw']).values
        summary = pd.DataFrame({
            'animal_id':  animals,
            f'{label}_mean': means,
            f'{label}_sd':   sds,
        }).sort_values(f'{label}_mean', ascending=False).reset_index(drop=True)
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n  Grand mean: {means.mean():.4f}")
        print(f"  Std across animals: {means.std():.4f}")
        print(f"  Range: [{means.min():.4f}, {means.max():.4f}]")
        out_path = os.path.join(results_dir, f'{breed}_{model_name}_{var_name}.csv')
        summary.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")

    _summarise_and_save('alpha_animal', 'alpha')   # partition intercept
    _summarise_and_save('u_animal',     'u')       # maintenance intercept


def run(breed: str = BREED, model_variant: str = MODEL_VARIANT):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_config = get_model_config(model_variant)

    df = load_data(breed, model_config)
    stan_data = build_stan_data(df, model_config)
    model_data = build_model_data(stan_data, model_config)

    print("\n" + "=" * 60)
    print(model_config.title)
    print("=" * 60)

    model = CmdStanModel(stan_file=model_config.stan_file, force_compile=True)
    fit = model.sample(
        data=model_data,
        chains=4,
        iter_warmup=1000,
        iter_sampling=1000,
        adapt_delta=0.95,
        show_progress=True,
    )

    print("\n── Diagnostics ──────────────────────────────────")
    print(fit.diagnose())

    idata = az.from_cmdstanpy(fit)
    summary = az.summary(idata, var_names=model_config.summary_vars)

    print("\n── Parameter summary ────────────────────────────")
    print(summary)

    out_path = os.path.join(RESULTS_DIR, f'{breed}_{model_config.name}_stage2.nc')
    idata.to_netcdf(out_path)
    print(f"\nFit saved to: {out_path}")

    plot_outputs = plot_actual_vs_predicted(
        idata=idata,
        summary=summary,
        breed=breed,
        model_config=model_config,
        results_dir=RESULTS_DIR,
    )

    if plot_outputs is not None:
        analyze_large_residual_events(
            df=df,
            adg_obs_clean=plot_outputs["adg_obs_clean"],
            adg_pred_clean=plot_outputs["adg_pred_clean"],
            mask=plot_outputs["mask"],
            rmse=plot_outputs["rmse"],
            results_dir=RESULTS_DIR,
            breed=breed,
            model_name=model_config.name,
        )

    save_animal_intercepts(
        df=df,
        idata=idata,
        breed=breed,
        model_name=model_config.name,
        results_dir=RESULTS_DIR,
    )

    return idata


if __name__ == '__main__':
    breed = sys.argv[1] if len(sys.argv) > 1 else BREED
    model_variant = sys.argv[2] if len(sys.argv) > 2 else MODEL_VARIANT
    run(breed=breed, model_variant=model_variant)
