import os
import json
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

from base_models.BaseModel import BaseModel


class FixedEffectsModel(BaseModel):
    """
    Linear mixed model with per-cow random intercepts (practical fixed-effects style).
    Group column defaults to 'cow_id'. Set group_col='cattleId' if your data uses that.
    """

    def __init__(self, independent_attrs, dependent_attr, n, title, group_col: str = "cow_id"):
        super().__init__(independent_attrs, dependent_attr, n, title)
        self.group_col = group_col
        self._required_cols_cached = None  # for speed

    # ----------------- helpers -----------------
    def _required_cols(self):
        if self._required_cols_cached is None:
            self._required_cols_cached = self.independent_attrs + [self.dependent_attr, self.group_col]
        return self._required_cols_cached

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self._required_cols() if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        return df[self._required_cols()].dropna()

    def _build_formula(self) -> str:
        return f"{self.dependent_attr} ~ {' + '.join(self.independent_attrs)}"

    # ----------------- Abstract methods -----------------
    def fit(self, df: pd.DataFrame):
        """Fit MixedLM with random intercepts by group (cow)."""
        df_clean = self._clean_df(df)
        self.formula = self._build_formula()

        # Random intercept per group
        self.model = smf.mixedlm(self.formula, df_clean, groups=df_clean[self.group_col])
        # Using ML (reml=False) for easier model comparison; use method="lbfgs" for robustness
        self.results = self.model.fit(reml=False, method="lbfgs")

        # Basic diagnostics on residuals (MixedLM doesn't expose OLSInfluence)
        try:
            resid = self.results.resid
            ad_stat, ad_p = normal_ad(resid)
            dw = durbin_watson(resid)

            self.diagnostics = {
                "residual_normality": {"anderson_darling_stat": float(ad_stat), "pvalue": float(ad_p)},
                "durbin_watson": float(dw),
                "groups": int(df_clean[self.group_col].nunique()),
                "n_obs": int(df_clean.shape[0]),
            }
        except Exception as e:
            self.diagnostics = {"error": str(e)}

        return self

    def predict(self, df: pd.DataFrame):
        """Predict using the fitted MixedLM (handles seen & unseen groups)."""
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        # We allow missing y during prediction; only need X and group
        needed = self.independent_attrs + [self.group_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for prediction: {missing}")

        return self.results.predict(df[needed] if set(needed) != set(df.columns) else df)

    def _check_corr(self, df_clean):
        temp = df_clean.drop(columns=["cow_id"])
        corr = temp.corr().round(3)

        # Threshold for "high correlation"
        threshold = 0.9

        # Collect highly correlated feature pairs
        high_corr_pairs = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col_i = corr.columns[i]
                col_j = corr.columns[j]
                value = corr.iloc[i, j]
                if abs(value) >= threshold:
                    high_corr_pairs.append((col_i, col_j, value))

        # Print summary
        if high_corr_pairs:
            print("⚠️ Highly correlated feature pairs (|r| ≥", threshold, "):")
            for col_i, col_j, value in high_corr_pairs:
                print(f"  {col_i:35s} ↔ {col_j:35s} | r = {value:.3f}")
        else:
            print("✅ No highly correlated feature pairs found.")

        # Optionally: visualize or return
        print("\nFull correlation matrix:")
        print(corr)

    def evaluate(self, df: pd.DataFrame):
        """Return R², MAE, RMSE on provided data."""
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")
        df_clean = self._clean_df(df)

        y_true = df_clean[self.dependent_attr].values
        y_pred = self.predict(df_clean).values

        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def cross_validate(self, df: pd.DataFrame, k=5, random_state=42):
        """K-fold/LOO CV with refit per fold."""
        df_clean = self._clean_df(df).reset_index(drop=True)

        self._check_corr(df_clean)

        if k == "loo":
            raise ValueError("Leave-One-Out (LOO) cross-validation is not supported for grouped data. Use GroupKFold instead.")
        else:
            cv = GroupKFold(n_splits=k)
            n_splits = k
            print(f"Performing {k}-Fold Cross-Validation...")

        fold_metrics = {"r2": [], "mae": [], "rmse": []}
        formula = self._build_formula()

        groups = df_clean[self.group_col]

        for i, (tr, te) in enumerate(cv.split(df_clean, groups=groups)):
            train_df = df_clean.iloc[tr]
            test_df = df_clean.iloc[te]

            m = smf.mixedlm(formula, train_df, groups=train_df[self.group_col])
            res = m.fit(reml=False, method="lbfgs")

            y_pred = res.predict(test_df)
            y_true = test_df[self.dependent_attr].values

            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

            fold_metrics["r2"].append(r2)
            fold_metrics["mae"].append(mae)
            fold_metrics["rmse"].append(rmse)

            if k != "loo" or i < 5:
                print(f"Fold {i+1}/{n_splits} -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        self.cv_results = {
            "mean_r2": float(np.mean(fold_metrics["r2"])),
            "std_r2": float(np.std(fold_metrics["r2"])),
            "mean_mae": float(np.mean(fold_metrics["mae"])),
            "std_mae": float(np.std(fold_metrics["mae"])),
            "mean_rmse": float(np.mean(fold_metrics["rmse"])),
            "std_rmse": float(np.std(fold_metrics["rmse"])),
            "fold_metrics": fold_metrics,
            "n_splits": n_splits,
        }

        print("\n" + "=" * 60)
        print("Cross-Validation Summary:")
        print("=" * 60)
        print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
        print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
        print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
        print("=" * 60)

        return self.cv_results

    def fit_with_cv(self, df: pd.DataFrame, k=5, random_state=42):
        """Run CV, then fit on full data."""
        self.cross_validate(df, k=k, random_state=random_state)
        print("\nFitting model on full dataset...")
        return self.fit(df)

    def summary(self):
        """Print statsmodels MixedLM summary plus CV if present."""
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")
        print(self.results.summary())
        if self.cv_results is not None:
            print("\n" + "=" * 60)
            print("Cross-Validation Results:")
            print("=" * 60)
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print("=" * 60)

    def print_diagnostics(self, show_arrays: bool = False):
        """Basic residual diagnostics (normality + DW) and model dimensions."""
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")

        d = self.diagnostics
        print("\n" + "=" * 60)
        print("Diagnostics Summary (MixedLM / FE-style):")
        print("=" * 60)
        if "error" in d:
            print("Diagnostics ERROR:", d["error"])
        else:
            rn = d.get("residual_normality", {})
            if rn:
                print(f"Residual normality (Anderson-Darling): stat={rn.get('anderson_darling_stat'):.4f}, "
                      f"p={rn.get('pvalue'):.4g}")
            dw = d.get("durbin_watson", None)
            if dw is not None:
                print(f"Durbin-Watson: {float(dw):.4f} (≈2 => no autocorrelation)")
            print(f"Groups: {d.get('groups')}, Observations: {d.get('n_obs')}")
        print("=" * 60)

    def save_results(self):
        """Save model info and CV to JSON."""
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")

        results_dir = os.path.join("model_results", self.title)
        os.makedirs(results_dir, exist_ok=True)

        # Random effects per group come as {group: Series([intercept])}
        try:
            random_effects = {str(k): float(v.values[0]) for k, v in self.results.random_effects.items()}
        except Exception:
            # Fallback in case of unexpected shape
            random_effects = {str(k): float(np.squeeze(v)) for k, v in self.results.random_effects.items()}

        payload = {
            "title": self.title,
            "n": self.n,
            "formula": self.formula,
            "group_col": self.group_col,
            "independent_variables": self.independent_attrs,
            "dependent_variable": self.dependent_attr,
            "summary": {
                "aic": float(self.results.aic),
                "bic": float(self.results.bic),
                "llf": float(self.results.llf),
                "scale": float(self.results.scale),
                "n_groups": int(getattr(self.results, "k_groups", len(random_effects))),
            },
            "fixed_effect_params": {k: float(v) for k, v in self.results.fe_params.items()},
            "variance_components": {k: float(v) for k, v in getattr(self.results, "vcov_re", pd.DataFrame()).sum().to_dict().items()} if hasattr(self.results, "vcov_re") else None,
            "random_effects": random_effects,
            "diagnostics": self.diagnostics,
            "cross_validation": self.cv_results,
        }

        path = os.path.join(results_dir, f"{self.n}_results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Results saved to {path}")
        return path

    def plot(self, df: pd.DataFrame, save: bool = True):
        """Plot predicted vs actual (or y vs x with line if single regressor)."""
        if self.results is None:
            raise ValueError("Model must be fitted before plotting")

        df_clean = self._clean_df(df)
        y_true = df_clean[self.dependent_attr].values
        y_pred = self.predict(df_clean).values

        plt.figure(figsize=(10, 6))
        if len(self.independent_attrs) == 1:
            x = df_clean[self.independent_attrs[0]].values
            plt.scatter(x, y_true, alpha=0.7, label="Data points")
            # sort for line
            order = np.argsort(x)
            plt.plot(x[order], y_pred[order], 'r--', label="Model fit")
            plt.xlabel(self.independent_attrs[0])
            plt.ylabel(self.dependent_attr)
            plt.title(f"{self.title}: {self.dependent_attr} vs {self.independent_attrs[0]}")
        else:
            plt.scatter(y_true, y_pred, alpha=0.7, label="Predicted vs Actual")
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, 'r--', label="Perfect fit")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{self.title}: {self.dependent_attr} - Predicted vs Actual")

        metrics = self.evaluate(df_clean)
        txt = f"R²: {metrics['r2']:.3f}\nMAE: {metrics['mae']:.3f}\nRMSE: {metrics['rmse']:.3f}"
        if self.cv_results is not None:
            txt += f"\n\nCV R²: {self.cv_results['mean_r2']:.3f} ± {self.cv_results['std_r2']:.3f}"
        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.legend()
        plt.tight_layout()

        if save:
            results_dir = os.path.join("model_results", f"FE_{self.title}")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{self.n}_plot.png")
            plt.savefig(path, dpi=300)
            print(f"Plot saved to {path}")

    def get_coefficients(self):
        """Return fixed-effect coefficients (β)."""
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.results.fe_params

