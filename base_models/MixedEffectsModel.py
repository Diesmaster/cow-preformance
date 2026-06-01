import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

from base_models.BaseModel import BaseModel


class MixedEffectsModel(BaseModel):
    def __init__(self, independent_attrs, dependent_attr, n, title,
                 group_col: str = "cow_id", min_group_size: int = 4, use_reml: bool = True):
        super().__init__(independent_attrs, dependent_attr, n, title)
        self.group_col = group_col
        self.min_group_size = min_group_size
        self.use_reml = use_reml
        self._required_cols_cached = None
        self.df = None  # stores last cleaned df for summary()

    def _required_cols(self):
        if self._required_cols_cached is None:
            self._required_cols_cached = self.independent_attrs + [self.dependent_attr, self.group_col]
        return self._required_cols_cached

    def _clean_df(self, df):
        missing = [c for c in self._required_cols() if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        df_clean = df[self._required_cols()].dropna()
        group_counts = df_clean[self.group_col].value_counts()
        small_groups = group_counts[group_counts < self.min_group_size].index.tolist()
        if small_groups:
            print(f"⚠️  Dropping {len(small_groups)} group(s) with < {self.min_group_size} observations")
            df_clean = df_clean[~df_clean[self.group_col].isin(small_groups)]
        self.df = df_clean
        return df_clean

    def _build_formula(self):
        return f"{self.dependent_attr} ~ {' + '.join(self.independent_attrs)}"

    def _check_tau(self):
        group_var = getattr(self.results, 'cov_re', None)
        if group_var is not None:
            try:
                gv_val = float(np.squeeze(group_var.values))
            except Exception:
                gv_val = 0.0
            if gv_val < 1e-6:
                print("⚠️  tau² near zero — random intercepts collapsed. Consider PanelOLS.")
            else:
                print(f"✓  Group Var (tau²) = {gv_val:.6f} — random intercepts active")

    def _check_corr(self, df_clean):
        temp = df_clean.drop(columns=[self.group_col])
        corr = temp.corr().round(3)
        threshold = 0.9
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col_i, col_j = corr.columns[i], corr.columns[j]
                value = corr.iloc[i, j]
                if abs(value) >= threshold:
                    high_corr_pairs.append((col_i, col_j, value))
        if high_corr_pairs:
            print("⚠️ Highly correlated feature pairs (|r| >=", threshold, "):")
            for col_i, col_j, value in high_corr_pairs:
                print(f"  {col_i:35s} <-> {col_j:35s} | r = {value:.3f}")
        else:
            print("✅ No highly correlated feature pairs found.")
        print("\nFull correlation matrix:")
        print(corr)

    # ────────────────────────────────────────────────────────────────
    # Core model methods
    # ────────────────────────────────────────────────────────────────

    def fit(self, df):
        df_clean = self._clean_df(df)
        self.formula = self._build_formula()
        self.model = smf.mixedlm(self.formula, df_clean, groups=df_clean[self.group_col])
        self.results = self.model.fit(reml=self.use_reml, method="lbfgs")
        self._check_tau()
        try:
            resid = self.results.resid
            ad_stat, ad_p = normal_ad(resid)
            dw = durbin_watson(resid)
            self.diagnostics = {
                "residual_normality": {"anderson_darling_stat": float(ad_stat), "pvalue": float(ad_p)},
                "durbin_watson": float(dw),
                "groups": int(df_clean[self.group_col].nunique()),
                "n_obs": int(df_clean.shape[0]),
                "reml": self.use_reml,
            }
        except Exception as e:
            self.diagnostics = {"error": str(e)}
        return self

    def predict(self, df, conditional: bool = True):
        """
        Predict ADG for rows in df.

        Parameters
        ----------
        conditional : bool, default True
            If True,  returns fixed-effect prediction + per-cow random intercept (BLUP).
                      This is the real in-sample fit — use for plotting and in-sample eval.
            If False, returns fixed-effect prediction only (marginal / population-level).
                      This is what you'd use for unseen cows with no BLUP available.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        needed = self.independent_attrs + [self.group_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for prediction: {missing}")

        # ── Fixed-effect (marginal) predictions ──────────────────────
        try:
            y_fixed = self.results.predict(df)
        except (ValueError, np.linalg.LinAlgError):
            print("⚠️  Singular covariance: predicting with fixed effects only")
            exog = df[self.independent_attrs].copy()
            exog.insert(0, "Intercept", 1.0)
            params    = self.results.fe_params
            col_order = [c for c in params.index if c in exog.columns]
            y_fixed   = pd.Series(
                exog[col_order].values @ params[col_order].values,
                index=df.index, name="predicted",
            )

        if not conditional:
            return y_fixed

        # ── Add per-cow random intercept (BLUP) ──────────────────────
        try:
            re_map = {
                group: float(blup.values[0])
                for group, blup in self.results.random_effects.items()
            }
            random_intercepts = df[self.group_col].map(re_map).fillna(0.0)
            return (y_fixed + random_intercepts).rename("predicted")
        except (ValueError, np.linalg.LinAlgError):
            print("⚠️  Random intercepts unavailable — returning marginal predictions")
            return y_fixed

    def evaluate(self, df, conditional: bool = True):
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")
        df_clean = self._clean_df(df)
        y_true = df_clean[self.dependent_attr].values
        y_pred = self.predict(df_clean, conditional=conditional).values
        return {
            "r2":   r2_score(y_true, y_pred),
            "mae":  mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    # ────────────────────────────────────────────────────────────────
    # Cross-validation
    # ────────────────────────────────────────────────────────────────

    def cross_validate(self, df, k=5, random_state=42):
        df_clean = self._clean_df(df).reset_index(drop=True)
        self._check_corr(df_clean)
        cv = GroupKFold(n_splits=k)
        print(f"Performing {k}-Fold Cross-Validation (GroupKFold by {self.group_col})...")
        fold_metrics = {"r2": [], "mae": [], "rmse": []}
        groups  = df_clean[self.group_col]
        formula = self._build_formula()
        for i, (tr, te) in enumerate(cv.split(df_clean, groups=groups)):
            train_df, test_df = df_clean.iloc[tr], df_clean.iloc[te]
            m   = smf.mixedlm(formula, train_df, groups=train_df[self.group_col])
            res = m.fit(reml=self.use_reml, method="lbfgs")
            # CV always uses marginal predictions — test cows have no trained BLUP
            try:
                y_pred = res.predict(test_df).values
            except (ValueError, np.linalg.LinAlgError):
                exog = test_df[self.independent_attrs].copy()
                exog.insert(0, "Intercept", 1.0)
                params    = res.fe_params
                col_order = [c for c in params.index if c in exog.columns]
                y_pred    = exog[col_order].values @ params[col_order].values
            y_true = test_df[self.dependent_attr].values
            fold_metrics["r2"].append(r2_score(y_true, y_pred))
            fold_metrics["mae"].append(mean_absolute_error(y_true, y_pred))
            fold_metrics["rmse"].append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
            print(f"Fold {i+1}/{k} -> R²: {fold_metrics['r2'][-1]:.4f}, "
                  f"MAE: {fold_metrics['mae'][-1]:.4f}, RMSE: {fold_metrics['rmse'][-1]:.4f}")
        self.cv_results = {
            "mean_r2":   float(np.mean(fold_metrics["r2"])),
            "std_r2":    float(np.std(fold_metrics["r2"])),
            "mean_mae":  float(np.mean(fold_metrics["mae"])),
            "std_mae":   float(np.std(fold_metrics["mae"])),
            "mean_rmse": float(np.mean(fold_metrics["rmse"])),
            "std_rmse":  float(np.std(fold_metrics["rmse"])),
            "fold_metrics": fold_metrics,
            "n_splits": k,
        }
        print(f"\n{'='*60}\nCross-Validation Summary:\n{'='*60}")
        print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
        print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
        print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
        print('='*60)
        return self.cv_results

    def fit_with_cv(self, df, k=5, random_state=42):
        self.cross_validate(df, k=k, random_state=random_state)
        print("\nFitting model on full dataset...")
        return self.fit(df)

    # ────────────────────────────────────────────────────────────────
    # R² — Nakagawa & Schielzeth (2013)
    # ────────────────────────────────────────────────────────────────

    def r2_nakagawa(self, df) -> dict:
        """
        Compute marginal and conditional R² (Nakagawa & Schielzeth, 2013).

        Marginal R²   — variance explained by fixed effects only.
        Conditional R² — variance explained by fixed + random effects.

        Returns
        -------
        dict with keys: marginal, conditional, var_fixed, var_random, var_resid, icc
        """
        if self.results is None:
            raise ValueError("Fit the model first.")
        df_clean  = self._clean_df(df)
        exog      = df_clean[self.independent_attrs].copy()
        exog.insert(0, "Intercept", 1.0)
        fe_params = self.results.fe_params
        col_order = [c for c in fe_params.index if c in exog.columns]
        y_fixed   = exog[col_order].values @ fe_params[col_order].values
        var_fixed  = float(np.var(y_fixed, ddof=0))
        cov_re     = getattr(self.results, "cov_re", None)
        try:
            var_random = float(np.squeeze(cov_re.values)) if cov_re is not None else 0.0
        except Exception:
            var_random = 0.0
        var_resid = float(self.results.scale)
        var_total = var_fixed + var_random + var_resid
        return {
            "marginal":    round(var_fixed / var_total, 4),
            "conditional": round((var_fixed + var_random) / var_total, 4),
            "var_fixed":   round(var_fixed,  6),
            "var_random":  round(var_random, 6),
            "var_resid":   round(var_resid,  6),
            "icc":         round(var_random / (var_random + var_resid), 4),
        }

    # ────────────────────────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────────────────────────

    def _print_random_effects(self):
        try:
            re = {str(k): float(v.values[0]) for k, v in self.results.random_effects.items()}
            print(f"\n{'='*60}\nRandom Intercepts by {self.group_col}:\n{'='*60}")
            print(f"  {'Group':<20s} {'Intercept':>12s}")
            print(f"  {'-'*20} {'-'*12}")
            for group, intercept in sorted(re.items(), key=lambda x: x[0]):
                print(f"  {group:<20s} {intercept:>12.4f}")
            print('='*60)
        except (ValueError, np.linalg.LinAlgError):
            print("\n⚠️  Random intercepts not available (singular covariance structure)")

    def summary(self):
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")
        print(self.results.summary())
        self._print_random_effects()

        # Nakagawa R² — uses df cached by the last _clean_df call (i.e. from fit)
        if self.df is not None:
            try:
                r2 = self.r2_nakagawa(self.df)
                print(f"\n{'='*60}")
                print(f"Nakagawa R² (Marginal / Conditional):")
                print(f"{'='*60}")
                print(f"  Marginal  R²  (fixed effects only) : {r2['marginal']:.4f}")
                print(f"  Conditional R² (fixed + random)    : {r2['conditional']:.4f}")
                print(f"  Δ random-effect contribution       : {r2['conditional'] - r2['marginal']:.4f}")
                print(f"  ICC  (between-group variance)      : {r2['icc']:.4f}")
                print(f"  ── variance decomposition ──")
                print(f"  σ²_fixed    : {r2['var_fixed']:.6f}")
                print(f"  τ²_random   : {r2['var_random']:.6f}")
                print(f"  σ²_residual : {r2['var_resid']:.6f}")
                print('='*60)
            except Exception as e:
                print(f"⚠️  Nakagawa R² unavailable: {e}")
        else:
            print("\n  ℹ️  Fit the model first to include Nakagawa R².")

        if self.cv_results is not None:
            print(f"\n{'='*60}\nCross-Validation Results (marginal — unseen cows):\n{'='*60}")
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print('='*60)

    def print_diagnostics(self, show_arrays: bool = False):
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")
        d = self.diagnostics
        print(f"\n{'='*60}\nDiagnostics Summary (MixedLM):\n{'='*60}")
        if "error" in d:
            print("Diagnostics ERROR:", d["error"])
        else:
            rn = d.get("residual_normality", {})
            if rn:
                print(f"Residual normality (Anderson-Darling): "
                      f"stat={rn.get('anderson_darling_stat'):.4f}, p={rn.get('pvalue'):.4g}")
            dw = d.get("durbin_watson", None)
            if dw is not None:
                print(f"Durbin-Watson: {float(dw):.4f} (approx 2 => no autocorrelation)")
            print(f"Groups: {d.get('groups')}, Observations: {d.get('n_obs')}")
            print(f"REML: {d.get('reml', False)}")
        print('='*60)
        self._print_random_effects()

    # ────────────────────────────────────────────────────────────────
    # Plotting
    # ────────────────────────────────────────────────────────────────

    def plot(self, df, save=True, conditional: bool = True):
        """
        Plot predicted vs actual (or fit line for single-predictor models).

        Parameters
        ----------
        conditional : bool, default True
            True  → uses fixed + random intercept (per-cow BLUP). Real in-sample fit.
            False → uses fixed effects only (marginal). Honest out-of-sample view.
        """
        if self.results is None:
            raise ValueError("Model must be fitted before plotting")
        df_clean   = self._clean_df(df)
        y_true     = df_clean[self.dependent_attr].values
        y_pred     = self.predict(df_clean, conditional=conditional).values
        pred_label = "conditional (fixed + random)" if conditional else "marginal (fixed only)"

        plt.figure(figsize=(10, 6))
        if len(self.independent_attrs) == 1:
            x     = df_clean[self.independent_attrs[0]].values
            order = np.argsort(x)
            plt.scatter(x, y_true, alpha=0.7, label="Data points")
            plt.plot(x[order], y_pred[order], "r--", label=f"Model fit ({pred_label})")
            plt.xlabel(self.independent_attrs[0])
            plt.ylabel(self.dependent_attr)
            plt.title(f"{self.title}: {self.dependent_attr} vs {self.independent_attrs[0]}")
        else:
            plt.scatter(y_true, y_pred, alpha=0.7, label=f"Predicted vs Actual ({pred_label})")
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            plt.plot(lims, lims, "r--", label="Perfect fit")
            plt.xlabel("Actual")
            plt.ylabel(f"Predicted ({pred_label})")
            plt.title(f"{self.title}: {self.dependent_attr} — Predicted vs Actual")

        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        txt  = f"R²: {r2:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}"
        if self.cv_results is not None:
            txt += (f"\n\nCV R² (marginal): "
                    f"{self.cv_results['mean_r2']:.3f} ± {self.cv_results['std_r2']:.3f}")

        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.legend()
        plt.tight_layout()

        if save:
            suffix     = "conditional" if conditional else "marginal"
            results_dir = os.path.join("model_results", f"FE_{self.title}")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{self.n}_{suffix}_plot.png")
            plt.savefig(path, dpi=300)
            print(f"Plot saved to {path}")

    # ────────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────────

    def save_results(self):
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")
        results_dir = os.path.join("model_results", self.title)
        os.makedirs(results_dir, exist_ok=True)
        try:
            random_effects = {str(k): float(v.values[0]) for k, v in self.results.random_effects.items()}
        except (ValueError, np.linalg.LinAlgError):
            random_effects = None
            print("⚠️  Random effects singular — saving fixed-effects coefficients only")
        try:
            n_groups = int(getattr(self.results, "k_groups", 0)) or (len(random_effects) if random_effects else 0)
        except Exception:
            n_groups = 0

        # Include Nakagawa R² in saved payload if possible
        nakagawa = None
        if self.df is not None:
            try:
                nakagawa = self.r2_nakagawa(self.df)
            except Exception:
                pass

        payload = {
            "title": self.title, "n": self.n, "formula": self.formula,
            "group_col": self.group_col, "min_group_size": self.min_group_size, "reml": self.use_reml,
            "independent_variables": self.independent_attrs, "dependent_variable": self.dependent_attr,
            "summary": {
                "aic": float(self.results.aic), "bic": float(self.results.bic),
                "llf": float(self.results.llf), "scale": float(self.results.scale),
                "n_groups": n_groups,
            },
            "fixed_effect_params": {k: float(v) for k, v in self.results.fe_params.items()},
            "random_effects":  random_effects,
            "nakagawa_r2":     nakagawa,
            "diagnostics":     self.diagnostics,
            "cross_validation": self.cv_results,
        }
        path = os.path.join(results_dir, f"{self.n}_results.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to {path}")
        return path

    def export(self):
        """Export the fitted model to results/exported_models/<title>_<n>.joblib"""
        if self.results is None:
            raise ValueError("Model must be fitted before exporting")
        export_dir = os.path.join("results", "exported_models")
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, f"{self.title}_{self.n}.joblib")
        joblib.dump(self, path)
        print(f"✓ Model exported to: {path}")
        return path

    # ────────────────────────────────────────────────────────────────
    # Accessors
    # ────────────────────────────────────────────────────────────────

    def get_coefficients(self):
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.results.fe_params

    def get_random_effects(self):
        if self.results is None:
            raise ValueError("Model must be fitted before accessing random effects")
        try:
            return {str(k): float(v.values[0]) for k, v in self.results.random_effects.items()}
        except (ValueError, np.linalg.LinAlgError):
            print("⚠️  Cannot retrieve random effects: singular covariance structure")
            return None
