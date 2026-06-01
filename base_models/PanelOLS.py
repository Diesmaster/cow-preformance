import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

from base_models.BaseModel import BaseModel


class PanelOLSModel(BaseModel):
    def __init__(self, independent_attrs, dependent_attr, n, title, group_col: str = "cow_id",
                 time_col: str = "pred_date", entity_effects: bool = True, time_effects: bool = False):
        title = f"Panel_{title}"
        super().__init__(independent_attrs, dependent_attr, n, title)
        self.group_col = group_col
        self.time_col = time_col
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self._required_cols_cached = None

    def _required_cols(self):
        if self._required_cols_cached is None:
            self._required_cols_cached = self.independent_attrs + [self.dependent_attr, self.group_col, self.time_col]
        return self._required_cols_cached

    def _clean_df(self, df):
        missing = [c for c in self._required_cols() if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        return df[self._required_cols()].dropna()

    def _prepare_panel_data(self, df):
        df_clean = self._clean_df(df)
        return df_clean.set_index([self.group_col, self.time_col])

    def fit(self, df):
        df_panel = self._prepare_panel_data(df)
        self._training_data = df_panel.copy()
        y = df_panel[self.dependent_attr]
        X = df_panel[self.independent_attrs]
        self.model = PanelOLS(y, X, entity_effects=self.entity_effects, time_effects=self.time_effects)
        self.results = self.model.fit(cov_type='clustered', cluster_entity=True)
        if self.entity_effects:
            self._entity_means = y.groupby(level=0).mean()
            self._grand_mean = y.mean()
            self._mean_fixed_effect = self._entity_means.mean()
        try:
            resid = self.results.resids
            ad_stat, ad_p = normal_ad(resid)
            dw = durbin_watson(resid)
            self.diagnostics = {
                "residual_normality": {"anderson_darling_stat": float(ad_stat), "pvalue": float(ad_p)},
                "durbin_watson": float(dw),
                "n_entities": int(self.results.entity_info.total),
                "n_obs": int(self.results.nobs),
                "mean_fixed_effect": float(self._mean_fixed_effect) if self.entity_effects else None,
            }
        except Exception as e:
            self.diagnostics = {"error": str(e)}
        return self

    def predict(self, df):
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        df_panel = self._prepare_panel_data(df)
        if hasattr(self, '_training_data'):
            common_idx = df_panel.index.intersection(self._training_data.index)
            if len(common_idx) == len(df_panel):
                y_actual = self._training_data[self.dependent_attr]
                residuals = self.results.resids
                y_pred = y_actual - residuals
                return y_pred.reindex(df_panel.index)
        X = df_panel[self.independent_attrs]
        y_pred_demeaned = X @ self.results.params
        if self.entity_effects and hasattr(self, '_entity_means'):
            y_pred = y_pred_demeaned.copy()
            entities = df_panel.index.get_level_values(0)
            for entity in entities.unique():
                entity_mask = entities == entity
                if entity in self._entity_means.index:
                    y_pred[entity_mask] = y_pred_demeaned[entity_mask] + self._entity_means.loc[entity]
                else:
                    y_pred[entity_mask] = y_pred_demeaned[entity_mask] + self._grand_mean
            return y_pred
        return y_pred_demeaned

    def _check_corr(self, df_clean):
        temp = df_clean.drop(columns=[self.group_col, self.time_col])
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
            print("⚠️ Highly correlated feature pairs (|r| ≥", threshold, "):")
            for col_i, col_j, value in high_corr_pairs:
                print(f"  {col_i:35s} ↔ {col_j:35s} | r = {value:.3f}")
        else:
            print("✅ No highly correlated feature pairs found.")
        print("\nFull correlation matrix:")
        print(corr)

    def evaluate(self, df):
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")
        df_panel = self._prepare_panel_data(df)
        y_true = df_panel[self.dependent_attr].values
        y_pred = self.predict(df).values
        return {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def cross_validate(self, df, k=5, random_state=42):
        df_clean = self._clean_df(df).reset_index(drop=True)
        self._check_corr(df_clean)
        cv = GroupKFold(n_splits=k)
        print(f"Performing {k}-Fold Cross-Validation (PanelOLS with entity effects)...")
        fold_metrics = {"r2": [], "mae": [], "rmse": []}
        groups = df_clean[self.group_col]
        for i, (tr, te) in enumerate(cv.split(df_clean, groups=groups)):
            train_df, test_df = df_clean.iloc[tr], df_clean.iloc[te]
            train_panel = train_df.set_index([self.group_col, self.time_col])
            test_panel  = test_df.set_index([self.group_col, self.time_col])
            y_train = train_panel[self.dependent_attr]
            X_train = train_panel[self.independent_attrs]
            y_test  = test_panel[self.dependent_attr]
            X_test  = test_panel[self.independent_attrs]
            m = PanelOLS(y_train, X_train, entity_effects=self.entity_effects, time_effects=self.time_effects)
            res = m.fit(cov_type='clustered', cluster_entity=True)
            y_pred_demeaned = X_test @ res.params
            if self.entity_effects:
                entity_means = y_train.groupby(level=0).mean()
                grand_mean   = y_train.mean()
                y_pred = y_pred_demeaned.copy()
                test_entities = test_panel.index.get_level_values(0)
                for entity in test_entities.unique():
                    entity_mask = test_entities == entity
                    if entity in entity_means.index:
                        y_pred[entity_mask] = y_pred_demeaned[entity_mask] + entity_means.loc[entity]
                    else:
                        y_pred[entity_mask] = y_pred_demeaned[entity_mask] + grand_mean
            else:
                y_pred = y_pred_demeaned
            y_true = y_test.values
            fold_metrics["r2"].append(r2_score(y_true, y_pred))
            fold_metrics["mae"].append(mean_absolute_error(y_true, y_pred))
            fold_metrics["rmse"].append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
            print(f"Fold {i+1}/{k} -> R²: {fold_metrics['r2'][-1]:.4f}, MAE: {fold_metrics['mae'][-1]:.4f}, RMSE: {fold_metrics['rmse'][-1]:.4f}")
        self.cv_results = {
            "mean_r2": float(np.mean(fold_metrics["r2"])), "std_r2": float(np.std(fold_metrics["r2"])),
            "mean_mae": float(np.mean(fold_metrics["mae"])), "std_mae": float(np.std(fold_metrics["mae"])),
            "mean_rmse": float(np.mean(fold_metrics["rmse"])), "std_rmse": float(np.std(fold_metrics["rmse"])),
            "fold_metrics": fold_metrics, "n_splits": k,
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

    def summary(self):
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")
        print(self.results)
        if self.entity_effects and hasattr(self, '_mean_fixed_effect'):
            print(f"\n{'='*60}\nFixed Effects Summary:\n{'='*60}")
            print(f"Mean Fixed Effect: {self._mean_fixed_effect:.4f}")
            print(f"Number of Entities: {len(self._entity_means)}")
            print(f"Entities: {self._entity_means}")
            print('='*60)
        if self.cv_results is not None:
            print(f"\n{'='*60}\nCross-Validation Results:\n{'='*60}")
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print('='*60)

    def print_diagnostics(self, show_arrays: bool = False):
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")
        d = self.diagnostics
        print(f"\n{'='*60}\nDiagnostics Summary (PanelOLS Fixed Effects):\n{'='*60}")
        if "error" in d:
            print("Diagnostics ERROR:", d["error"])
        else:
            rn = d.get("residual_normality", {})
            if rn:
                print(f"Residual normality (Anderson-Darling): stat={rn.get('anderson_darling_stat'):.4f}, p={rn.get('pvalue'):.4g}")
            dw = d.get("durbin_watson", None)
            if dw is not None:
                print(f"Durbin-Watson: {float(dw):.4f} (≈2 => no autocorrelation)")
            print(f"Entities: {d.get('n_entities')}, Observations: {d.get('n_obs')}")
            mean_fe = d.get("mean_fixed_effect", None)
            if mean_fe is not None:
                print(f"Mean Fixed Effect: {mean_fe:.4f}")
        print('='*60)

    def save_results(self):
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")
        results_dir = os.path.join("model_results", f"{self.title}")
        os.makedirs(results_dir, exist_ok=True)
        entity_fixed_effects = None
        if self.entity_effects and hasattr(self, '_entity_means'):
            entity_fixed_effects = {str(k): float(v) for k, v in self._entity_means.items()}
        payload = {
            "title": self.title, "n": self.n, "group_col": self.group_col,
            "time_col": self.time_col, "entity_effects": self.entity_effects, "time_effects": self.time_effects,
            "independent_variables": self.independent_attrs, "dependent_variable": self.dependent_attr,
            "summary": {
                "r2": float(self.results.rsquared), "r2_within": float(self.results.rsquared_within),
                "r2_between": float(self.results.rsquared_between), "r2_overall": float(self.results.rsquared_overall),
                "f_statistic": float(self.results.f_statistic.stat), "f_pvalue": float(self.results.f_statistic.pval),
                "n_entities": int(self.results.entity_info.total), "n_obs": int(self.results.nobs),
                "mean_fixed_effect": float(self._mean_fixed_effect) if self.entity_effects else None,
            },
            "coefficients": {k: float(v) for k, v in self.results.params.items()},
            "std_errors": {k: float(v) for k, v in self.results.std_errors.items()},
            "pvalues": {k: float(v) for k, v in self.results.pvalues.items()},
            "entity_fixed_effects": entity_fixed_effects,
            "diagnostics": self.diagnostics,
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
        export_dir = os.path.join('results', 'exported_models')
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, f"{self.title}_{self.n}.joblib")
        joblib.dump(self, path)
        print(f"✓ Model exported to: {path}")
        return path

    def plot(self, df, save=True):
        if self.results is None:
            raise ValueError("Model must be fitted before plotting")
        df_panel = self._prepare_panel_data(df)
        y_true = df_panel[self.dependent_attr].values
        y_pred = self.predict(df).values
        plt.figure(figsize=(10, 6))
        if len(self.independent_attrs) == 1:
            x = df_panel[self.independent_attrs[0]].values
            plt.scatter(x, y_true, alpha=0.7, label="Data points")
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
        metrics = self.evaluate(df)
        txt = f"R²: {metrics['r2']:.3f}\nMAE: {metrics['mae']:.3f}\nRMSE: {metrics['rmse']:.3f}"
        if self.cv_results is not None:
            txt += f"\n\nCV R²: {self.cv_results['mean_r2']:.3f} ± {self.cv_results['std_r2']:.3f}"
        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.legend()
        plt.tight_layout()
        if save:
            results_dir = os.path.join("model_results", f"{self.title}")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{self.n}_plot.png")
            plt.savefig(path, dpi=300)
            print(f"Plot saved to {path}")

    def get_coefficients(self):
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.results.params

    def get_mean_fixed_effect(self):
        if self.results is None:
            raise ValueError("Model must be fitted before accessing mean fixed effect")
        if not self.entity_effects:
            raise ValueError("Model was not fitted with entity effects")
        return self._mean_fixed_effect
