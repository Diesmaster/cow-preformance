import os
import json
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence

from base_models.BaseModel import BaseModel


class OLSModel(BaseModel):
    def __init__(self, independent_attrs: list, dependent_attr: str, n: int, title: str):
        if isinstance(independent_attrs, str):
            self.independent_attrs = [independent_attrs]
        else:
            self.independent_attrs = independent_attrs
        self.dependent_attr = dependent_attr
        self.n = n
        self.title = f"OLS_{title}"
        self.model = None
        self.results = None
        self.formula = None
        self.cv_results = None
        self.diagnostics = None

    def _build_formula(self):
        self.formula = f"{self.dependent_attr} ~ {' + '.join(self.independent_attrs)}"
        return self.formula

    def fit(self, df):
        required_cols = self.independent_attrs + [self.dependent_attr] + ['cow_id']
        df_clean = df[required_cols].dropna()
        formula = self._build_formula()
        self.model = smf.ols(formula, data=df_clean)
        self.results = self.model.fit()
        try:
            self.diagnostics = self._run_diagnostic_tests(df_clean)
        except Exception as e:
            self.diagnostics = {'error': str(e)}
        return self

    def _run_diagnostic_tests(self, df_clean):
        if self.results is None:
            raise ValueError("Model must be fitted before diagnostics")
        resid = self.results.resid
        exog = self.results.model.exog
        exog_names = list(self.results.model.exog_names)
        diagnostics = {}
        try:
            ad_stat, ad_p = normal_ad(resid)
            diagnostics['normality'] = {'anderson_darling_stat': float(ad_stat), 'pvalue': float(ad_p)}
        except Exception as e:
            diagnostics['normality'] = {'error': str(e)}
        try:
            bp_test = het_breuschpagan(resid, exog)
            diagnostics['breusch_pagan'] = {'lm_stat': float(bp_test[0]), 'lm_pvalue': float(bp_test[1]),
                                             'f_stat': float(bp_test[2]), 'f_pvalue': float(bp_test[3])}
        except Exception as e:
            diagnostics['breusch_pagan'] = {'error': str(e)}
        try:
            white_test = het_white(resid, exog)
            diagnostics['white_test'] = {'stat': float(white_test[0]), 'pvalue': float(white_test[1]),
                                          'f_stat': float(white_test[2]), 'f_pvalue': float(white_test[3])}
        except Exception as e:
            diagnostics['white_test'] = {'error': str(e)}
        try:
            dw = durbin_watson(resid)
            diagnostics['durbin_watson'] = float(dw)
        except Exception as e:
            diagnostics['durbin_watson'] = {'error': str(e)}
        try:
            vifs = []
            for i in range(exog.shape[1]):
                try:
                    vif_val = variance_inflation_factor(exog, i)
                except Exception:
                    vif_val = np.nan
                vifs.append(float(vif_val) if not np.isnan(vif_val) else np.nan)
            diagnostics['vif'] = dict(zip(exog_names, vifs))
        except Exception as e:
            diagnostics['vif'] = {'error': str(e)}
        try:
            reset_res = linear_reset(self.results)
            diagnostics['reset'] = {'fvalue': float(getattr(reset_res, 'fvalue', np.nan)),
                                     'pvalue': float(getattr(reset_res, 'pvalue', np.nan))}
        except Exception as e:
            diagnostics['reset'] = {'error': str(e)}
        try:
            influence = OLSInfluence(self.results)
            cooks_d = np.asarray(influence.cooks_distance[0])
            leverage = np.asarray(influence.hat_matrix_diag)
            student_resid = np.asarray(influence.resid_studentized_external)
            diagnostics['influence'] = {
                'cooks_distance_max': float(np.nanmax(cooks_d)),
                'cooks_distance_mean': float(np.nanmean(cooks_d)),
                'leverage_max': float(np.nanmax(leverage)),
                'leverage_mean': float(np.nanmean(leverage)),
                'cooks_distance_array': cooks_d.tolist(),
                'leverage_array': leverage.tolist(),
                'studentized_resid_array': student_resid.tolist()
            }
        except Exception as e:
            diagnostics['influence'] = {'error': str(e)}
        return diagnostics

    def cross_validate(self, df, k=5, random_state=42):
        required_cols = self.independent_attrs + [self.dependent_attr]
        df_clean = df[required_cols].dropna().reset_index(drop=True)
        if k == 'loo':
            cv = LeaveOneOut()
            n_splits = len(df_clean)
            print(f"Performing Leave-One-Out CV with {n_splits} folds...")
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
            n_splits = k
            print(f"Performing {k}-Fold Cross-Validation...")
        fold_metrics = {'r2': [], 'mae': [], 'rmse': []}
        formula = self._build_formula()
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df_clean)):
            train_df = df_clean.iloc[train_idx]
            test_df = df_clean.iloc[test_idx]
            model = smf.ols(formula, data=train_df)
            result = model.fit()
            y_pred = result.predict(test_df)
            y_true = test_df[self.dependent_attr]
            fold_metrics['r2'].append(r2_score(y_true, y_pred))
            fold_metrics['mae'].append(mean_absolute_error(y_true, y_pred))
            fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            if k != 'loo' or fold_idx < 5:
                print(f"Fold {fold_idx+1}/{n_splits} -> R²: {fold_metrics['r2'][-1]:.4f}, "
                      f"MAE: {fold_metrics['mae'][-1]:.4f}, RMSE: {fold_metrics['rmse'][-1]:.4f}")
        self.cv_results = {
            'mean_r2': np.mean(fold_metrics['r2']), 'std_r2': np.std(fold_metrics['r2']),
            'mean_mae': np.mean(fold_metrics['mae']), 'std_mae': np.std(fold_metrics['mae']),
            'mean_rmse': np.mean(fold_metrics['rmse']), 'std_rmse': np.std(fold_metrics['rmse']),
            'fold_metrics': fold_metrics, 'n_splits': n_splits
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
        self.fit(df)
        return self

    def predict(self, df):
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.results.predict(df)

    def evaluate(self, df):
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")
        y_pred = self.predict(df)
        y_true = df[self.dependent_attr].dropna()
        min_len = min(len(y_pred), len(y_true))
        return {
            'r2': r2_score(y_true[:min_len], y_pred[:min_len]),
            'mae': mean_absolute_error(y_true[:min_len], y_pred[:min_len]),
            'rmse': np.sqrt(mean_squared_error(y_true[:min_len], y_pred[:min_len]))
        }

    def summary(self):
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")
        print(self.results.summary())
        if self.cv_results is not None:
            print(f"\n{'='*60}\nCross-Validation Results:\n{'='*60}")
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print('='*60)

    def print_diagnostics(self, show_arrays=False):
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")
        d = self.diagnostics
        print(f"\n{'='*60}\nDiagnostics Summary:\n{'='*60}")
        normal = d.get('normality', {})
        if 'error' in normal:
            print("Normality test: ERROR -", normal['error'])
        else:
            print(f"Normality (Anderson-Darling) stat: {normal['anderson_darling_stat']:.4f}, p: {normal['pvalue']:.4g}")
        bp = d.get('breusch_pagan', {})
        if 'error' in bp:
            print("Breusch-Pagan: ERROR -", bp['error'])
        else:
            print(f"Breusch-Pagan LM p-value: {bp['lm_pvalue']:.4g}, f-test p: {bp['f_pvalue']:.4g}")
        white = d.get('white_test', {})
        if 'error' in white:
            print("White test: ERROR -", white['error'])
        else:
            print(f"White test p-value: {white['pvalue']:.4g}")
        dw = d.get('durbin_watson', {})
        if isinstance(dw, dict) and 'error' in dw:
            print("Durbin-Watson: ERROR -", dw['error'])
        else:
            print(f"Durbin-Watson: {float(dw):.4f} (≈2 => no autocorrelation)")
        reset = d.get('reset', {})
        if 'error' in reset:
            print("RESET test: ERROR -", reset['error'])
        else:
            print(f"Ramsey RESET F p-value: {reset['pvalue']:.4g}")
        vif = d.get('vif', {})
        if isinstance(vif, dict):
            print("\nVariance Inflation Factors (VIF):")
            for name, val in vif.items():
                try:
                    print(f"  {name}: {val:.4f}")
                except Exception:
                    print(f"  {name}: {val}")
        infl = d.get('influence', {})
        if 'error' in infl:
            print("Influence diagnostics: ERROR -", infl['error'])
        else:
            print(f"\nInfluence summary:")
            print(f"  Max Cook's distance: {infl['cooks_distance_max']:.6g}")
            print(f"  Mean Cook's distance: {infl['cooks_distance_mean']:.6g}")
            print(f"  Max leverage: {infl['leverage_max']:.6g}")
            print(f"  Mean leverage: {infl['leverage_mean']:.6g}")
            if show_arrays:
                print("\n  Cook's distance array:", infl.get('cooks_distance_array'))
                print("  Leverage array:", infl.get('leverage_array'))
                print("  Studentized residuals array:", infl.get('studentized_resid_array'))
        print('='*60)

    def save_results(self):
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")
        results_dir = os.path.join('model_results', self.title)
        os.makedirs(results_dir, exist_ok=True)
        results_dict = {
            'title': self.title, 'n': self.n, 'formula': self.formula,
            'independent_variables': self.independent_attrs, 'dependent_variable': self.dependent_attr,
            'summary': {
                'r_squared': float(self.results.rsquared), 'adj_r_squared': float(self.results.rsquared_adj),
                'f_statistic': float(self.results.fvalue), 'f_pvalue': float(self.results.f_pvalue),
                'aic': float(self.results.aic), 'bic': float(self.results.bic),
                'n_observations': int(self.results.nobs)
            },
            'coefficients': {
                name: {
                    'value': float(self.results.params[name]), 'std_err': float(self.results.bse[name]),
                    't_stat': float(self.results.tvalues[name]), 'p_value': float(self.results.pvalues[name]),
                    'conf_int_lower': float(self.results.conf_int().loc[name, 0]),
                    'conf_int_upper': float(self.results.conf_int().loc[name, 1])
                }
                for name in self.results.params.index
            },
            'diagnostics': self.diagnostics,
            'cross_validation': self.cv_results if self.cv_results else None
        }
        json_path = os.path.join(results_dir, f'{self.n}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {json_path}")
        return json_path

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
        y_pred = self.predict(df)
        y_true = df[self.dependent_attr].dropna()
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        plt.figure(figsize=(10, 6))
        if len(self.independent_attrs) == 1:
            x_vals = df[self.independent_attrs[0]].dropna()[:min_len]
            plt.scatter(x_vals, y_true, color='blue', alpha=0.7, label='Data points')
            sort_idx = np.argsort(x_vals)
            plt.plot(x_vals.iloc[sort_idx], y_pred[sort_idx], 'r--', label='Regression line')
            plt.xlabel(self.independent_attrs[0])
            plt.ylabel(self.dependent_attr)
            plt.title(f'{self.title}: {self.dependent_attr} vs {self.independent_attrs[0]}')
        else:
            plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Data points')
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect fit')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{self.title}: {self.dependent_attr} - Predicted vs Actual')
        metrics = self.evaluate(df)
        metrics_text = f"R²: {metrics['r2']:.3f}\nMAE: {metrics['mae']:.3f}\nRMSE: {metrics['rmse']:.3f}"
        if self.cv_results is not None:
            metrics_text += f"\n\nCV R²: {self.cv_results['mean_r2']:.3f} ± {self.cv_results['std_r2']:.3f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend()
        plt.tight_layout()
        if save:
            results_dir = os.path.join('model_results', self.title)
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'{self.n}_plot.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Plot saved to {plot_path}")

    def get_coefficients(self):
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")
        return self.results.params
