import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
import matplotlib.pyplot as plt
import os
import json

from base_models.BaseModel import BaseModel

# Additional imports for diagnostics
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence



class OLSModel(BaseModel):
    """
    A simple OLS (Ordinary Least Squares) regression model wrapper with automatic diagnostics.

    Attributes:
        independent_attrs (list): List of independent variable names
        dependent_attr (str): Name of the dependent variable
        title (str): Model title for saving results
        model: The statsmodels OLS model (unfitted or fitted)
        results: The fitted model results
        diagnostics: dict with diagnostic test results (populated after fit)
    """

    def __init__(self, independent_attrs: list, dependent_attr: str, n: int, title: str):
        # Ensure independent_attrs is a list
        if isinstance(independent_attrs, str):
            self.independent_attrs = [independent_attrs]
        else:
            self.independent_attrs = independent_attrs

        self.dependent_attr = dependent_attr
        self.n = n
        self.title = title
        self.model = None
        self.results = None
        self.formula = None
        self.cv_results = None
        self.diagnostics = None

    def _build_formula(self):
        """Build the regression formula with cow fixed effects."""
        # Add categorical term for cow_id
        fixed_effects_term = "C(cow_id)"
        self.formula = f"{self.dependent_attr} ~ {' + '.join(self.independent_attrs)} + {fixed_effects_term}"
        return self.formula

    def fit(self, df):
        """
        Fit the OLS model to the data and automatically run diagnostics.

        Args:
            df (pd.DataFrame): DataFrame containing the data

        Returns:
            self: Returns self for method chaining
        """
        # Drop missing values
        required_cols = self.independent_attrs + [self.dependent_attr] + ['cow_id']
        df_clean = df[required_cols].dropna()

        # Build and fit the model
        formula = self._build_formula()
        self.model = smf.ols(formula, data=df_clean)
        self.results = self.model.fit()

        # Run diagnostics automatically and store results
        try:
            self.diagnostics = self._run_diagnostic_tests(df_clean)
        except Exception as e:
            # If a diagnostic fails, store the exception message for debugging
            self.diagnostics = {'error': str(e)}

        return self

    def _run_diagnostic_tests(self, df_clean):
        """
        Internal: run common OLS diagnostic tests and influence measures.

        Returns:
            dict: Results of assumption tests and influence diagnostics
        """
        if self.results is None:
            raise ValueError("Model must be fitted before diagnostics")

        resid = self.results.resid
        exog = self.results.model.exog  # includes intercept column if present
        exog_names = list(self.results.model.exog_names)

        diagnostics = {}

        # 1) Normality of residuals (Anderson-Darling)
        try:
            ad_stat, ad_p = normal_ad(resid)
            diagnostics['normality'] = {'anderson_darling_stat': float(ad_stat), 'pvalue': float(ad_p)}
        except Exception as e:
            diagnostics['normality'] = {'error': str(e)}

        # 2) Homoskedasticity (Breusch-Pagan) and White's test
        try:
            bp_test = het_breuschpagan(resid, exog)
            # bp_test returns (Lagrange multiplier stat, pvalue, f_stat, f_pvalue)
            diagnostics['breusch_pagan'] = {
                'lm_stat': float(bp_test[0]),
                'lm_pvalue': float(bp_test[1]),
                'f_stat': float(bp_test[2]),
                'f_pvalue': float(bp_test[3])
            }
        except Exception as e:
            diagnostics['breusch_pagan'] = {'error': str(e)}

        try:
            white_test = het_white(resid, exog)
            # het_white returns (Statistic, p-value, f-value, f p-value)
            diagnostics['white_test'] = {
                'stat': float(white_test[0]),
                'pvalue': float(white_test[1]),
                'f_stat': float(white_test[2]),
                'f_pvalue': float(white_test[3])
            }
        except Exception as e:
            diagnostics['white_test'] = {'error': str(e)}

        # 3) Autocorrelation (Durbin-Watson)
        try:
            dw = durbin_watson(resid)
            diagnostics['durbin_watson'] = float(dw)
        except Exception as e:
            diagnostics['durbin_watson'] = {'error': str(e)}

        # 4) Multicollinearity (VIF)
        try:
            # Compute VIF for each exog column
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

        # 5) Specification test (Ramsey RESET)
        try:
            reset_res = linear_reset(self.results)
            # linear_reset returns an object with .fvalue and .pvalue (depending on statsmodels version)
            diagnostics['reset'] = {
                'fvalue': float(getattr(reset_res, 'fvalue', np.nan)),
                'pvalue': float(getattr(reset_res, 'pvalue', np.nan))
            }
        except Exception as e:
            diagnostics['reset'] = {'error': str(e)}

        # 6) Influence and outliers (Cook's distance, leverage, studentized residuals)
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
                # expose arrays if user needs them
                'cooks_distance_array': cooks_d.tolist(),  # Convert to list for JSON
                'leverage_array': leverage.tolist(),
                'studentized_resid_array': student_resid.tolist()
            }
        except Exception as e:
            diagnostics['influence'] = {'error': str(e)}

        return diagnostics

    def cross_validate(self, df, k=5, random_state=42):
        """
        Perform k-fold cross-validation.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            k (int or str): Number of folds. Use 'loo' for leave-one-out CV
            random_state (int): Random state for reproducibility

        Returns:
            dict: Dictionary containing cross-validation results
        """
        # Drop missing values
        required_cols = self.independent_attrs + [self.dependent_attr]
        df_clean = df[required_cols].dropna().reset_index(drop=True)

        # Set up cross-validation
        if k == 'loo':
            cv = LeaveOneOut()
            n_splits = len(df_clean)
            print(f"Performing Leave-One-Out CV with {n_splits} folds...")
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
            n_splits = k
            print(f"Performing {k}-Fold Cross-Validation...")

        # Store metrics for each fold
        fold_metrics = {
            'r2': [],
            'mae': [],
            'rmse': []
        }

        formula = self._build_formula()

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df_clean)):
            # Split data
            train_df = df_clean.iloc[train_idx]
            test_df = df_clean.iloc[test_idx]

            # Fit model on training data
            model = smf.ols(formula, data=train_df)
            result = model.fit()

            # Predict on test data
            y_pred = result.predict(test_df)
            y_true = test_df[self.dependent_attr]

            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            fold_metrics['r2'].append(r2)
            fold_metrics['mae'].append(mae)
            fold_metrics['rmse'].append(rmse)

            if k != 'loo' or fold_idx < 5:  # Print first 5 folds for LOO
                print(f"Fold {fold_idx + 1}/{n_splits} -> R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Calculate average metrics
        self.cv_results = {
            'mean_r2': np.mean(fold_metrics['r2']),
            'std_r2': np.std(fold_metrics['r2']),
            'mean_mae': np.mean(fold_metrics['mae']),
            'std_mae': np.std(fold_metrics['mae']),
            'mean_rmse': np.mean(fold_metrics['rmse']),
            'std_rmse': np.std(fold_metrics['rmse']),
            'fold_metrics': fold_metrics,
            'n_splits': n_splits
        }

        print("\n" + "=" * 60)
        print("Cross-Validation Summary:")
        print("=" * 60)
        print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
        print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
        print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
        print("=" * 60)

        return self.cv_results

    def fit_with_cv(self, df, k=5, random_state=42):
        """
        Perform cross-validation and then fit on the full dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            k (int or str): Number of folds. Use 'loo' for leave-one-out CV
            random_state (int): Random state for reproducibility

        Returns:
            self: Returns self for method chaining
        """
        # Perform cross-validation
        self.cross_validate(df, k=k, random_state=random_state)

        # Fit on full dataset
        print("\nFitting model on full dataset...")
        self.fit(df)

        return self

    def predict(self, df):
        """
        Make predictions using the fitted model.

        Args:
            df (pd.DataFrame): DataFrame containing the independent variables

        Returns:
            np.ndarray: Predicted values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions")

        return self.results.predict(df)

    def evaluate(self, df):
        """
        Evaluate the model and return metrics.

        Args:
            df (pd.DataFrame): DataFrame containing both independent and dependent variables

        Returns:
            dict: Dictionary containing R², MAE, and RMSE
        """
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")

        # Get predictions
        y_pred = self.predict(df)
        y_true = df[self.dependent_attr].dropna()

        # Ensure same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        # Calculate metrics
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }

        return metrics

    def summary(self):
        """Print the model summary and CV results."""
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")

        print(self.results.summary())

        # Also print CV results if available
        if self.cv_results is not None:
            print("\n" + "=" * 60)
            print("Cross-Validation Results:")
            print("=" * 60)
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print("=" * 60)

    def print_diagnostics(self, show_arrays=False):
        """
        Nicely print stored diagnostics. Set show_arrays=True to display influence arrays.

        Args:
            show_arrays (bool): If True, prints the full arrays for cooks_distance, leverage, etc.
        """
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")

        d = self.diagnostics

        print("\n" + "=" * 60)
        print("Diagnostics Summary:")
        print("=" * 60)

        # Normality
        normal = d.get('normality', {})
        if 'error' in normal:
            print("Normality test: ERROR -", normal['error'])
        else:
            print(f"Normality (Anderson-Darling) stat: {normal['anderson_darling_stat']:.4f}, p: {normal['pvalue']:.4g}")

        # Breusch-Pagan
        bp = d.get('breusch_pagan', {})
        if 'error' in bp:
            print("Breusch-Pagan: ERROR -", bp['error'])
        else:
            print(f"Breusch-Pagan LM p-value: {bp['lm_pvalue']:.4g}, f-test p: {bp['f_pvalue']:.4g}")

        # White
        white = d.get('white_test', {})
        if 'error' in white:
            print("White test: ERROR -", white['error'])
        else:
            print(f"White test p-value: {white['pvalue']:.4g}")

        # Durbin-Watson
        dw = d.get('durbin_watson', {})
        if isinstance(dw, dict) and 'error' in dw:
            print("Durbin-Watson: ERROR -", dw['error'])
        else:
            print(f"Durbin-Watson: {float(dw):.4f} (≈2 => no autocorrelation)")

        # RESET
        reset = d.get('reset', {})
        if 'error' in reset:
            print("RESET test: ERROR -", reset['error'])
        else:
            print(f"Ramsey RESET F p-value: {reset['pvalue']:.4g}")

        # VIF
        vif = d.get('vif', {})
        if isinstance(vif, dict):
            print("\nVariance Inflation Factors (VIF):")
            for name, val in vif.items():
                try:
                    print(f"  {name}: {val:.4f}")
                except Exception:
                    print(f"  {name}: {val}")
        else:
            print("VIF: ERROR -", vif)

        # Influence
        infl = d.get('influence', {})
        if 'error' in infl:
            print("Influence diagnostics: ERROR -", infl['error'])
        else:
            print("\nInfluence summary:")
            print(f"  Max Cook's distance: {infl['cooks_distance_max']:.6g}")
            print(f"  Mean Cook's distance: {infl['cooks_distance_mean']:.6g}")
            print(f"  Max leverage: {infl['leverage_max']:.6g}")
            print(f"  Mean leverage: {infl['leverage_mean']:.6g}")
            if show_arrays:
                print("\n  Cook's distance array:", infl.get('cooks_distance_array'))
                print("  Leverage array:", infl.get('leverage_array'))
                print("  Studentized residuals array:", infl.get('studentized_resid_array'))

        print("=" * 60)

    def save_results(self):
        """
        Save model results, diagnostics, and summary to JSON file.
        Saves to model_results/<title>/<n>_results.json
        """
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")

        # Create directory structure
        results_dir = os.path.join('model_results', self.title)
        os.makedirs(results_dir, exist_ok=True)

        # Prepare results dictionary
        results_dict = {
            'title': self.title,
            'n': self.n,
            'formula': self.formula,
            'independent_variables': self.independent_attrs,
            'dependent_variable': self.dependent_attr,
            'summary': {
                'r_squared': float(self.results.rsquared),
                'adj_r_squared': float(self.results.rsquared_adj),
                'f_statistic': float(self.results.fvalue),
                'f_pvalue': float(self.results.f_pvalue),
                'aic': float(self.results.aic),
                'bic': float(self.results.bic),
                'n_observations': int(self.results.nobs)
            },
            'coefficients': {
                name: {
                    'value': float(self.results.params[name]),
                    'std_err': float(self.results.bse[name]),
                    't_stat': float(self.results.tvalues[name]),
                    'p_value': float(self.results.pvalues[name]),
                    'conf_int_lower': float(self.results.conf_int().loc[name, 0]),
                    'conf_int_upper': float(self.results.conf_int().loc[name, 1])
                }
                for name in self.results.params.index
            },
            'diagnostics': self.diagnostics,
            'cross_validation': self.cv_results if self.cv_results else None
        }

        # Save to JSON
        json_path = os.path.join(results_dir, f'{self.n}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to {json_path}")
        return json_path

    def plot(self, df, save=True):
        """
        Plot actual vs predicted values.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            save (bool): If True, saves plot to model_results/<title>/<n>_plot.png
        """
        if self.results is None:
            raise ValueError("Model must be fitted before plotting")

        y_pred = self.predict(df)
        y_true = df[self.dependent_attr].dropna()

        # Ensure same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        plt.figure(figsize=(10, 6))

        if len(self.independent_attrs) == 1:
            # Single variable: plot actual vs independent variable with regression line
            x_vals = df[self.independent_attrs[0]].dropna()[:min_len]
            plt.scatter(x_vals, y_true, color='blue', alpha=0.7, label='Data points')

            # Sort for line plotting
            sort_idx = np.argsort(x_vals)
            plt.plot(x_vals.iloc[sort_idx], y_pred[sort_idx], 'r--', label='Regression line')

            plt.xlabel(self.independent_attrs[0])
            plt.ylabel(self.dependent_attr)
            plt.title(f'{self.title}: {self.dependent_attr} vs {self.independent_attrs[0]}')
        else:
            # Multiple variables: plot predicted vs actual
            plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Data points')
            plt.plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', label='Perfect fit')

            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{self.title}: {self.dependent_attr} - Predicted vs Actual')

        # Add metrics to plot
        metrics = self.evaluate(df)
        metrics_text = f"R²: {metrics['r2']:.3f}\nMAE: {metrics['mae']:.3f}\nRMSE: {metrics['rmse']:.3f}"

        # Add CV metrics if available
        if self.cv_results is not None:
            metrics_text += f"\n\nCV R²: {self.cv_results['mean_r2']:.3f} ± {self.cv_results['std_r2']:.3f}"

        plt.text(0.05, 0.95,
                 metrics_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
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
        """
        Get model coefficients.

        Returns:
            pd.Series: Model coefficients
        """
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")

        return self.results.params
