import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
import os
import json
# Additional imports for diagnostics
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence


from base_models.BaseModel import BaseModel


class SIMEXModel(BaseModel):
    """
    SIMEX (Simulation-Extrapolation) OLS regression model for correcting measurement error
    in the dependent variable.

    Attributes:
        independent_attrs (list): List of independent variable names
        dependent_attr (str): Name of the dependent variable (with measurement error)
        measurement_error_sd: Standard deviation of measurement error (scalar or array)
        title (str): Model title for saving results
        lambdas (np.ndarray): Lambda values for SIMEX simulation
        n_simulations (int): Number of simulations per lambda
        extrapolation_method (str): Method for extrapolation ('quadratic', 'linear', 'nonlinear')
        model: The naive OLS model (without correction)
        results: The naive model results
        simex_results: The corrected SIMEX results
        diagnostics: dict with diagnostic test results (populated after fit)
        cv_results: Cross-validation results
    """

    def __init__(self, independent_attrs: list, dependent_attr: str, 
                 measurement_error_sd, n: int, title: str,
                 lambdas=None, n_simulations=100, extrapolation_method='quadratic'):
        """
        Initialize SIMEX model.

        Args:
            independent_attrs (list): Independent variable names
            dependent_attr (str): Dependent variable name
            measurement_error_sd: Measurement error SD (scalar or array matching data length)
            n (int): Model number/identifier
            title (str): Model title
            lambdas (array-like): Lambda multipliers for added error. Default: [0, 0.5, 1.0, 1.5, 2.0]
            n_simulations (int): Number of simulations per lambda
            extrapolation_method (str): 'quadratic', 'linear', or 'nonlinear'
        """
        if isinstance(independent_attrs, str):
            self.independent_attrs = [independent_attrs]
        else:
            self.independent_attrs = independent_attrs

        self.dependent_attr = dependent_attr
        self.measurement_error_sd = measurement_error_sd
        self.n = n
        self.title = title
        self.lambdas = np.array([0, 0.5, 1.0, 1.5, 2.0]) if lambdas is None else np.array(lambdas)
        self.n_simulations = n_simulations
        self.extrapolation_method = extrapolation_method
        
        self.model = None
        self.results = None  # Naive results (without correction)
        self.simex_results = None  # Corrected results
        self.formula = None
        self.simulation_results = []  # Store results for each lambda
        self.diagnostics = None  # Diagnostic tests
        self.cv_results = None  # Cross-validation results
        
    def _build_formula(self):
        """Build the regression formula."""
        self.formula = f"{self.dependent_attr} ~ {' + '.join(self.independent_attrs)}"
        return self.formula

    def fit(self, df):
        """
        Fit the SIMEX model: run simulations and extrapolate to get corrected estimates.

        Args:
            df (pd.DataFrame): DataFrame containing the data

        Returns:
            self: Returns self for method chaining
        """
        # Drop missing values
        required_cols = self.independent_attrs + [self.dependent_attr]
        df_clean = df[required_cols].dropna().reset_index(drop=True)
        
        # Get measurement error SD for each observation
        if np.isscalar(self.measurement_error_sd):
            error_sd = np.full(len(df_clean), self.measurement_error_sd)
        else:
            error_sd = np.array(self.measurement_error_sd)[:len(df_clean)]
        
        # Build formula
        formula = self._build_formula()
        
        # Fit naive model (lambda=0, no added error)
        print(f"Fitting naive OLS model...")
        self.model = smf.ols(formula, data=df_clean)
        self.results = self.model.fit()
        
        print(f"\nRunning SIMEX simulations...")
        print(f"Lambdas: {self.lambdas}")
        print(f"Simulations per lambda: {self.n_simulations}")
        
        # Storage for simulation results
        self.simulation_results = []
        
        # Run SIMEX simulation step
        for lam in self.lambdas:
            print(f"\nLambda = {lam}:")
            
            lambda_coefs = []
            lambda_params = {param: [] for param in self.results.params.index}
            
            for sim in range(self.n_simulations):
                # Add measurement error scaled by lambda
                added_error = np.random.normal(0, error_sd * np.sqrt(lam), size=len(df_clean))
                df_contaminated = df_clean.copy()
                df_contaminated[self.dependent_attr] = df_clean[self.dependent_attr] + added_error
                
                # Fit model with contaminated data
                model_sim = smf.ols(formula, data=df_contaminated)
                result_sim = model_sim.fit()
                
                # Store coefficients
                for param in result_sim.params.index:
                    lambda_params[param].append(result_sim.params[param])
            
            # Average coefficients across simulations
            avg_params = {param: np.mean(values) for param, values in lambda_params.items()}
            
            self.simulation_results.append({
                'lambda': lam,
                'params': avg_params,
                'params_std': {param: np.std(values) for param, values in lambda_params.items()}
            })
            
            print(f"  Average coefficients: {avg_params}")
        
        # Extrapolation step
        print(f"\nExtrapolating to lambda = -1 using {self.extrapolation_method} method...")
        self.simex_results = self._extrapolate()
        
        # Run diagnostics on the naive model
        print("\nRunning diagnostics on naive model...")
        try:
            self.diagnostics = self._run_diagnostic_tests(df_clean)
        except Exception as e:
            self.diagnostics = {'error': str(e)}
        
        print("\n" + "="*60)
        print("SIMEX Correction Complete")
        print("="*60)
        print("\nNaive OLS coefficients:")
        for param, value in self.results.params.items():
            print(f"  {param}: {value:.6f}")
        
        print("\nSIMEX-corrected coefficients:")
        for param, value in self.simex_results['corrected_params'].items():
            print(f"  {param}: {value:.6f}")
        
        print("\nCorrection factors:")
        for param in self.results.params.index:
            naive_val = self.results.params[param]
            corrected_val = self.simex_results['corrected_params'][param]
            if naive_val != 0:
                correction_pct = ((corrected_val - naive_val) / naive_val) * 100
                print(f"  {param}: {correction_pct:+.2f}%")
        print("="*60)
        
        return self

    def _extrapolate(self):
        """
        Extrapolate simulation results to lambda = -1 to get corrected estimates.
        
        Returns:
            dict: Corrected parameters and extrapolation details
        """
        lambdas_array = np.array([r['lambda'] for r in self.simulation_results])
        
        corrected_params = {}
        extrapolation_info = {}
        
        # Get all parameter names
        param_names = list(self.simulation_results[0]['params'].keys())
        
        for param in param_names:
            # Extract parameter values across lambdas
            param_values = np.array([r['params'][param] for r in self.simulation_results])
            
            if self.extrapolation_method == 'quadratic':
                # Fit quadratic: param = a + b*lambda + c*lambda^2
                poly_coeffs = np.polyfit(lambdas_array, param_values, deg=2)
                corrected_value = np.polyval(poly_coeffs, -1)
                
                extrapolation_info[param] = {
                    'method': 'quadratic',
                    'coefficients': poly_coeffs.tolist(),
                    'lambda_values': lambdas_array.tolist(),
                    'param_values': param_values.tolist()
                }
                
            elif self.extrapolation_method == 'linear':
                # Fit linear: param = a + b*lambda
                poly_coeffs = np.polyfit(lambdas_array, param_values, deg=1)
                corrected_value = np.polyval(poly_coeffs, -1)
                
                extrapolation_info[param] = {
                    'method': 'linear',
                    'coefficients': poly_coeffs.tolist(),
                    'lambda_values': lambdas_array.tolist(),
                    'param_values': param_values.tolist()
                }
                
            else:  # nonlinear (exponential form)
                # Fit: param = a + b * exp(c * lambda)
                from scipy.optimize import curve_fit
                
                def exp_func(x, a, b, c):
                    return a + b * np.exp(c * x)
                
                try:
                    popt, _ = curve_fit(exp_func, lambdas_array, param_values, 
                                       p0=[param_values[0], param_values[-1]-param_values[0], 1])
                    corrected_value = exp_func(-1, *popt)
                    
                    extrapolation_info[param] = {
                        'method': 'nonlinear',
                        'coefficients': popt.tolist(),
                        'lambda_values': lambdas_array.tolist(),
                        'param_values': param_values.tolist()
                    }
                except:
                    # Fallback to quadratic if nonlinear fails
                    poly_coeffs = np.polyfit(lambdas_array, param_values, deg=2)
                    corrected_value = np.polyval(poly_coeffs, -1)
                    
                    extrapolation_info[param] = {
                        'method': 'quadratic_fallback',
                        'coefficients': poly_coeffs.tolist(),
                        'lambda_values': lambdas_array.tolist(),
                        'param_values': param_values.tolist()
                    }
            
            corrected_params[param] = float(corrected_value)
        
        return {
            'corrected_params': corrected_params,
            'extrapolation_info': extrapolation_info,
            'naive_params': {k: float(v) for k, v in self.results.params.items()}
        }

    def _run_diagnostic_tests(self, df_clean):
        """
        Run common OLS diagnostic tests on the naive model.
        Same diagnostics as OLSModel.

        Returns:
            dict: Results of assumption tests and influence diagnostics
        """
        if self.results is None:
            raise ValueError("Model must be fitted before diagnostics")

        resid = self.results.resid
        exog = self.results.model.exog
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
                'cooks_distance_array': cooks_d.tolist(),
                'leverage_array': leverage.tolist(),
                'studentized_resid_array': student_resid.tolist()
            }
        except Exception as e:
            diagnostics['influence'] = {'error': str(e)}

        return diagnostics

    def print_diagnostics(self, show_arrays=False):
        """
        Print stored diagnostics from the naive model.

        Args:
            show_arrays (bool): If True, prints the full arrays for cooks_distance, leverage, etc.
        """
        if self.diagnostics is None:
            raise ValueError("Diagnostics are not available. Fit the model first.")

        d = self.diagnostics

        print("\n" + "=" * 60)
        print("Diagnostics Summary (Naive Model):")
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

    def cross_validate(self, df, k=5, random_state=42, use_corrected=True):
        """
        Perform k-fold cross-validation with SIMEX correction.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            k (int or str): Number of folds. Use 'loo' for leave-one-out CV
            random_state (int): Random state for reproducibility
            use_corrected (bool): If True, use SIMEX-corrected coefficients for predictions

        Returns:
            dict: Dictionary containing cross-validation results
        """
        # Drop missing values
        required_cols = self.independent_attrs + [self.dependent_attr]
        df_clean = df[required_cols].dropna().reset_index(drop=True)

        # Get measurement error SD for each observation
        if np.isscalar(self.measurement_error_sd):
            error_sd = np.full(len(df_clean), self.measurement_error_sd)
        else:
            error_sd = np.array(self.measurement_error_sd)[:len(df_clean)]

        # Set up cross-validation
        if k == 'loo':
            cv = LeaveOneOut()
            n_splits = len(df_clean)
            print(f"Performing Leave-One-Out CV with {n_splits} folds...")
        else:
            cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
            n_splits = k
            print(f"Performing {k}-Fold Cross-Validation with SIMEX...")

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
            train_error_sd = error_sd[train_idx]

            if use_corrected:
                # Fit SIMEX model on training data
                simex_fold = SIMEXModel(
                    independent_attrs=self.independent_attrs,
                    dependent_attr=self.dependent_attr,
                    measurement_error_sd=train_error_sd,
                    n=self.n,
                    title=self.title,
                    lambdas=self.lambdas,
                    n_simulations=self.n_simulations,
                    extrapolation_method=self.extrapolation_method
                )
                simex_fold.fit(train_df)
                y_pred = simex_fold.predict(test_df, use_corrected=True)
            else:
                # Fit naive model on training data
                model = smf.ols(formula, data=train_df)
                result = model.fit()
                y_pred = result.predict(test_df)

            y_true = test_df[self.dependent_attr]

            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            fold_metrics['r2'].append(r2)
            fold_metrics['mae'].append(mae)
            fold_metrics['rmse'].append(rmse)

            if k != 'loo' or fold_idx < 5:
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
            'n_splits': n_splits,
            'corrected': use_corrected
        }

        print("\n" + "=" * 60)
        print(f"Cross-Validation Summary ({'SIMEX-Corrected' if use_corrected else 'Naive'}):")
        print("=" * 60)
        print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
        print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
        print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
        print("=" * 60)

        return self.cv_results

    def fit_with_cv(self, df, k=5, random_state=42, use_corrected=True):
        """
        Perform cross-validation and then fit on the full dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            k (int or str): Number of folds. Use 'loo' for leave-one-out CV
            random_state (int): Random state for reproducibility
            use_corrected (bool): If True, use SIMEX-corrected coefficients in CV

        Returns:
            self: Returns self for method chaining
        """
        # Perform cross-validation
        self.cross_validate(df, k=k, random_state=random_state, use_corrected=use_corrected)

        # Fit on full dataset
        print("\nFitting SIMEX model on full dataset...")
        self.fit(df)

        return self

    def predict(self, df, use_corrected=True):
        """
        Make predictions using either naive or corrected coefficients.

        Args:
            df (pd.DataFrame): DataFrame containing the independent variables
            use_corrected (bool): If True, use SIMEX-corrected coefficients

        Returns:
            np.ndarray: Predicted values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if use_corrected:
            if self.simex_results is None:
                raise ValueError("SIMEX correction not performed")
            
            # Manual prediction using corrected coefficients
            X = df[self.independent_attrs].copy()
            X['Intercept'] = 1
            
            # Reorder to match coefficient order
            param_names = list(self.simex_results['corrected_params'].keys())
            X = X[param_names]
            
            coeffs = np.array([self.simex_results['corrected_params'][p] for p in param_names])
            predictions = X.values @ coeffs
            
            return predictions
        else:
            return self.results.predict(df)

    def evaluate(self, df, use_corrected=True):
        """
        Evaluate the model using either naive or corrected coefficients.

        Args:
            df (pd.DataFrame): DataFrame containing both independent and dependent variables
            use_corrected (bool): If True, use SIMEX-corrected coefficients

        Returns:
            dict: Dictionary containing R², MAE, and RMSE
        """
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")

        y_pred = self.predict(df, use_corrected=use_corrected)
        y_true = df[self.dependent_attr].dropna()

        # Ensure same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'corrected': use_corrected
        }

        return metrics

    def summary(self):
        """Print comparison of naive and corrected results."""
        if self.results is None:
            raise ValueError("Model must be fitted before viewing summary")

        print("\n" + "="*60)
        print("NAIVE OLS RESULTS (without SIMEX correction)")
        print("="*60)
        print(self.results.summary())
        
        if self.simex_results is not None:
            print("\n" + "="*60)
            print("SIMEX-CORRECTED COEFFICIENTS")
            print("="*60)
            
            print(f"\n{'Parameter':<20} {'Naive':<15} {'Corrected':<15} {'Change %':<10}")
            print("-" * 60)
            
            for param in self.results.params.index:
                naive_val = self.results.params[param]
                corrected_val = self.simex_results['corrected_params'][param]
                change_pct = ((corrected_val - naive_val) / naive_val * 100) if naive_val != 0 else 0
                
                print(f"{param:<20} {naive_val:>14.6f} {corrected_val:>14.6f} {change_pct:>9.2f}%")
        
        # Print CV results if available
        if self.cv_results is not None:
            print("\n" + "=" * 60)
            print(f"Cross-Validation Results ({'SIMEX-Corrected' if self.cv_results.get('corrected', True) else 'Naive'}):")
            print("=" * 60)
            print(f"R²:   {self.cv_results['mean_r2']:.4f} ± {self.cv_results['std_r2']:.4f}")
            print(f"MAE:  {self.cv_results['mean_mae']:.4f} ± {self.cv_results['std_mae']:.4f}")
            print(f"RMSE: {self.cv_results['mean_rmse']:.4f} ± {self.cv_results['std_rmse']:.4f}")
            print("=" * 60)

    def plot_extrapolation(self, save=True):
        """
        Plot the extrapolation curves for each parameter.

        Args:
            save (bool): If True, saves plot to model_results/<title>/<n>_simex_extrapolation.png
        """
        if self.simex_results is None:
            raise ValueError("Model must be fitted before plotting")

        param_names = list(self.simex_results['corrected_params'].keys())
        n_params = len(param_names)
        
        # Create subplots
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
        if n_params == 1:
            axes = [axes]
        
        for idx, param in enumerate(param_names):
            ax = axes[idx]
            info = self.simex_results['extrapolation_info'][param]
            
            lambdas = np.array(info['lambda_values'])
            param_values = np.array(info['param_values'])
            
            # Plot simulation points
            ax.scatter(lambdas, param_values, color='blue', s=100, 
                      label='Simulation results', zorder=3)
            
            # Plot extrapolation curve
            lambda_range = np.linspace(-1, max(lambdas), 100)
            
            if info['method'] in ['quadratic', 'quadratic_fallback']:
                coeffs = info['coefficients']
                curve_values = np.polyval(coeffs, lambda_range)
            elif info['method'] == 'linear':
                coeffs = info['coefficients']
                curve_values = np.polyval(coeffs, lambda_range)
            else:  # nonlinear
                coeffs = info['coefficients']
                curve_values = coeffs[0] + coeffs[1] * np.exp(coeffs[2] * lambda_range)
            
            ax.plot(lambda_range, curve_values, 'r--', 
                   label=f'{info["method"].capitalize()} extrapolation')
            
            # Mark corrected value at lambda=-1
            corrected_val = self.simex_results['corrected_params'][param]
            ax.scatter([-1], [corrected_val], color='red', s=200, marker='*',
                      label=f'Corrected (λ=-1): {corrected_val:.4f}', zorder=4)
            
            # Mark naive value at lambda=0
            naive_val = param_values[lambdas == 0][0]
            ax.scatter([0], [naive_val], color='green', s=150, marker='s',
                      label=f'Naive (λ=0): {naive_val:.4f}', zorder=4)
            
            ax.axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Lambda (λ)')
            ax.set_ylabel(f'{param}')
            ax.set_title(f'SIMEX Extrapolation: {param}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            results_dir = os.path.join('model_results', self.title)
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'{self.n}_simex_extrapolation.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Extrapolation plot saved to {plot_path}")
        
        return fig

    def plot(self, df, save=True):
        """
        Plot actual vs predicted values for both naive and corrected models.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            save (bool): If True, saves plot
        """
        if self.results is None:
            raise ValueError("Model must be fitted before plotting")

        # Get predictions
        y_pred_naive = self.predict(df, use_corrected=False)
        y_pred_corrected = self.predict(df, use_corrected=True)
        y_true = df[self.dependent_attr].dropna()

        # Ensure same length
        min_len = min(len(y_pred_naive), len(y_true))
        y_pred_naive = y_pred_naive[:min_len]
        y_pred_corrected = y_pred_corrected[:min_len]
        y_true = y_true[:min_len]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Naive model plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred_naive, color='blue', alpha=0.5, label='Predictions')
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', label='Perfect fit')
        
        metrics_naive = self.evaluate(df, use_corrected=False)
        ax1.text(0.05, 0.95,
                f"R²: {metrics_naive['r2']:.3f}\n"
                f"MAE: {metrics_naive['mae']:.3f}\n"
                f"RMSE: {metrics_naive['rmse']:.3f}",
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'Naive OLS (No Correction)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # SIMEX-corrected model plot
        ax2 = axes[1]
        ax2.scatter(y_true, y_pred_corrected, color='green', alpha=0.5, label='Predictions')
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', label='Perfect fit')
        
        metrics_corrected = self.evaluate(df, use_corrected=True)
        ax2.text(0.05, 0.95,
                f"R²: {metrics_corrected['r2']:.3f}\n"
                f"MAE: {metrics_corrected['mae']:.3f}\n"
                f"RMSE: {metrics_corrected['rmse']:.3f}",
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'SIMEX-Corrected Model')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{self.title}: {self.dependent_attr}', fontsize=14, y=1.02)
        plt.tight_layout()

        if save:
            results_dir = os.path.join('model_results', self.title)
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'{self.n}_simex_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {plot_path}")

        return fig

    def save_results(self):
        """Save SIMEX model results to JSON file."""
        if self.results is None:
            raise ValueError("Model must be fitted before saving results")

        results_dir = os.path.join('model_results', self.title)
        os.makedirs(results_dir, exist_ok=True)

        results_dict = {
            'title': self.title,
            'n': self.n,
            'formula': self.formula,
            'independent_variables': self.independent_attrs,
            'dependent_variable': self.dependent_attr,
            'measurement_error_sd': float(self.measurement_error_sd) if np.isscalar(self.measurement_error_sd) else 'heteroscedastic',
            'simex_parameters': {
                'lambdas': self.lambdas.tolist(),
                'n_simulations': self.n_simulations,
                'extrapolation_method': self.extrapolation_method
            },
            'naive_results': {
                'r_squared': float(self.results.rsquared),
                'adj_r_squared': float(self.results.rsquared_adj),
                'f_statistic': float(self.results.fvalue),
                'f_pvalue': float(self.results.f_pvalue),
                'aic': float(self.results.aic),
                'bic': float(self.results.bic),
                'n_observations': int(self.results.nobs),
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
                }
            },
            'simex_results': {
                'corrected_coefficients': self.simex_results['corrected_params'],
                'extrapolation_info': self.simex_results['extrapolation_info']
            },
            'correction_summary': {
                param: {
                    'naive': float(self.results.params[param]),
                    'corrected': float(self.simex_results['corrected_params'][param]),
                    'change_percent': float(((self.simex_results['corrected_params'][param] - self.results.params[param]) / self.results.params[param] * 100) if self.results.params[param] != 0 else 0)
                }
                for param in self.results.params.index
            },
            'diagnostics': self.diagnostics,
            'cross_validation': self.cv_results if self.cv_results else None,
            'simulation_results': self.simulation_results
        }

        json_path = os.path.join(results_dir, f'{self.n}_simex_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nSIMEX results saved to {json_path}")
        return json_path

    def get_coefficients(self, corrected=True):
        """
        Get model coefficients.

        Args:
            corrected (bool): If True, return SIMEX-corrected coefficients

        Returns:
            dict: Model coefficients
        """
        if self.results is None:
            raise ValueError("Model must be fitted before accessing coefficients")

        if corrected and self.simex_results is not None:
            return self.simex_results['corrected_params']
        else:
            return {k: float(v) for k, v in self.results.params.items()}
