import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
import matplotlib.pyplot as plt
import warnings
import os


class KalmanSmoother:
    """
    Kalman filter/smoother for removing measurement noise from panel data.
    
    This class applies Kalman filtering and smoothing to remove measurement noise
    from observations grouped by entities (e.g., individual cows, patients, machines).
    
    It uses a local level model:
        State equation: true_value_t = true_value_{t-1} + process_noise
        Observation equation: measured_value_t = true_value_t + measurement_noise
    
    Attributes:
        measurement_noise: Expected measurement error variance (e.g., 20^2 for ±20kg)
        process_noise: How much true value can change between measurements (auto-estimated if None)
        entity_results: Dict storing fitted models for each entity
    """
    
    def __init__(self, measurement_noise: float = None, process_noise: float = None, 
                 fix_measurement_noise: bool = False, use_trend: bool = False):
        """
        Initialize the Kalman smoother.
        
        Args:
            measurement_noise: Expected measurement error variance. 
                             E.g., if measurements vary ±20kg, use 400 (20^2)
                             If None, will be estimated from data
            process_noise: Process noise variance (how much true value drifts).
                          If None, will be estimated from data (recommended)
            fix_measurement_noise: If True, measurement_noise is fixed and not estimated.
                                  Only works if measurement_noise is provided.
            use_trend: If True, uses local linear trend model (for growing animals).
                      If False, uses local level model (for stable values).
        """
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.fix_measurement_noise = fix_measurement_noise
        self.use_trend = use_trend
        self.entity_results = {}
        self.fitted_ = False
    
    def filter(self, df: pd.DataFrame, 
               target_attr: str,
               group_attr: str = None,
               time_attr: str = None,
               inplace: bool = False,
               print_results: bool = True
               ) -> pd.DataFrame:
        """
        Apply Kalman filtering/smoothing to remove measurement noise.
        
        Args:
            df: DataFrame containing the data
            target_attr: Name of the column to smooth (e.g., 'weight')
            group_attr: Name of the grouping column (e.g., 'cow_id'). 
                       If None, treats all data as one entity
            time_attr: Name of the time/sequence column (e.g., 'date').
                      If None, assumes data is already sorted within groups
            inplace: If True, modifies df in place and returns it. If False, returns a copy.
        
        Returns:
            DataFrame with added columns:
                - {target_attr}_filtered: Kalman filtered values
                - {target_attr}_smoothed: Kalman smoothed values (better estimates)
                - {target_attr}_smoothed_se: Standard error of smoothed estimates
        """
        if inplace:
            df_result = df
        else:
            df_result = df.copy()
        
        # If no grouping, treat as single entity
        if group_attr is None:
            print("No group_attr specified, treating all data as one entity...")
            df_result['_temp_group'] = 'all'
            group_attr = '_temp_group'
            temp_group = True
        else:
            temp_group = False
        
        # Sort by group and time if time_attr provided
        if time_attr is not None:
            df_result = df_result.sort_values([group_attr, time_attr]).reset_index(drop=True)
        
        # Initialize output columns
        filtered_col = f'{target_attr}_filtered'
        smoothed_col = f'{target_attr}_smoothed'
        se_col = f'{target_attr}_smoothed_se'
        
        df_result[filtered_col] = np.nan
        df_result[smoothed_col] = np.nan
        df_result[se_col] = np.nan
        
        # Get unique entities
        entities = df_result[group_attr].unique()
        print(f"Applying Kalman filter to {len(entities)} entities...")
        
        # Process each entity separately
        for entity in entities:
            entity_mask = df_result[group_attr] == entity
            entity_df = df_result[entity_mask].copy()
            
            # Extract observations
            observations = entity_df[target_attr].values
            
            # Extract time index if available
            time_index = None
            if time_attr is not None:
                time_index = pd.to_datetime(entity_df[time_attr])
            
            # Skip if all NaN
            if np.all(np.isnan(observations)):
                print(f"  Entity {entity}: All NaN, skipping...")
                continue
            
            # Skip if too few observations
            if np.sum(~np.isnan(observations)) < 3:
                print(f"  Entity {entity}: Too few observations ({np.sum(~np.isnan(observations))}), skipping...")
                continue
            
            try:
                # Fit Kalman filter/smoother with time information
                filtered, smoothed, smoothed_se = self._fit_entity(
                    observations, 
                    entity_id=entity,
                    time_index=time_index  # Pass time information!
                )
                
                # Store results
                df_result.loc[entity_mask, filtered_col] = filtered
                df_result.loc[entity_mask, smoothed_col] = smoothed
                df_result.loc[entity_mask, se_col] = smoothed_se
                
            except Exception as e:
                print(f"  Entity {entity}: Failed with error: {e}")
                continue
        
        # Remove temporary group column if created
        if temp_group:
            df_result = df_result.drop(columns=['_temp_group'])
        
        self.fitted_ = True
        
        # Print summary
        n_smoothed = df_result[smoothed_col].notna().sum()
        print(f"\nSuccessfully smoothed {n_smoothed}/{len(df_result)} observations")
       
        if print_results == True:
            self.print_summary()

        return df_result

    def print_summary(self):
        """Print summary of fitted models across all entities."""
        if not self.fitted_:
            raise ValueError("Model not fitted. Run filter() first.")
        
        if not self.entity_results:
            print("No entity results stored.")
            return
        
        print("\n" + "=" * 70)
        print("Kalman Smoother Summary")
        print("=" * 70)
        print(f"Number of entities processed: {len(self.entity_results)}")
        
        # Collect statistics, handling zeros and NaNs
        obs_vars = [r['sigma2_obs'] for r in self.entity_results.values() if not np.isnan(r['sigma2_obs'])]
        state_vars = [r['sigma2_state'] for r in self.entity_results.values() if not np.isnan(r['sigma2_state'])]
        aics = [r['aic'] for r in self.entity_results.values() if not np.isnan(r['aic'])]
        
        # Filter out zeros from obs_vars to avoid division by zero
        clean_obs_vars = [v for v in obs_vars if v > 1e-6]
        clean_state_vars = [s for s, o in zip(state_vars, obs_vars) if o > 1e-6]
        
        print(f"\nMeasurement noise variance (σ²_obs):")
        print(f"  Mean: {np.mean(obs_vars):.4f}")
        print(f"  Std:  {np.std(obs_vars):.4f}")
        print(f"  Min:  {np.min(obs_vars):.4f}")
        print(f"  Max:  {np.max(obs_vars):.4f}")
        
        print(f"\nProcess noise variance (σ²_state):")
        print(f"  Mean: {np.mean(state_vars):.4f}")
        print(f"  Std:  {np.std(state_vars):.4f}")
        print(f"  Min:  {np.min(state_vars):.4f}")
        print(f"  Max:  {np.max(state_vars):.4f}")
        
        print(f"\nSignal-to-noise ratio (σ²_state / σ²_obs):")
        if clean_obs_vars:
            snr = [s/o for s, o in zip(clean_state_vars, clean_obs_vars)]
            median_snr = np.median(snr)
            print(f"  Median: {median_snr:.4f}")
            print(f"  25th percentile: {np.percentile(snr, 25):.4f}")
            print(f"  75th percentile: {np.percentile(snr, 75):.4f}")
        else:
            print("  Could not calculate (zero or undefined observation variance)")
        print(f"  (Higher = more true variation vs noise)")
        
        print(f"\nModel fit (AIC):")
        print(f"  Mean: {np.mean(aics):.2f}")
        print(f"  Min:  {np.min(aics):.2f}")
        print(f"  Max:  {np.max(aics):.2f}")
        print("=" * 70)

    def _fit_entity(self, observations, entity_id=None, time_index=None):
        """
        Fit Kalman filter/smoother for a single entity.
        
        Args:
            observations: Array of observations (can contain NaN)
            entity_id: Identifier for this entity (for storing results)
            time_index: DatetimeIndex for irregular time spacing (IMPORTANT!)
        
        Returns:
            filtered: Filtered state estimates
            smoothed: Smoothed state estimates
            smoothed_se: Standard errors of smoothed estimates
        """
        # CRITICAL: Convert to pandas Series with DatetimeIndex if time_index provided
        # This allows statsmodels to handle irregular time spacing automatically
        if time_index is not None:
            endog = pd.Series(observations, index=time_index)
        else:
            endog = observations
        
        # Build the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create model with fixed or free parameters
            if self.fix_measurement_noise and self.measurement_noise is not None:
                # Fix observation variance, only estimate state variance
                model = UnobservedComponents(
                    endog=endog,  # Now includes time information!
                    level='local linear trend' if self.use_trend else 'local level',
                    irregular=True,
                    stochastic_level=True,
                    stochastic_trend=self.use_trend
                )
                
                # Set starting parameters
                state_var = self.process_noise if self.process_noise is not None else np.nanvar(observations) * 0.1
                
                if self.use_trend:
                    start_params = [self.measurement_noise, state_var, state_var * 0.1]
                else:
                    start_params = [self.measurement_noise, state_var]
                
                # Fit with fixed observation variance
                try:
                    from scipy.optimize import minimize
                    
                    def neg_loglike(params):
                        """Negative log-likelihood with fixed obs variance."""
                        if self.use_trend:
                            full_params = np.array([self.measurement_noise, params[0], params[1]])
                        else:
                            full_params = np.array([self.measurement_noise, params[0]])
                        
                        model.update(full_params)
                        return -model.loglike(full_params)
                    
                    # Optimize state variance(s)
                    if self.use_trend:
                        result = minimize(
                            neg_loglike,
                            x0=[state_var, state_var * 0.1],
                            method='L-BFGS-B',
                            bounds=[(1e-8, None), (1e-8, None)]
                        )
                        optimal_params = result.x
                        final_params = np.array([self.measurement_noise, optimal_params[0], optimal_params[1]])
                    else:
                        result = minimize(
                            neg_loglike,
                            x0=[state_var],
                            method='L-BFGS-B',
                            bounds=[(1e-8, None)]
                        )
                        final_params = np.array([self.measurement_noise, result.x[0]])
                    
                    model.update(final_params)
                    
                    # Run smoother with final parameters
                    results = model.smooth(final_params)
                    
                    # Add params attribute for consistency
                    results.params = final_params
                    n_params = len(final_params)
                    results.aic = -2 * (-result.fun) + 2 * n_params
                    results.bic = -2 * (-result.fun) + np.log(len(observations)) * n_params
                    
                except Exception as e:
                    print(f"    Warning: Fixed variance optimization failed: {e}, falling back to standard MLE")
                    results = model.fit(start_params=start_params, disp=False, maxiter=1000)
            else:
                # Standard model - estimate both variances
                model = UnobservedComponents(
                    endog=endog,  # Now includes time information!
                    level='local linear trend' if self.use_trend else 'local level',
                    irregular=True,
                    stochastic_level=True,
                    stochastic_trend=self.use_trend
                )
                
                # Set starting parameters if provided
                start_params = None
                if self.measurement_noise is not None or self.process_noise is not None:
                    obs_var = self.measurement_noise if self.measurement_noise is not None else np.nanvar(observations) * 0.5
                    state_var = self.process_noise if self.process_noise is not None else np.nanvar(observations) * 0.1
                    
                    if self.use_trend:
                        start_params = [obs_var, state_var, state_var * 0.1]
                    else:
                        start_params = [obs_var, state_var]
                
                # Fit the model
                try:
                    if start_params is not None:
                        results = model.fit(start_params=start_params, disp=False, maxiter=1000)
                    else:
                        results = model.fit(disp=False, maxiter=1000)
                except:
                    # If MLE fails, try with simpler method
                    results = model.fit(method='nm', disp=False, maxiter=500)
            
            # Extract results
            filtered_states = results.filtered_state[0, :]  # Level is first state
            smoothed_states = results.smoothed_state[0, :]
            smoothed_state_cov = results.smoothed_state_cov[0, 0, :]  # Variance of level
            smoothed_se = np.sqrt(smoothed_state_cov)
            
            # Store the fitted model
            if entity_id is not None:
                sigma2_obs = np.nan
                sigma2_state = np.nan
                
                try:
                    if len(results.params) >= 2:
                        sigma2_obs = float(results.params[0])
                        sigma2_state = float(results.params[1])
                    elif len(results.params) == 1:
                        sigma2_obs = float(results.params[0])
                        sigma2_state = 0.0
                except Exception as e:
                    print(f"    Warning: Could not extract variances: {e}")
                
                self.entity_results[entity_id] = {
                    'model': model,
                    'results': results,
                    'sigma2_obs': sigma2_obs,
                    'sigma2_state': sigma2_state,
                    'aic': float(results.aic) if hasattr(results, 'aic') else np.nan,
                    'bic': float(results.bic) if hasattr(results, 'bic') else np.nan
                }
        
        return filtered_states, smoothed_states, smoothed_se
    
    def get_entity_diagnostics(self, entity_id):
        """
        Get diagnostics for a specific entity.
        
        Args:
            entity_id: The entity identifier
        
        Returns:
            dict: Dictionary with model diagnostics
        """
        if entity_id not in self.entity_results:
            raise ValueError(f"Entity {entity_id} not found. Did you run filter()?")
        
        return self.entity_results[entity_id]
    
   
    def plot_entity(self, df, entity_id, target_attr, group_attr, time_attr=None, 
                   save=False, save_path=None):
        """
        Plot original observations vs smoothed estimates for a specific entity.
        
        Args:
            df: DataFrame with smoothed results
            entity_id: Entity to plot
            target_attr: Name of target column
            group_attr: Name of group column
            time_attr: Name of time column (optional, uses index if None)
            save: Whether to save the plot
            save_path: Path to save plot to
        """
        entity_df = df[df[group_attr] == entity_id].copy()
        
        if len(entity_df) == 0:
            raise ValueError(f"Entity {entity_id} not found in dataframe")
        
        # Get time values
        if time_attr is not None:
            time_vals = pd.to_datetime(entity_df[time_attr])
            xlabel = time_attr
        else:
            time_vals = np.arange(len(entity_df))
            xlabel = 'Index'
        
        # Get data
        observations = entity_df[target_attr].values
        smoothed = entity_df[f'{target_attr}_smoothed'].values
        smoothed_se = entity_df[f'{target_attr}_smoothed_se'].values
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Observations
        ax.scatter(time_vals, observations, alpha=0.5, s=50, 
                  label='Observations', color='gray', zorder=1)
        
        # Smoothed estimate
        ax.plot(time_vals, smoothed, 'r-', linewidth=2, 
               label='Kalman smoothed', zorder=2)
        
        # Confidence interval
        ax.fill_between(time_vals, 
                       smoothed - 2*smoothed_se, 
                       smoothed + 2*smoothed_se,
                       alpha=0.2, color='red', 
                       label='95% confidence interval', zorder=0)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(target_attr)
        ax.set_title(f'Kalman Smoothing for Entity: {entity_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add diagnostics text
        if entity_id in self.entity_results:
            diag = self.entity_results[entity_id]
            text = f"σ²_obs: {diag['sigma2_obs']:.2f}\n"
            text += f"σ²_state: {diag['sigma2_state']:.2f}\n"
            text += f"AIC: {diag['aic']:.1f}"
            
            ax.text(0.02, 0.98, text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save and save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
    
    def plot_comparison(self, df, target_attr, group_attr, time_attr=None, 
                       max_entities=6, save=False, save_path=None):
        """
        Plot comparison of original vs smoothed for multiple entities.
        
        Args:
            df: DataFrame with smoothed results
            target_attr: Name of target column
            group_attr: Name of group column
            time_attr: Name of time column (optional)
            max_entities: Maximum number of entities to plot
            save: Whether to save the plot
            save_path: Path to save plot to
        """
        entities = df[group_attr].unique()[:max_entities]
        n_entities = len(entities)
        
        fig, axes = plt.subplots(n_entities, 1, figsize=(12, 3*n_entities))
        if n_entities == 1:
            axes = [axes]
        
        for idx, entity_id in enumerate(entities):
            ax = axes[idx]
            entity_df = df[df[group_attr] == entity_id].copy()
            
            # Get time values
            if time_attr is not None:
                time_vals = pd.to_datetime(entity_df[time_attr])
            else:
                time_vals = np.arange(len(entity_df))
            
            # Get data
            observations = entity_df[target_attr].values
            smoothed = entity_df[f'{target_attr}_smoothed'].values
            
            # Plot
            ax.scatter(time_vals, observations, alpha=0.5, s=30, 
                      label='Observations', color='gray')
            ax.plot(time_vals, smoothed, 'r-', linewidth=2, 
                   label='Smoothed')
            
            ax.set_ylabel(target_attr)
            ax.set_title(f'Entity: {entity_id}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            if idx == n_entities - 1:
                ax.set_xlabel(time_attr if time_attr else 'Index')
        
        plt.tight_layout()
        
        if save and save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
