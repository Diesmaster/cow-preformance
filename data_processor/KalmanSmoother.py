import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Tuple, Dict, List
import logging


class KalmanSmoother:
    """
    Simple Kalman smoother to remove measurement errors from cattle weights.
    
    Purpose: Clean noisy weight measurements before feeding into regression models.
    
    Model:
        - True weight follows a smooth trajectory (your regression will model this)
        - Observed weight = True weight + measurement_error
        - measurement_error ~ N(0, σ²) includes: scale error, gut fill, hydration
    
    This is NOT a growth model - it's a preprocessor to denoise measurements.
    Your regression model will learn the actual growth patterns from feed, 
    medical history, breed, etc.
    
    Example:
        >>> # Auto-tune the measurement noise
        >>> smoother = MeasurementErrorSmoother(auto_tune=True)
        >>> df_clean = smoother.smooth(df, 'weight', 'cow_id', 'date')
        >>> 
        >>> # Use cleaned weights in your regression
        >>> X = df_clean[['weight_filtered', 'feed_features', ...]]
        >>> y = df_clean['future_weight']
    """
    
    def __init__(self,
                 measurement_noise: Optional[float] = None,
                 auto_tune: bool = True,
                 tune_on_first_n: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize the smoother.
        
        Args:
            measurement_noise: Total measurement error variance (scale + gut + hydration).
                             Typical range: 400-600 (±20-25 kg)
                             If None and auto_tune=True, will be estimated from data.
            auto_tune: Estimate measurement_noise from data
            tune_on_first_n: Limit tuning to first N entities (for speed)
            verbose: Print progress
        """
        self.measurement_noise = measurement_noise
        self.auto_tune = auto_tune
        self.tune_on_first_n = tune_on_first_n
        self.verbose = verbose
        
        self.logger = self._setup_logger()
        self.entity_results = {}
        self.fitted_ = False
        self.tuning_result_ = None
    
    def _setup_logger(self):
        logger = logging.getLogger('MeasurementErrorSmoother')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    

    def smooth(self,
               df: pd.DataFrame,
               target_attr: str = 'weight',
               group_attr: Optional[str] = None,
               time_attr: Optional[str] = None,
               return_se: bool = True) -> pd.DataFrame:
        """
        Smooth noisy measurements to remove measurement errors.
        
        Args:
            df: Input DataFrame
            target_attr: Column to smooth (default: 'weight')
            group_attr: Grouping column (e.g., 'cow_id')
            time_attr: Time column (e.g., 'date')
            return_se: If True, also returns standard error columns (default: True)
        
        Returns:
            DataFrame with new columns:
                - '{target_attr}_filtered': Cleaned weight (causal - use for prediction!)
                - '{target_attr}_smoothed': Smoother estimate (non-causal - visualization only)
                - '{target_attr}_filtered_se': Standard error of filtered estimate (if return_se=True)
                - '{target_attr}_smoothed_se': Standard error of smoothed estimate (if return_se=True)
        
        Example:
            >>> df_clean = smoother.smooth(df, 'weight', 'cow_id', 'date')
            >>> # Use 'weight_filtered' as input to your regression model
            >>> X = df_clean[['weight_filtered', 'feed_intake', 'breed', ...]]
            >>> 
            >>> # Check uncertainty
            >>> high_uncertainty = df_clean['weight_filtered_se'] > 10
            >>> print(f"Found {high_uncertainty.sum()} measurements with high uncertainty")
        """
        if time_attr is None:
            raise ValueError("time_attr required for proper smoothing")
        
        df_result = df.copy()
        
        # Handle grouping
        temp_group = False
        if group_attr is None:
            df_result['_temp_group'] = 'all'
            group_attr = '_temp_group'
            temp_group = True
        
        # Sort
        df_result = df_result.sort_values([group_attr, time_attr]).reset_index(drop=True)
        
        # Auto-tune if needed
        if self.auto_tune and self.measurement_noise is None:
            self._tune_measurement_noise(df_result, target_attr, group_attr, time_attr)
        
        if self.measurement_noise is None:
            raise ValueError("measurement_noise not set. Set it manually or enable auto_tune=True")
        
        # Initialize output columns
        df_result[f'{target_attr}_filtered'] = np.nan
        df_result[f'{target_attr}_smoothed'] = np.nan
        
        if return_se:
            df_result[f'{target_attr}_filtered_se'] = np.nan
            df_result[f'{target_attr}_smoothed_se'] = np.nan
        
        # Store return_se flag for processing
        self._return_se = return_se
        
        # Process each entity
        self._process_all_entities(df_result, target_attr, group_attr, time_attr)
        
        # Clean up
        if temp_group:
            df_result = df_result.drop(columns=['_temp_group'])
        
        self.fitted_ = True
        self._print_summary()
        
        return df_result

    def _process_all_entities(self, df, target_attr, group_attr, time_attr):
        """Process all entities."""
        entities = df[group_attr].unique()
        
        self.logger.info(f"Smoothing {len(entities)} entities...")
        
        for idx, entity in enumerate(entities):
            if self.verbose and (idx + 1) % 10 == 0:
                self.logger.info(f"  Progress: {idx + 1}/{len(entities)}")
            
            entity_mask = df[group_attr] == entity
            entity_df = df[entity_mask].copy().reset_index(drop=True)
            
            obs = entity_df[target_attr].values
            time_idx = pd.to_datetime(entity_df[time_attr])
            
            if np.sum(~np.isnan(obs)) < 2:
                continue
            
            try:
                # Now returns SE if requested
                if self._return_se:
                    filtered, smoothed, filtered_se, smoothed_se = self._smooth_entity(
                        obs, time_idx, entity, return_se=True
                    )
                    df.loc[entity_mask, f'{target_attr}_filtered_se'] = filtered_se
                    df.loc[entity_mask, f'{target_attr}_smoothed_se'] = smoothed_se
                else:
                    filtered, smoothed = self._smooth_entity(
                        obs, time_idx, entity, return_se=False
                    )
                
                df.loc[entity_mask, f'{target_attr}_filtered'] = filtered
                df.loc[entity_mask, f'{target_attr}_smoothed'] = smoothed
                
            except Exception as e:
                self.logger.warning(f"Entity {entity} failed: {e}")

    def _smooth_entity(self, observations, time_index, entity_id, return_se=True):
        """
        Smooth a single entity's measurements with drift estimation.
        
        Model:
            True weight_t = baseline + drift × t + biological_variation
            Observed weight_t = True weight_t + measurement_error
        
        Args:
            observations: Array of observations
            time_index: Time index (pandas datetime)
            entity_id: Entity identifier
            return_se: Whether to return standard errors
        
        Returns:
            If return_se=True: (filtered, smoothed, filtered_se, smoothed_se)
            If return_se=False: (filtered, smoothed)
        """
        # ===================================================================
        # STEP 1: Estimate drift rate (growth rate) from data
        # ===================================================================
        valid_mask = ~np.isnan(observations)
        valid_obs = observations[valid_mask]
        valid_times = time_index[valid_mask]
        
        if len(valid_obs) >= 2:
            # Convert to days from first observation
            days = (valid_times - valid_times.iloc[0]).dt.days.values.astype(float)
            
            # Linear regression: weight = intercept + drift * days
            drift_rate, intercept = np.polyfit(days, valid_obs, 1)
        else:
            drift_rate = 0.0
            intercept = valid_obs[0] if len(valid_obs) > 0 else 0.0
        
        # ===================================================================
        # STEP 2: Initialize Kalman filter with control input
        # ===================================================================
        kf = KalmanFilter(dim_x=1, dim_z=1)
        
        # Initial state
        first_valid = np.where(~np.isnan(observations))[0][0]
        kf.x = np.array([[observations[first_valid]]])
        kf.P = np.array([[self.measurement_noise]])
        
        # Observation model: we observe the state directly
        kf.H = np.array([[1.0]])
        kf.R = np.array([[self.measurement_noise]])
        
        # State transition: identity (constant + drift)
        kf.F = np.array([[1.0]])
        kf.B = np.array([[1.0]])  # Control input matrix for drift
        
        # Process noise per day (biological variation around growth trend)
        # This should be much smaller than measurement noise since drift handles growth
        process_noise_per_day = self.measurement_noise * 0.05  # 5% of measurement noise
        
        # ===================================================================
        # STEP 3: Forward pass (filtering) with time-scaled drift and noise
        # ===================================================================
        filtered_states = []
        filtered_covs = []
        time_deltas = []
        prev_time = time_index.iloc[0]
        
        for i in range(len(observations)):
            # Calculate days since last measurement
            if i > 0:
                dt = (time_index.iloc[i] - prev_time).days
                dt = max(dt, 1)  # At least 1 day
            else:
                dt = 1
            
            time_deltas.append(dt)
            prev_time = time_index.iloc[i]
            
            # Predict with drift and time-scaled process noise
            if i > 0:
                # Control input: expected growth over dt days
                u = np.array([[drift_rate * dt]])
                
                # Process noise scales with time interval
                kf.Q = np.array([[process_noise_per_day * dt]])
                
                kf.predict(u=u)
            
            # Update with observation (if not NaN)
            if not np.isnan(observations[i]):
                kf.update(observations[i])
            
            # Store filtered estimate
            filtered_states.append(float(kf.x[0, 0]))
            
            if return_se:
                filtered_covs.append(float(kf.P[0, 0]))
        
        # ===================================================================
        # STEP 4: Backward pass (RTS smoothing)
        # ===================================================================
        if return_se:
            smoothed_states, smoothed_covs = self._rts_smooth_with_drift(
                filtered_states, filtered_covs, time_deltas, 
                drift_rate, process_noise_per_day
            )
            
            # Convert covariances to standard errors
            filtered_se = np.sqrt(np.array(filtered_covs))
            smoothed_se = np.sqrt(np.array(smoothed_covs))
        else:
            smoothed_states = self._rts_smooth_simple_with_drift(
                filtered_states, time_deltas, drift_rate, process_noise_per_day
            )
        
        # ===================================================================
        # STEP 5: Store diagnostics
        # ===================================================================
        self.entity_results[entity_id] = {
            'measurement_noise': self.measurement_noise,
            'drift_rate': drift_rate,  # kg/day
            'process_noise_per_day': process_noise_per_day,
            'n_observations': len(observations),
            'n_valid': np.sum(~np.isnan(observations)),
            'mean_change': np.nanmean(np.abs(observations - np.array(filtered_states))),
            'mean_filtered': np.nanmean(filtered_states),
            'mean_obs': np.nanmean(valid_obs)
        }
        
        if return_se:
            return filtered_states, smoothed_states, filtered_se, smoothed_se
        else:
            return filtered_states, smoothed_states


    def _rts_smooth_with_drift(self, filtered_means, filtered_covs, time_deltas,
                               drift_rate, process_noise_per_day):
        """
        RTS smoother with drift and time-scaled process noise.
        
        Args:
            filtered_means: Forward pass means
            filtered_covs: Forward pass covariances
            time_deltas: Time intervals between measurements (days)
            drift_rate: Estimated growth rate (kg/day)
            process_noise_per_day: Process noise variance per day
        
        Returns:
            (smoothed_means, smoothed_covs)
        """
        n = len(filtered_means)
        smoothed_means = filtered_means.copy()
        smoothed_covs = filtered_covs.copy()
        
        for i in range(n - 2, -1, -1):
            # Time interval to next measurement
            dt = max(time_deltas[i + 1], 1.0) if i + 1 < len(time_deltas) else 1.0
            
            # Predicted mean: current + drift over dt
            predicted_mean = filtered_means[i] + drift_rate * dt
            
            # Predicted covariance: current + process noise over dt
            predicted_cov = filtered_covs[i] + process_noise_per_day * dt
            
            # Smoother gain
            if predicted_cov > 1e-10:
                C = filtered_covs[i] / predicted_cov
            else:
                C = 0.0
            
            # Smoothed estimate
            smoothed_means[i] = (
                filtered_means[i] + C * (smoothed_means[i + 1] - predicted_mean)
            )
            
            # Smoothed covariance
            smoothed_covs[i] = (
                filtered_covs[i] + C**2 * (smoothed_covs[i + 1] - predicted_cov)
            )
        
        return smoothed_means, smoothed_covs


    def _rts_smooth_simple_with_drift(self, filtered_means, time_deltas,
                                      drift_rate, process_noise_per_day):
        """
        Simple RTS smoother without covariance tracking (faster).
        
        Args:
            filtered_means: Forward pass means
            time_deltas: Time intervals between measurements (days)
            drift_rate: Estimated growth rate (kg/day)
            process_noise_per_day: Process noise variance per day
        
        Returns:
            smoothed_means
        """
        n = len(filtered_means)
        smoothed = filtered_means.copy()
        
        # Use a reasonable fixed gain for simplicity
        for i in range(n - 2, -1, -1):
            dt = max(time_deltas[i + 1], 1.0) if i + 1 < len(time_deltas) else 1.0
            
            # Predicted mean with drift
            predicted_mean = filtered_means[i] + drift_rate * dt
            
            # Simple gain (approximation)
            C = 0.5
            
            # Smoothed estimate
            smoothed[i] = filtered_means[i] + C * (smoothed[i + 1] - predicted_mean)
        
        return smoothed

    
    def _tune_measurement_noise(self, df, target_attr, group_attr, time_attr):
        """Auto-tune measurement noise from data."""
        import time
        start = time.time()
        
        self.logger.info("="*70)
        self.logger.info("AUTO-TUNING MEASUREMENT NOISE")
        self.logger.info("="*70)
        
        # Collect data
        entities = df[group_attr].unique()
        if self.tune_on_first_n:
            entities = entities[:self.tune_on_first_n]
        
        self.logger.info(f"Analyzing {len(entities)} entities...")
        
        # Strategy: Measurement noise shows up as deviations from smooth trend
        all_second_diffs = []
        
        for entity in entities:
            entity_df = df[df[group_attr] == entity].copy()
            obs = entity_df[target_attr].values
            
            # Remove NaN
            valid_obs = obs[~np.isnan(obs)]
            
            if len(valid_obs) < 3:
                continue
            
            # Second differences remove linear trend
            # If true weight is linear, second diff = 0 + measurement noise
            second_diffs = np.diff(valid_obs, n=2)
            all_second_diffs.extend(second_diffs)
        
        all_second_diffs = np.array(all_second_diffs)
        
        # Var(X_i - 2*X_{i+1} + X_{i+2}) = Var(ε_i) + 4*Var(ε_{i+1}) + Var(ε_{i+2})
        # = 6 * σ² if errors are independent
        measurement_noise = np.var(all_second_diffs) / 6.0
        measurement_noise = np.clip(measurement_noise, 200.0, 800.0)
        
        self.measurement_noise = measurement_noise
        
        elapsed = time.time() - start
        self.logger.info(f"✓ Estimated measurement_noise: {self.measurement_noise:.0f} "
                        f"(±{np.sqrt(self.measurement_noise):.1f} kg)")
        self.logger.info(f"  Time: {elapsed:.2f}s")
        self.logger.info("="*70 + "\n")
    
    
    
    def _print_summary(self):
        """Print summary with drift statistics."""
        if not self.fitted_:
            return
        
        print("\n" + "="*70)
        print("MEASUREMENT ERROR SMOOTHER - SUMMARY")
        print("="*70)
        print(f"Measurement noise: {self.measurement_noise:.0f} (±{np.sqrt(self.measurement_noise):.1f} kg)")
        print(f"  Includes: scale error + gut fill + hydration")
        
        print(f"\nEntities processed: {len(self.entity_results)}")
        
        if self.entity_results:
            changes = [r['mean_change'] for r in self.entity_results.values()]
            print(f"Average correction per measurement: ±{np.mean(changes):.1f} kg")
            
            # Drift rate statistics
            drift_rates = [r['drift_rate'] for r in self.entity_results.values()]
            print(f"\nGrowth Rates (kg/day):")
            print(f"  Mean:   {np.mean(drift_rates):.3f}")
            print(f"  Median: {np.median(drift_rates):.3f}")
            print(f"  Std:    {np.std(drift_rates):.3f}")
            print(f"  Range:  [{np.min(drift_rates):.3f}, {np.max(drift_rates):.3f}]")
            
            # Process noise used
            process_noise = list(self.entity_results.values())[0]['process_noise_per_day']
            print(f"\nProcess noise per day: {process_noise:.2f} (±{np.sqrt(process_noise):.2f} kg/day)")
            print(f"  Biological variation around growth trend")
        
        print("\nOutput columns:")
        print("  - 'weight_filtered': Denoised weight (CAUSAL - use for prediction)")
        print("  - 'weight_smoothed': Smoother estimate (NON-CAUSAL - visualization)")
        print("\n⚠️  Use 'weight_filtered' as input to your regression model!")
        print("="*70 + "\n")

    
    def plot_entity(self, df, entity_id, target_attr, group_attr, time_attr):
        """Plot results for one entity."""
        entity_df = df[df[group_attr] == entity_id].copy()
        
        if len(entity_df) == 0:
            raise ValueError(f"Entity {entity_id} not found")
        
        time_vals = pd.to_datetime(entity_df[time_attr])
        raw = entity_df[target_attr].values
        filtered = entity_df[f'{target_attr}_filtered'].values
        smoothed = entity_df[f'{target_attr}_smoothed'].values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top plot: All three
        ax1.scatter(time_vals, raw, alpha=0.6, s=80, label='Raw (noisy)', 
                   color='gray', zorder=1, edgecolors='black', linewidths=1.5)
        ax1.plot(time_vals, filtered, 'b-', linewidth=2.5, 
                label='Filtered (causal - use for prediction)', zorder=2)
        ax1.plot(time_vals, smoothed, 'r-', linewidth=2.5, 
                label='Smoothed (non-causal - visualization)', zorder=3, alpha=0.8)
        
        ax1.set_ylabel('Weight (kg)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Measurement Error Removal: {entity_id}', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Residuals (what we removed)
        residuals_filtered = raw - filtered
        residuals_smoothed = raw - smoothed
        
        ax2.scatter(time_vals, residuals_filtered, alpha=0.6, s=60, 
                   label='Noise removed (filtered)', color='blue')
        ax2.scatter(time_vals, residuals_smoothed, alpha=0.6, s=60, 
                   label='Noise removed (smoothed)', color='red', marker='x')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(np.sqrt(self.measurement_noise), color='green', 
                   linestyle=':', linewidth=1, alpha=0.7, 
                   label=f'Expected noise: ±{np.sqrt(self.measurement_noise):.1f} kg')
        ax2.axhline(-np.sqrt(self.measurement_noise), color='green', 
                   linestyle=':', linewidth=1, alpha=0.7)
        
        ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('Measurement Errors Removed', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_all_entities(self, df, target_attr, group_attr, time_attr, 
                         max_entities=None, show_raw=True, show_filtered=True, 
                         show_smoothed=False, save_path=None, alpha=0.3):
        """
        Plot all entities on the same graph with shared axes.
        
        Args:
            df: DataFrame with smoothed results
            target_attr: Column name for target attribute
            group_attr: Column name for grouping
            time_attr: Column name for time
            max_entities: Maximum number of entities to plot (None = all)
            show_raw: Show raw observations
            show_filtered: Show filtered estimates
            show_smoothed: Show smoothed estimates
            save_path: Optional path to save the figure
            alpha: Transparency for lines (default: 0.3)
        """
        entities = df[group_attr].unique()
        if max_entities is not None:
            entities = entities[:max_entities]
        
        n_entities = len(entities)
        
        if n_entities == 0:
            raise ValueError("No entities found in DataFrame")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Generate colors for entities
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_entities, 20)))
        if n_entities > 20:
            colors = plt.cm.viridis(np.linspace(0, 1, n_entities))
        
        # Plot each entity
        for idx, entity in enumerate(entities):
            entity_df = df[df[group_attr] == entity].copy().sort_values(time_attr)
            
            if len(entity_df) == 0:
                continue
            
            time_vals = pd.to_datetime(entity_df[time_attr])
            raw = entity_df[target_attr].values
            color = colors[idx % len(colors)]
            
            # Plot raw observations
            if show_raw:
                ax.scatter(time_vals, raw, alpha=alpha*2, s=30, 
                          color=color, zorder=1, edgecolors='none')
            
            # Plot filtered
            if show_filtered and f'{target_attr}_filtered' in entity_df.columns:
                filtered = entity_df[f'{target_attr}_filtered'].values
                ax.plot(time_vals, filtered, '-', linewidth=1.5, 
                       color=color, alpha=alpha, zorder=2)
            
            # Plot smoothed
            if show_smoothed and f'{target_attr}_smoothed' in entity_df.columns:
                smoothed = entity_df[f'{target_attr}_smoothed'].values
                ax.plot(time_vals, smoothed, '--', linewidth=1, 
                       color=color, alpha=alpha*1.5, zorder=3)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Weight (kg)', fontsize=13, fontweight='bold')
        
        title = f'All Entities ({n_entities} total)'
        if show_filtered and show_smoothed:
            title += ' - Solid: Filtered, Dashed: Smoothed'
        elif show_filtered:
            title += ' - Filtered Estimates'
        elif show_smoothed:
            title += ' - Smoothed Estimates'
        
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add info box
        info_text = (
            f"Entities: {n_entities}\n"
            f"Measurement noise: {self.measurement_noise:.0f} (±{np.sqrt(self.measurement_noise):.1f} kg)\n"
            f"Total observations: {len(df)}"
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5),
               fontsize=11, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        self.logger.info(f"Plotted {n_entities} entities on shared axes")
