import argparse
from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel
from base_models.PanelOLS import PanelOLSModel 
from models.models import models, OLS_models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_error_correlations(df, errors, model_name, 
                               exclude_cols=None, 
                               top_n=20, 
                               min_abs_corr=0.05,
                               save_dir='results/error_correlations',
                               show_plot=False):
    """
    Analyze correlations between model errors and all numeric columns in dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe containing all potential features
    errors : pd.Series or np.array
        Model residuals/errors (should align with df index)
    model_name : str
        Name of the model for saving results
    exclude_cols : list, optional
        Columns to exclude from analysis (e.g., dependent var, IDs)
    top_n : int, default=20
        Number of top correlations to display/plot
    min_abs_corr : float, default=0.05
        Minimum absolute correlation to include in results
    save_dir : str, default='results/error_correlations'
        Directory to save results
    show_plot : bool, default=False
        Whether to display plot
        
    Returns:
    --------
    pd.DataFrame : Correlation results sorted by absolute correlation
    """
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure errors is a Series - PRESERVE ORIGINAL INDEX
    if isinstance(errors, np.ndarray):
        # Flatten if 2D (common with panel models)
        if errors.ndim > 1:
            errors = errors.flatten()
        # Don't force df.index - let it be inferred or use RangeIndex
        errors = pd.Series(errors, name='error')
    elif isinstance(errors, pd.Series):
        # Already a Series, keep as-is with its original index
        pass
    elif hasattr(errors, 'values'):
        # Handle pandas DataFrame or other 2D structures
        error_vals = errors.values
        if error_vals.ndim > 1:
            error_vals = error_vals.flatten()
        # Preserve the original index if it exists
        if hasattr(errors, 'index'):
            errors = pd.Series(error_vals, index=errors.index, name='error')
        else:
            errors = pd.Series(error_vals, name='error')
    else:
        errors = pd.Series(errors, name='error')
    
    # Align df and errors using their indices - find common rows
    if hasattr(errors.index, 'names') and errors.index.names != [None]:
        # MultiIndex - need to align differently
        # Ensure errors has a name for the column
        if errors.name is None:
            errors.name = 'error'
        
        # Reset both to regular indices for alignment
        df_reset = df.reset_index()
        errors_df = errors.reset_index()  # This creates a DataFrame with index cols + error col
        
        # Merge to find common rows
        index_cols = list(errors.index.names)
        if all(col in df_reset.columns for col in index_cols):
            merged = df_reset.merge(errors_df, on=index_cols, how='inner')
            # Separate df_aligned and errors - keep only original df columns
            df_cols = [col for col in df_reset.columns if col != errors.name]
            df_aligned = merged[df_cols].set_index(index_cols)
            # Extract the error column
            errors = merged.set_index(index_cols)[errors.name]
        else:
            # Fallback: just use df as-is and hope indices align
            df_aligned = df.copy()
    else:
        # Regular index - use loc with intersection
        common_idx = df.index.intersection(errors.index)
        if len(common_idx) == 0:
            # No common index - assume they align positionally
            df_aligned = df.iloc[:len(errors)].copy()
            errors = errors.reset_index(drop=True)
            errors.index = df_aligned.index
        else:
            df_aligned = df.loc[common_idx].copy()
            errors = errors.loc[common_idx]
    
# Get numeric columns only
    numeric_cols = df_aligned.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude specified columns
    if exclude_cols is not None:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Exclude columns starting with 'pred_' unless specifically requested
    numeric_cols = [col for col in numeric_cols if not col.startswith('pred_')]
    
    print(f"\nAnalyzing correlations for {len(numeric_cols)} numeric columns...")
    
    # Calculate correlations
    correlations = {}
    for col in numeric_cols:
        # Skip columns with all NaN or no variance
        if df_aligned[col].notna().sum() > 0 and df_aligned[col].std() > 0:
            # Calculate correlation only on non-null values
            valid_mask = df_aligned[col].notna() & errors.notna()
            if valid_mask.sum() > 10:  # Require at least 10 valid observations
                corr = df_aligned.loc[valid_mask, col].corr(errors[valid_mask])
                correlations[col] = corr
    
    # Create results dataframe
    corr_df = pd.DataFrame({
        'column': list(correlations.keys()),
        'correlation': list(correlations.values())
    })
    
    # Add absolute correlation for sorting
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    
    # Filter by minimum correlation
    corr_df = corr_df[corr_df['abs_correlation'] >= min_abs_corr]
    
    # Sort by absolute correlation
    corr_df = corr_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Error Correlation Analysis: {model_name}")
    print(f"{'='*80}")
    print(f"Total columns analyzed: {len(numeric_cols)}")
    print(f"Columns with |corr| >= {min_abs_corr}: {len(corr_df)}")
    print(f"\nTop {min(top_n, len(corr_df))} Correlations:")
    print("-"*80)
    
    for idx, row in corr_df.head(top_n).iterrows():
        print(f"{idx+1:2d}. {row['column']:40s} | r = {row['correlation']:7.4f} | |r| = {row['abs_correlation']:7.4f}")
    
    print("="*80)
    
    # Save results to CSV
    csv_path = Path(save_dir) / f"{model_name}_error_correlations.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    if len(corr_df) > 0:
        plot_path = Path(save_dir) / f"{model_name}_error_correlations.png"
        create_correlation_plot(corr_df, model_name, top_n, plot_path, show_plot)
        print(f"Plot saved to: {plot_path}")
    
    return corr_df


def create_correlation_plot(corr_df, model_name, top_n, save_path, show_plot=False):
    """Create and save correlation plot"""
    
    plot_data = corr_df.head(top_n).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_data) * 0.3)))
    
    # Create horizontal bar plot
    colors = ['red' if x < 0 else 'blue' for x in plot_data['correlation']]
    bars = ax.barh(range(len(plot_data)), plot_data['correlation'], color=colors, alpha=0.6)
    
    # Customize plot
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['column'])
    ax.set_xlabel('Correlation with Error', fontsize=12)
    ax.set_title(f'Top {len(plot_data)} Error Correlations: {model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, plot_data['correlation'])):
        x_pos = corr + (0.01 if corr > 0 else -0.01)
        ha = 'left' if corr > 0 else 'right'
        ax.text(x_pos, i, f'{corr:.3f}', 
                ha=ha, va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis so highest correlation is on top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_error_correlations_by_group(df, errors, model_name, 
                                        group_col='breed',
                                        exclude_cols=None,
                                        top_n=15,
                                        save_dir='results/error_correlations'):
    """
    Analyze error correlations separately for each group (e.g., breed).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    errors : pd.Series or np.array
        Model errors
    model_name : str
        Model name
    group_col : str
        Column to group by
    exclude_cols : list, optional
        Columns to exclude
    top_n : int
        Number of top correlations per group
    save_dir : str
        Directory to save results
        
    Returns:
    --------
    dict : Dictionary with group names as keys and correlation DataFrames as values
    """
    
    # Ensure errors is a Series - PRESERVE ORIGINAL INDEX
    if isinstance(errors, np.ndarray):
        # Flatten if 2D (common with panel models)
        if errors.ndim > 1:
            errors = errors.flatten()
        errors = pd.Series(errors, name='error')
    elif isinstance(errors, pd.Series):
        # Already a Series, keep as-is
        pass
    elif hasattr(errors, 'values'):
        # Handle pandas DataFrame or other 2D structures
        error_vals = errors.values
        if error_vals.ndim > 1:
            error_vals = error_vals.flatten()
        if hasattr(errors, 'index'):
            errors = pd.Series(error_vals, index=errors.index, name='error')
        else:
            errors = pd.Series(error_vals, name='error')
    else:
        errors = pd.Series(errors, name='error')
    
    # Align indices - handle MultiIndex
    if hasattr(errors.index, 'names') and errors.index.names != [None]:
        # MultiIndex alignment
        # Ensure errors has a name
        if errors.name is None:
            errors.name = 'error'
        
        df_reset = df.reset_index()
        errors_df = errors.reset_index()
        index_cols = list(errors.index.names)
        
        if all(col in df_reset.columns for col in index_cols):
            merged = df_reset.merge(errors_df, on=index_cols, how='inner')
            # Separate df_aligned and errors
            df_cols = [col for col in df_reset.columns if col != errors.name]
            df_aligned = merged[df_cols].set_index(index_cols)
            errors = merged.set_index(index_cols)[errors.name]
        else:
            df_aligned = df.copy()
    else:
        # Regular index
        common_idx = df.index.intersection(errors.index)
        if len(common_idx) > 0:
            df_aligned = df.loc[common_idx].copy()
            errors = errors.loc[common_idx]
        else:
            df_aligned = df.iloc[:len(errors)].copy()
            errors = errors.reset_index(drop=True)
            errors.index = df_aligned.index
    
    results = {}
    
    # Check if group_col is in columns or index
    if group_col not in df_aligned.columns:
        if hasattr(df_aligned.index, 'names') and group_col in df_aligned.index.names:
            # Group column is in the MultiIndex, need to reset index
            df_aligned = df_aligned.reset_index()
        else:
            print(f"Warning: Group column '{group_col}' not found in dataframe columns or index")
            return results
    
    print(f"\n{'='*80}")
    print(f"Group-wise Error Correlation Analysis: {model_name}")
    print(f"Grouping by: {group_col}")
    print(f"{'='*80}\n")
    
    for group_name, group_df in df_aligned.groupby(group_col):
        print(f"\n{'-'*80}")
        print(f"Group: {group_name} (n={len(group_df)})")
        print(f"{'-'*80}")
        
        # Get errors for this group
        group_errors = errors.loc[group_df.index]
        
        # Analyze correlations for this group
        group_corr = analyze_error_correlations(
            group_df,
            group_errors,
            f"{model_name}_{group_name}",
            exclude_cols=exclude_cols,
            top_n=top_n,
            save_dir=save_dir,
            show_plot=False
        )
        
        results[group_name] = group_corr
    
    return results


def run_ols_model(df, independent_attr, dependent_attr, n, prefix, model_name, 
                  analyze_errors=False, error_params=None):
    """Run OLS model with optional error analysis"""
    full_model_name = f'{prefix}_{model_name}' if prefix else model_name
    
    print(f"\n{'='*80}")
    print(f"Running OLS Model: {full_model_name}")
    print(f"{'='*80}")
    
    ols_model = OLSModel(independent_attr, dependent_attr, n, full_model_name)
    try:
        ols_model.fit(df)
        
        print('='*20)
        print(f"Title: {full_model_name}")
        ols_model.summary()
        print('='*20)
        
        # Save results to JSON
        ols_model.save_results()
        
        # Create and save plot
        ols_model.plot(df, save=True)
        
        # Error correlation analysis
        if analyze_errors and error_params:
            print("\n" + "="*80)
            print("RUNNING ERROR CORRELATION ANALYSIS")
            print("="*80)
            
            # Get residuals - try multiple common locations
            errors = None
            residual_locations = [
                ('results.idiosyncratic', lambda: ols_model.results.idiosyncratic if hasattr(ols_model, 'results') else None),
                ('results.resid', lambda: ols_model.results.resid if hasattr(ols_model, 'results') else None),
                ('results.residuals', lambda: ols_model.results.residuals if hasattr(ols_model, 'results') else None),
                ('residuals', lambda: ols_model.residuals if hasattr(ols_model, 'residuals') else None),
                ('resid', lambda: ols_model.resid if hasattr(ols_model, 'resid') else None),
                ('resids', lambda: ols_model.resids if hasattr(ols_model, 'resids') else None),
                ('fitted_model.resid', lambda: ols_model.fitted_model.resid if hasattr(ols_model, 'fitted_model') else None),
            ]
            
            for location_name, getter in residual_locations:
                try:
                    test_errors = getter()
                    if test_errors is not None and len(test_errors) > 0:
                        errors = test_errors
                        # Flatten if 2D (common with panel models - idiosyncratic is often 2D)
                        if hasattr(errors, 'ndim') and errors.ndim > 1:
                            errors = errors.squeeze()
                        elif hasattr(errors, 'values') and hasattr(errors.values, 'ndim') and errors.values.ndim > 1:
                            errors = errors.squeeze()
                        print(f"✓ Found residuals at: {location_name}")
                        break
                except (AttributeError, TypeError):
                    continue
            
            # Try to calculate residuals manually from fitted values as last resort
            if errors is None:
                try:
                    if hasattr(ols_model, 'results') and hasattr(ols_model.results, 'fitted_values'):
                        y_true = df[dependent_attr]
                        fitted_vals = ols_model.results.fitted_values
                        # Align indices
                        common_idx = y_true.index.intersection(fitted_vals.index)
                        errors = y_true.loc[common_idx] - fitted_vals.loc[common_idx]
                        print(f"✓ Calculated residuals from fitted_values (n={len(errors)})")
                except Exception as e:
                    pass
            
            if errors is None:
                print("✗ Warning: Could not extract residuals from model")
                print("  Available attributes:", [a for a in dir(ols_model) if not a.startswith('_')][:15])
                if hasattr(ols_model, 'results'):
                    print("  Results attributes:", [a for a in dir(ols_model.results) if not a.startswith('_')][:15])
                print("\n  → Run diagnose_model.py to find where residuals are stored")
                return ols_model
            
            # Columns to exclude
            exclude_cols = [dependent_attr] + independent_attr
            
            # Run correlation analysis
            analyze_error_correlations(
                df=df,
                errors=errors,
                model_name=full_model_name,
                exclude_cols=exclude_cols,
                top_n=error_params['top_n'],
                min_abs_corr=error_params['min_corr'],
                save_dir='results/error_correlations',
                show_plot=False
            )
        
        return ols_model
        
    except Exception as e:
        print("-"*20)
        print(f"Error in OLS model: {e}")
        print("-"*20)
        import traceback
        traceback.print_exc()
        return None


def run_panel_model(df, independent_attr, dependent_attr, n, prefix, model_name, 
                   group_col='cow_id', time_col='pred_date', use_cv=True, k_folds=5,
                   analyze_errors=False, analyze_by_breed=False, error_params=None):
    """Run Panel OLS model with fixed effects and optional error analysis"""
    full_model_name = f'{prefix}_{model_name}' if prefix else model_name
    
    print(f"\n{'='*80}")
    print(f"Running Panel OLS Model: {full_model_name}")
    print(f"Cross-Validation: {'Enabled' if use_cv else 'Disabled'}")
    if use_cv:
        print(f"K-Folds: {k_folds}")
    print(f"{'='*80}")
    
    panel_model = PanelOLSModel(
        independent_attr, 
        dependent_attr, 
        n, 
        full_model_name,
        group_col=group_col,
        time_col=time_col,
        entity_effects=True,
        time_effects=False
    )
    
    try:
        if use_cv:
            panel_model.fit_with_cv(df, k=k_folds)
        else:
            panel_model.fit(df)
        
        print('='*20)
        print(f"Title: {full_model_name}")
        panel_model.summary()
        print('='*20)
        
        panel_model.print_diagnostics()
        
        # Save results to JSON
        panel_model.save_results()
        
        # Create and save plot
        panel_model.plot(df, save=True)
        
        # Error correlation analysis
        if analyze_errors and error_params:
            print("\n" + "="*80)
            print("RUNNING ERROR CORRELATION ANALYSIS")
            print("="*80)
            
            # Get residuals - try multiple common locations
            errors = None
            residual_locations = [
                ('results.idiosyncratic', lambda: panel_model.results.idiosyncratic if hasattr(panel_model, 'results') else None),
                ('results.resid', lambda: panel_model.results.resid if hasattr(panel_model, 'results') else None),
                ('results.residuals', lambda: panel_model.results.residuals if hasattr(panel_model, 'results') else None),
                ('residuals', lambda: panel_model.residuals if hasattr(panel_model, 'residuals') else None),
                ('resid', lambda: panel_model.resid if hasattr(panel_model, 'resid') else None),
                ('resids', lambda: panel_model.resids if hasattr(panel_model, 'resids') else None),
                ('fitted_model.resid', lambda: panel_model.fitted_model.resid if hasattr(panel_model, 'fitted_model') else None),
            ]
            
            for location_name, getter in residual_locations:
                try:
                    test_errors = getter()
                    if test_errors is not None and len(test_errors) > 0:
                        errors = test_errors
                        # Flatten if 2D (common with panel models - idiosyncratic is often 2D)
                        if hasattr(errors, 'ndim') and errors.ndim > 1:
                            errors = errors.squeeze()
                        elif hasattr(errors, 'values') and hasattr(errors.values, 'ndim') and errors.values.ndim > 1:
                            errors = errors.squeeze()
                        print(f"✓ Found residuals at: {location_name}")
                        break
                except (AttributeError, TypeError):
                    continue
            
            # Try to calculate residuals manually from fitted values as last resort
            if errors is None:
                try:
                    if hasattr(panel_model, 'results') and hasattr(panel_model.results, 'fitted_values'):
                        y_true = df[dependent_attr]
                        fitted_vals = panel_model.results.fitted_values
                        # Align indices
                        common_idx = y_true.index.intersection(fitted_vals.index)
                        errors = y_true.loc[common_idx] - fitted_vals.loc[common_idx]
                        print(f"✓ Calculated residuals from fitted_values (n={len(errors)})")
                except Exception as e:
                    pass
            
            if errors is None:
                print("✗ Warning: Could not extract residuals from model")
                print("  Available attributes:", [a for a in dir(panel_model) if not a.startswith('_')][:15])
                if hasattr(panel_model, 'results'):
                    print("  Results attributes:", [a for a in dir(panel_model.results) if not a.startswith('_')][:15])
                print("\n  → Run diagnose_model.py to find where residuals are stored")
                return panel_model
            
            # Columns to exclude
            exclude_cols = [dependent_attr] + independent_attr + [group_col, time_col]
            
            # Run overall correlation analysis
            analyze_error_correlations(
                df=df,
                errors=errors,
                model_name=full_model_name,
                exclude_cols=exclude_cols,
                top_n=error_params['top_n'],
                min_abs_corr=error_params['min_corr'],
                save_dir='results/error_correlations',
                show_plot=False
            )
            
            # Optionally analyze by breed
            if analyze_by_breed and 'breed' in df.columns:
                analyze_error_correlations_by_group(
                    df=df,
                    errors=errors,
                    model_name=full_model_name,
                    group_col='breed',
                    exclude_cols=exclude_cols,
                    top_n=error_params['top_n'],
                    save_dir='results/error_correlations'
                )
        
        return panel_model
        
    except Exception as e:
        print("-"*20)
        print(f"Error in Panel model: {e}")
        print("-"*20)
        import traceback
        traceback.print_exc()
        return None


def filter_breed(df, model_name):
    """Filter dataframe by breed based on model name"""

    model_name = model_name.lower()

    if model_name.startswith('simental'):
        df_filtered = df[
            df['breed'].isin(['Simental', 'Simmental']) &
            (df['pred_adgLatest_average'] < 3)
        ].copy()

        breed_name = 'Simental'

    elif model_name.startswith('limousine'):
        df_filtered = df[df['breed'].isin(['Limousin', 'Limousine'])].copy()
        breed_name = 'Limousin'

    else:
        df_filtered = df.copy()
        breed_name = ''

    print(f"\nFiltered to {breed_name if breed_name else 'all breeds'}: {len(df_filtered)} entries")
    return df_filtered, breed_name

    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run OLS or Panel models on cattle data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Panel model with Kalman smoothing (default)
  python script.py --model-type panel
  
  # Run Panel model without cross-validation
  python script.py --model-type panel --no-cv
  
  # Run Panel model with custom k-folds
  python script.py --model-type panel --k-folds 10
  
  # Run OLS model without Kalman smoothing
  python script.py --model-type ols --no-kalman
  
  # Run specific model with tail cutting
  python script.py --model-type panel --model-name limousine_model1 --cut-tails
  
  # Run with custom measurement noise
  python script.py --model-type panel --kalman --measurement-noise 500
  
  # Run with error correlation analysis
  python script.py --model-type panel --analyze-errors
  
  # Run with error analysis by breed
  python script.py --model-type panel --analyze-errors --analyze-by-breed
        """
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['ols', 'panel'],
        default='panel',
        help='Type of model to run: "ols" or "panel" (default: panel)'
    )
    
    parser.add_argument(
        '--kalman',
        dest='kalman',
        action='store_true',
        help='Apply Kalman smoothing (default: True)'
    )
    parser.add_argument(
        '--no-kalman',
        dest='kalman',
        action='store_false',
        help='Do not apply Kalman smoothing'
    )
    parser.set_defaults(kalman=True)
    
    parser.add_argument(
        '--cut-tails',
        action='store_true',
        default=False,
        help='Remove bottom and top 2.5%% of data (default: False)'
    )
    
    parser.add_argument(
        '--measurement-noise',
        type=float,
        default=None,
        help='Measurement noise for Kalman filter (default: 400)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Specific model name to run (optional, runs all if not specified)'
    )
    
    parser.add_argument(
        '--n-weighings',
        type=int,
        nargs='+',
        default=[1],
        help='List of n_weighings values (default: [1])'
    )
    
    parser.add_argument(
        '--cv',
        dest='use_cv',
        action='store_true',
        help='Enable cross-validation for Panel models (default: True)'
    )
    parser.add_argument(
        '--no-cv',
        dest='use_cv',
        action='store_false',
        help='Disable cross-validation for Panel models'
    )
    parser.set_defaults(use_cv=True)
    
    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--analyze-errors',
        action='store_true',
        default=False,
        help='Run error correlation analysis after fitting models (default: False)'
    )
    
    parser.add_argument(
        '--analyze-by-breed',
        action='store_true',
        default=False,
        help='Analyze errors separately by breed (requires --analyze-errors, default: False)'
    )
    
    parser.add_argument(
        '--error-min-corr',
        type=float,
        default=0.05,
        help='Minimum absolute correlation to report in error analysis (default: 0.05)'
    )
    
    parser.add_argument(
        '--error-top-n',
        type=int,
        default=20,
        help='Number of top correlations to display in error analysis (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    print(f"Model Type:        {args.model_type.upper()}")
    print(f"Kalman Smoothing:  {args.kalman}")
    print(f"Cut Tails:         {args.cut_tails}")
    print(f"Measurement Noise: {args.measurement_noise}")
    print(f"N Weighings:       {args.n_weighings}")
    if args.model_type == 'panel':
        print(f"Cross-Validation:  {args.use_cv}")
        if args.use_cv:
            print(f"K-Folds:           {args.k_folds}")
    print(f"Specific Model:    {args.model_name if args.model_name else 'All models'}")
    if args.analyze_errors:
        print(f"Error Analysis:    Enabled")
        print(f"  - By Breed:      {args.analyze_by_breed}")
        print(f"  - Min Corr:      {args.error_min_corr}")
        print(f"  - Top N:         {args.error_top_n}")
    print("="*80 + "\n")
    
    # Prepare error analysis parameters
    error_params = {
        'top_n': args.error_top_n,
        'min_corr': args.error_min_corr
    }
    
    # Initialize data processor
    processor = DataProcessing()
    
    # Get dataframes with specified parameters
    dfs = processor.get_dfs(
        n_weighings=args.n_weighings,
        measurement_noise=args.measurement_noise,
        apply_smoothing=args.kalman,
        cut_tails=args.cut_tails
    )
    
    # Select model configurations based on model type
    if args.model_type == 'ols':
        model_configs = OLS_models
        run_model_func = run_ols_model
        print("\nUsing OLS models configuration")
    else:  # panel
        model_configs = models
        run_model_func = run_panel_model
        print("\nUsing Panel models configuration")
    
    # Filter to specific model if requested
    if args.model_name:
        if args.model_name in model_configs:
            model_configs = {args.model_name: model_configs[args.model_name]}
            print(f"Running only model: {args.model_name}")
        else:
            print(f"Error: Model '{args.model_name}' not found in configuration")
            print(f"Available models: {list(model_configs.keys())}")
            return
    
    # Iterate through each model configuration
    for model_name, model_config in model_configs.items():

        ori_model_name = model_name

        if args.kalman == True:
            model_name = 'Kal_' + model_name 

        if args.cut_tails == True:
            model_name = 'Cut_' + model_name

        if args.kalman == False and args.cut_tails == False:
            model_name = 'Raw_' + model_name

        # Skip if model is marked to pass
        if model_config.get('pass', False) and args.model_name == None:
            # print(f"\nSkipping model '{model_name}' (marked as pass)")
            continue
        
        dependent_attr = model_config['depended_attr']
        independent_attr = model_config['indpended_attr']
        
        print(f"\n{'='*80}")
        print(f"Processing Model: {model_name}")
        print(f"Dependent Variable: {dependent_attr}")
        print(f"Independent Variables: {independent_attr}")
        print(f"{'='*80}\n")
        
        # Iterate through each dataset
        for n, df in dfs.items():
            print(f"\n{'='*80}")
            print(f"Dataset n = {n} (size: {len(df)} entries)")
            print(f"{'='*80}\n")
          

            # Filter by breed if needed
            df_filtered, breed_prefix = filter_breed(df, ori_model_name)
           
            breed_prefix = ''

            if len(df_filtered) == 0:
                print(f"Warning: No data after filtering for model '{model_name}'")
                continue
            
            # Run the appropriate model
            if args.model_type == 'panel':
                run_model_func(
                    df_filtered, 
                    independent_attr, 
                    dependent_attr, 
                    n, 
                    breed_prefix, 
                    model_name,
                    use_cv=args.use_cv,
                    k_folds=args.k_folds,
                    analyze_errors=args.analyze_errors,
                    analyze_by_breed=args.analyze_by_breed,
                    error_params=error_params
                )
            else:
                run_model_func(
                    df_filtered, 
                    independent_attr, 
                    dependent_attr, 
                    n, 
                    breed_prefix, 
                    model_name,
                    analyze_errors=args.analyze_errors,
                    error_params=error_params
                )
    
    print("\n" + "="*80)
    print("ALL MODELS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
