import argparse
from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel
from base_models.PanelOLS import PanelOLSModel 
from models.models import models, OLS_models


def run_ols_model(df, independent_attr, dependent_attr, n, prefix, model_name):
    """Run OLS model"""
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
        
        return ols_model
        
    except Exception as e:
        print("-"*20)
        print(f"Error in OLS model: {e}")
        print("-"*20)
        return None


def run_panel_model(df, independent_attr, dependent_attr, n, prefix, model_name, 
                   group_col='cow_id', time_col='pred_date', use_cv=True, k_folds=5):
    """Run Panel OLS model with fixed effects"""
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
        
        return panel_model
        
    except Exception as e:
        print("-"*20)
        print(f"Error in Panel model: {e}")
        print("-"*20)
        return None


def filter_breed(df, model_name):
    """Filter dataframe by breed based on model name"""
    if model_name.startswith('simental'):
        df_filtered = df[df['breed'] == 'Simental'].copy()
        breed_name = 'Simental'
    elif model_name.startswith('limousine'):
        df_filtered = df[df['breed'].isin(['Limousin'])].copy()
        breed_name = 'Limousin'
        
        # Debug info for Limousin
        if 'hasBEF' in df_filtered.columns:
            bef_entries = df_filtered[df_filtered['hasBEF'] == True]
            if not bef_entries.empty:
                print("\nEntries with hasBEF = True:\n")
                print(bef_entries[['date', 'cow_id', 'pred_adgLatest_average']].to_string(index=False))
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
        default=400,
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
    print("="*80 + "\n")
    
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

        if args.kalman == True:
            model_name = 'Kal_' + model_name 

        if args.cut_tails == True:
            model_name = 'Cut_' + model_name

        if args.kalman == False and args.cut_tails == False:
            model_name = 'Raw_' + model_name

        # Skip if model is marked to pass
        if model_config.get('pass', False):
            print(f"\nSkipping model '{model_name}' (marked as pass)")
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
            df_filtered, breed_prefix = filter_breed(df, model_name)
           
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
                    k_folds=args.k_folds
                )
            else:
                run_model_func(
                    df_filtered, 
                    independent_attr, 
                    dependent_attr, 
                    n, 
                    breed_prefix, 
                    model_name
                )
    
    print("\n" + "="*80)
    print("ALL MODELS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
