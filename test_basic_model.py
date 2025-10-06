import numpy as np

from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel
from base_models.FixedEffectsModel import FixedEffectsModel 
from base_models.SIMEXModel import SIMEXModel 
from models.models import models

from utils.filter_utils import filter_few_datapoints



def main():
    processor = DataProcessing()
    n_weigings = [3]
    dfs = processor.get_dfs(n_weigings)
    
    # Iterate through each model configuration
    for model_name, model_config in models.items():
        dependent_attr = model_config['depended_attr']
        independent_attr = model_config['indpended_attr']
        
        print(f"\n{'='*80}")
        print(f"Running Model: {model_name}")
        print(f"Dependent Variable: {dependent_attr}")
        print(f"Independent Variables: {independent_attr}")
        print(f"{'='*80}\n")
        
        # Iterate through each dataset
        for n, df in dfs.items():
            print(f"\n{'='*80}")
            print(f"Processing model '{model_name}' for n = {n}")
            print(f"Dataset has {len(df)} entries")
            print(f"{'='*80}\n")
          
            if model_name != 'naive_adg':
                continue

            assumed_absolute_error = 25
            err_sd = assumed_absolute_error / np.sqrt(3)
            # Create and fit the OLS model with cross-validation
            ols_model = OLSModel(independent_attr, dependent_attr,  n, model_name)

            try:
                ols_model.fit(df)
            except Exception as e:
                print("-"*20)
                print(e)
                print("-"*20)
                continue
            # Print summary and diagnostics
            print("\n" + "="*80)
            print("MODEL SUMMARY")
            print("="*80)
            ols_model.summary()
            
            print("\n" + "="*80)
            print("DIAGNOSTIC TESTS")
            print("="*80)
            try:
                ols_model.print_diagnostics(show_arrays=False)
                
                # Save results to JSON
                ols_model.save_results()
                
                # Create and save plot
                ols_model.plot(df, save=True)
                
                # Print coefficients
                print("\n" + "="*80)
                print("MODEL COEFFICIENTS")
                print("="*80)
                print(ols_model.get_coefficients())
                print("="*80)
                
                # Evaluate on full dataset
                metrics = ols_model.evaluate(df)
                print("\n" + "="*80)
                print("EVALUATION METRICS (Full Dataset)")
                print("="*80)
                print(f"R²:   {metrics['r2']:.4f}")
                print(f"MAE:  {metrics['mae']:.4f}")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print("="*80)
            except Exception as e:
                print('-'*20)
                print(e)
                print('-'*20)


if __name__ == "__main__":
    main()
