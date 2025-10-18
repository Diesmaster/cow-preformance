import numpy as np

from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel
from base_models.FixedEffectsModel import FixedEffectsModel 
from base_models.PanelOLS import PanelOLSModel 
from base_models.SIMEXModel import SIMEXModel 
from models.models import models, OLS_models

from utils.filter_utils import filter_few_datapoints

def run_model(df, independent_attr, dependent_attr, n, prefix, model_name):
    
    model_name = f'{prefix}_{model_name}'

    ols_model = OLSModel(independent_attr, dependent_attr, n, model_name)

    try:
        ols_model.fit(df)
        
        print('='*20)
        print(f"title: {model_name}")
        ols_model.summary()
        print('='*20)
        # ols_model.print_diagnostics(show_arrays=False)
        
        # Save results to JSON
        ols_model.save_results()
        
        # Create and save plot
        ols_model.plot(df, save=True)
        
    except Exception as e:
        print("-"*20)
        print(e)
        print("-"*20)

def main():
    processor = DataProcessing()
    n_weigings = [1]
    dfs = processor.get_dfs(n_weigings)
    
    # Iterate through each model configuration
    for model_name, model_config in OLS_models.items():
        if 'pass' in model_config:
            if model_config['pass'] == True:
                continue

        dependent_attr = model_config['depended_attr']
        independent_attr = model_config['indpended_attr']
        
        print(f"\n{'='*80}")
        print(f"Running Model: {model_name}")
        print(f"Dependent Variable: {dependent_attr}")
        print(f"Independent Variables: {independent_attr}")
        print(f"{'='*80}\n")
        
        # Iterate through each dataset
        for n, df in dfs.items():
            if model_name.startswith('simental'):
                df_simental = df[df['breed'] == 'Simental'].copy()
                run_model(df_simental, independent_attr, dependent_attr, n, '', model_name)
            if model_name.startswith('limousine'):
                print(f"\n{'='*80}")
                print(f"Processing model '{model_name}' for n = {n}")
                print(f"Dataset has {len(df)} entries")
                print(f"{'='*80}\n")
              
                df_limousin = df[df['breed'].isin(['Limousin' ])].copy()

                   # Create and fit the OLS model with cross-validation
                run_model(df_limousin, independent_attr, dependent_attr, n, 'Limousin', model_name)

            
if __name__ == "__main__":
    main()
