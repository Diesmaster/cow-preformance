from data_processor.DataProcessor import DataProcessing
from base_models.OLSModel import OLSModel

def main():
    processor = DataProcessing()
    n_weigings = [3]
    dfs = processor.get_dfs(n_weigings)
    dependent_attr = 'pred_adgLatest_average'
    independent_attr = ['weight']
    
    for n, df in dfs.items():
        print(f"\n{'='*80}")
        print(f"Processing model for n = {n}")
        print(f"Dataset has {len(df)} entries")
        print(f"{'='*80}\n")
        
        # Create and fit the OLS model with cross-validation
        ols_model = OLSModel(independent_attr, dependent_attr, n, 'weight_test')
        ols_model.fit_with_cv(df, k=5, random_state=42)
        
        # Print summary and diagnostics
        print("\n" + "="*80)
        print("MODEL SUMMARY")
        print("="*80)
        ols_model.summary()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC TESTS")
        print("="*80)
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

if __name__ == "__main__":
    main()
