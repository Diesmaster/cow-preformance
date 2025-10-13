import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from data_processor.DataProcessor import DataProcessing

def plot_variables(df, dependent_vars, independent_vars, output_dir='plots', figsize=(10, 6)):
    """
    Create individual scatter plots for each dependent variable against each independent variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    dependent_vars : list
        List of dependent variable column names
    independent_vars : list
        List of independent variable column names
    output_dir : str
        Directory to save the plots (default: 'plots')
    figsize : tuple
        Figure size for each plot (default: (10, 6))
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each dependent variable
    for dep_var in dependent_vars:
        # Check if dependent variable exists in dataframe
        if dep_var not in df.columns:
            print(f"Warning: Dependent variable '{dep_var}' not found in dataframe. Skipping.")
            continue
            
        # Iterate through each independent variable
        for indep_var in independent_vars:
            # Check if independent variable exists in dataframe
            if indep_var not in df.columns:
                print(f"Warning: Independent variable '{indep_var}' not found in dataframe. Skipping.")
                continue
            
            # Create new figure
            plt.figure(figsize=figsize)
            
            # Remove NaN values for plotting
            plot_df = df[[dep_var, indep_var]].dropna()
            
            # Add cow_id if it exists
            if 'cow_id' in df.columns:
                plot_df = df[[dep_var, indep_var, 'cow_id']].dropna()
            
            if len(plot_df) == 0:
                print(f"Warning: No valid data points for {dep_var} vs {indep_var}. Skipping.")
                plt.close()
                continue
            
            # Create scatter plot with unique color per cow_id
            if 'cow_id' in plot_df.columns:
                unique_cows = plot_df['cow_id'].unique()
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cows)))
                
                for i, cow_id in enumerate(unique_cows):
                    cow_data = plot_df[plot_df['cow_id'] == cow_id]
                    plt.scatter(cow_data[indep_var], cow_data[dep_var], 
                              alpha=0.6, edgecolors='k', linewidth=0.5,
                              color=colors[i], label=f'Cow {cow_id}')
            else:
                plt.scatter(plot_df[indep_var], plot_df[dep_var], alpha=0.6, edgecolors='k', linewidth=0.5)
            
            # Add trend line
            try:
                z = np.polyfit(plot_df[indep_var], plot_df[dep_var], 1)
                p = np.poly1d(z)
                plt.plot(plot_df[indep_var], p(plot_df[indep_var]), "r--", alpha=0.8, linewidth=2, label='Trend line')
                
                # Calculate and display R-squared
                correlation = np.corrcoef(plot_df[indep_var], plot_df[dep_var])[0, 1]
                r_squared = correlation ** 2
                plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
                        transform=plt.gca().transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                print(f"Warning: Could not fit trend line for {dep_var} vs {indep_var}")
            
            # Labels and title
            plt.xlabel(indep_var, fontsize=12)
            plt.ylabel(dep_var, fontsize=12)
            plt.title(f'{dep_var} vs {indep_var}\n(n={len(plot_df)})', fontsize=14, fontweight='bold')
            
            # Add legend only if there aren't too many cows
            if 'cow_id' in plot_df.columns and len(plot_df['cow_id'].unique()) <= 20:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            elif 'cow_id' not in plot_df.columns:
                plt.legend()
                
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            filename = f'{dep_var}_vs_{indep_var}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            
            # Close figure to free memory
            plt.close()
    
    print(f"\nAll plots saved to '{output_dir}' directory")


def main():
    # ============================================================
    # CONFIGURE YOUR VARIABLES HERE
    # ============================================================
    
    # List of dependent variables (y-axis)
    dependent_vars = ['pred_adgLatest_average']  # <-- CHANGE THIS
    
    # List of independent variables (x-axis)
    independent_vars = ['tdn_slobber_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_silage_over_mw_dt']  # <-- CHANGE THIS
    
    # ============================================================
    
    # Initialize data processor
    processor = DataProcessing()
    n_weigings = [1]
    dfs = processor.get_dfs(n_weigings)
    
    # Iterate through each dataset
    for n, df in dfs.items():
        # Filter data
        df = df[df['pred_adgLatest_average'] >= 0]
        
        print(f"\n{'='*80}")
        print(f"Processing dataset with n = {n}")
        print(f"Dataset has {len(df)} entries")
        print(f"Dependent variables: {dependent_vars}")
        print(f"Independent variables: {independent_vars}")
        print(f"{'='*80}\n")
        
        # Process Limousin breed
        df_limousin = df[df['breed'] == 'Limousin'].copy()
        if len(df_limousin) > 0:
            print(f"Creating plots for Limousin (n={len(df_limousin)} samples)")
            plot_variables(
                df_limousin,
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                output_dir=f'plots/Limousin_n{n}'
            )
        else:
            print(f"No Limousin data available")
        
        # Process Simental breed
        df_simental = df[df['breed'] == 'Simental'].copy()
        if len(df_simental) > 0:
            print(f"Creating plots for Simental (n={len(df_simental)} samples)")
            plot_variables(
                df_simental,
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                output_dir=f'plots/Simental_n{n}'
            )
        else:
            print(f"No Simental data available")
    
    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
