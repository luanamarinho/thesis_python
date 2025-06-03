import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats  # Fixed: Import stats for t-distribution
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
import shap
from joblib import load, dump


# Load the data
path_base = '../data'
path_output = os.path.join(path_base, 'output_memmap_trust_real_0_1070_momentum.dat')
num_maps_tsne = 1070//2
output_trust = np.memmap(path_output, shape=(num_maps_tsne, 2),dtype='float64', mode='r')

path_output = os.path.join(path_base, 'output_memmap_stress_real_0_1070_momentum.dat')
output_stress = np.memmap(path_output, shape=(num_maps_tsne, 1), dtype='float64', mode='r')

#Sanity check
print(len(output_stress) == num_maps_tsne)
print(len(output_trust) == num_maps_tsne)
print(sum(output_stress > 1))
maps_to_check_stress = output_stress > 1
print(sum(output_trust[:, 0] >= 1))
print(sum(output_trust[:, 1] >= 1))

# Load the metrics
path_metrics = os.path.join("output", 'df_metrics_final.joblib.gz')
df_metrics_original = load(path_metrics)
df_metrics_original['T(30)'] = np.copy(output_trust[:,0])
df_metrics_original['T(300)'] = np.copy(output_trust[:,1])
df_metrics_original['Stress'] = np.copy(output_stress)
# dump(df_metrics, 'output/df_metric_final_wresults.joblib') # File saved

df_metrics = df_metrics_original.copy()

df_metrics.rename(
    columns={'Early_exaggeration': 'Early exaggeration',
             'Initial_momentum': 'Initial momentum',
             'Final_momentum': 'Final momentum',
             'tSNE_runtime_min': 'Runtime (min)',
             'KL_divergence': 'KL'
}, inplace=True)

df_metrics['Runtime (sec)'] = df_metrics['Runtime (min)'] * 60
print(df_metrics.columns)
# dump(df_metrics, 'output/df_metric_final_wresults_stressTrust.joblib')
df_metrics.head()
parameters = ['Perplexity', 'Early exaggeration', 'Initial momentum', 'Final momentum', 'Theta']
outcomes = [ 'KL', 'T(30)', 'T(300)', 'Stress', 'Runtime (sec)']


## Check ranges
for out in outcomes:
    print('Number of unique values of', out, ':', len(pd.unique(df_metrics[out])))
    print('Range of', out, ':', (df_metrics[out].min(), df_metrics[out].max()))



# Correlation matrix with p-values
def corr_with_pvalues(df):
    n = len(df)  # Replace with actual DataFrame
    cor_df = df.corr().iloc[:5, 5:]
    # --- Step 1: Compute p-values for each correlation ---
    rows = cor_df.index
    cols = cor_df.columns
    p_matrix = pd.DataFrame(index=rows, columns=cols)

    for i in rows:
        for j in cols:
            r = cor_df.loc[i, j]
            t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))  # Now stats is defined
            p_matrix.loc[i, j] = p_value

    # --- Step 2: Bonferroni correction ---
    p_values = p_matrix.to_numpy().flatten()
    _, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
    p_adjusted_matrix = pd.DataFrame(
        p_adjusted.reshape(p_matrix.shape),
        index=rows,
        columns=cols
    )

    # --- Step 3: Filter significant correlations ---
    significant = p_adjusted_matrix < 0.05
    return cor_df, significant, p_adjusted_matrix
    

# Scatter plot
def plot_scatter_with_regression(data, parameters, outcomes, regression_type=None, alpha=0.5, degree=1, figsize=(12, 8), font_size=12, font_size_ticks=12, label_offset=-0.4, save_path=None, legend=False, color_by=None, fm_thresholds=None, theta_threshold=None, perplexity_threshold=None):
    """
    Create scatter plots with optional regression lines and subplot labels.

    Parameters:
    - data: Pandas DataFrame containing the data.
    - parameters: List of columns to be used as x-axes.
    - outcomes: List of columns to be used as y-axes.
    - regression_type: Type of regression line ('polyfit' or 'loess'). Default is None.
    - alpha: Transparency level for scatter plot points. Default is 0.5.
    - degree: Degree of the polynomial for 'polyfit' regression. Default is 1.
    - figsize: Size of the figure. Default is (12, 8).
    - font_size: Font size for the plot labels, and titles. Default is 12.
    - font_size_ticks: Font size for ticks. Default is 12.
    - label_offset: Vertical offset for subplot labels to avoid overlap. Default is -0.4.
    - save_path: File path to save the plot. If None, the plot will not be saved. Default is None.
    - legend: Boolean flag to control the display of the legend. Default is False.
    - color_by: Column name to use for color coding points. Can be categorical or continuous. Default is None.
    - fm_thresholds: Value or list of values of Final momentum to draw vertical lines at. Default is None.
    - theta_threshold: Value of Theta to draw vertical line at. Default is None.
    - perplexity_threshold: Value of Perplexity to draw vertical line at. Default is None.
    """

    # Calculate the number of rows and columns
    n_rows = len(outcomes)
    n_cols = len(parameters)

    # Create the figure with subplot2grid layout
    fig = plt.figure(figsize=figsize)

    # Create a counter for subplot labels (e.g., (a), (b), etc.)
    subplot_label_counter = 0

    # Determine if color_by is categorical
    is_categorical = False
    if color_by is not None and color_by in data.columns:
        is_categorical = data[color_by].dtype == 'object' or data[color_by].dtype.name == 'category'

    # Convert single threshold to list for consistent handling
    if fm_thresholds is not None and not isinstance(fm_thresholds, list):
        fm_thresholds = [fm_thresholds]

    # Loop through each subplot
    for i, outcome in enumerate(outcomes):
        for j, parameter in enumerate(parameters):
            # Use subplot2grid to create a custom grid position
            ax = plt.subplot2grid((n_rows, n_cols), (i, j))
            
            # Scatter plot with optional color coding
            if color_by is not None and color_by in data.columns:
                if is_categorical:
                    # Use seaborn's scatterplot for categorical variables
                    sns.scatterplot(data=data, x=parameter, y=outcome, 
                                  hue=color_by, palette='Set2', alpha=alpha, ax=ax, legend=False)
                else:
                    # Use matplotlib's scatter for continuous variables
                    scatter = ax.scatter(data[parameter], data[outcome], 
                                       c=data[color_by], cmap='viridis', alpha=alpha)
            else:
                ax.scatter(data[parameter], data[outcome], alpha=alpha)
            
            # Add vertical lines if Final momentum is on x-axis and thresholds are provided
            if parameter == 'Final momentum' and fm_thresholds is not None:
                for threshold in fm_thresholds:
                    ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
            
            # Add vertical line if Theta is on x-axis and threshold is provided
            if parameter == 'Theta' and theta_threshold is not None:
                ax.axvline(x=theta_threshold, color='gray', linestyle='--', alpha=0.5)
            
            # Add vertical line if Perplexity is on x-axis and threshold is provided
            if parameter == 'Perplexity' and perplexity_threshold is not None:
                ax.axvline(x=perplexity_threshold, color='blue', linestyle='--', alpha=0.5)
            
            # Fit regression line if specified
            if regression_type == 'polyfit':
                coeffs = np.polyfit(data[parameter], data[outcome], deg=degree)
                p = np.poly1d(coeffs)
                x_sorted = np.sort(data[parameter])
                ax.plot(x_sorted, p(x_sorted), color='red', label=f'Polyfit (deg={degree})')
            elif regression_type == 'loess':
                smoothed = lowess(data[outcome], data[parameter], frac=0.67)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOESS')
            
            # Set labels only for leftmost plots and bottom plots
            if j == 0:
                ax.set_ylabel(outcome, fontsize=font_size)
            if i == len(outcomes) - 1:
                ax.set_xlabel(parameter, fontsize=font_size)
            
            # Set the font size for tick labels
            ax.tick_params(axis='both', which='major', labelsize=font_size_ticks)
            
            # Remove ticks and labels for other plots
            if i < len(outcomes) - 1:
                ax.set_xticks([])
                ax.set_xlabel('')
            if j > 0:
                ax.set_yticks([])
                ax.set_ylabel('')
            
            # Add subplot label (e.g., (a), (b), etc.) below the parameter titles
            if i == len(outcomes) - 1:
                subplot_label_counter += 1
                subplot_label = f'({chr(96 + subplot_label_counter)})'
                ax.text(0.5, label_offset, subplot_label, transform=ax.transAxes,
                        fontsize=font_size, ha='center', va='top')

    # Adjust the layout to ensure everything fits nicely
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save with high resolution
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()

# Scater plot color-coding by 'Final momentum'
def pairscatter_colorcoded(data, x, y, color_col, title=None, figsize=(10, 6), fontsize = 14, save_path=None):
    plt.figure(figsize=figsize)
    
    # Create a temporary Series for hue mapping if color_col is 'Source'
    if color_col == 'Source':
        hue_values = data[color_col].map({'Old': 1, 'New': 2})
    else:
        hue_values = data[color_col]
    
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue_values,  # Use the mapped values for hue
        palette='viridis',  # Use the viridis colormap
        hue_norm=(hue_values.min(), hue_values.max()),  # Sync scale
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel(x, fontsize = fontsize)
    plt.ylabel(y, fontsize = fontsize)

    # Add a legend
    plt.legend(title=color_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def boxplots(df, outcomes, figsize=(12, 8)):
    # Number of rows and columns for the grid
    n_cols = 2  # Number of columns
    n_rows = len(outcomes) // n_cols + (len(outcomes) % n_cols > 0)  # Calculate number of rows needed

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the array of axes for easy indexing
    axs = axs.flatten()

    # Loop over each metric to create a boxplot in a separate subplot
    for i, out in enumerate(outcomes):
        axs[i].boxplot(df[out], patch_artist=True, notch=True, sym='o')
        axs[i].set_title(out)
        axs[i].yaxis.grid(True)

    # Remove any empty subplots (if the number of metrics is odd)
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def identify_outliers(df, column):
    """
    Identify outliers in a specified column of a DataFrame using the IQR method.
    
    Parameters:
    - df: Pandas DataFrame containing the data.
    - column: The name of the column for which to identify outliers.
    
    Returns:
    - A tuple containing:
      - Number of outliers.
      - A DataFrame containing the outliers.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound

def corr_plot(corr_data, significant, figsize=(13,10), rotation_x_tick=30, fontsize=14, rows=None, top_margin = 0.95, bottom_margin = 0.05, width_padding = 0.5):
    """
    Plot correlation heatmap showing only significant correlations.
    
    Parameters:
    - corr_data: Correlation matrix from corr_with_pvalues
    - significant: Boolean matrix indicating significant correlations from corr_with_pvalues
    - figsize: Figure size tuple
    - rotation_x_tick: Rotation angle for x-axis labels
    - fontsize: Font size for labels
    - rows: Optional list of row indices to show. If provided, will show correlations for these rows with all columns
    - width_padding: Width padding
    """
    # Create a copy of correlation data and mask non-significant values
    corr_masked = corr_data.copy()
    corr_masked[~significant] = np.nan
    
    # If rows are specified, select those rows but keep all columns
    if rows is not None:
        corr_masked = corr_masked.iloc[rows, :]
        significant = significant.iloc[rows, :]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    heatmap = sns.heatmap(
        corr_masked, 
        cmap=cmap, 
        vmin=corr_data.min().min(), 
        vmax=corr_data.max().max(), 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar=False,  # Remove colorbar
        annot=True, 
        fmt=".2f"
    )
    
    # Format labels
    heatmap.xaxis.tick_top()
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=fontsize, rotation=rotation_x_tick, ha='left')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=fontsize, rotation=0, ha='right')
    
    # Adjust layout to reduce margins
    plt.tight_layout(w_pad=width_padding)
    # plt.subplots_adjust(top=top_margin, bottom=bottom_margin)  # Adjust top and bottom margins
    
    plt.show()


def analyze_correlation_by_perplexity(data, x, y_list, perplexity_col='Perplexity', 
                                    perplexity_bins_old=None, perplexity_bins_new=None,
                                    figsize=(20, 15), font_size=15, label_offset=-0.12, title_offset=1):
    """
    Calculate and plot Spearman's correlation between x and multiple y variables for different Perplexity levels,
    with Old and New data side by side.
    
    Parameters:
    - data: DataFrame containing the data
    - x: First variable name (e.g., 'Final momentum')
    - y_list: List of second variables to analyze (e.g., ['Stress', 'T(300)'])
    - perplexity_col: Name of the Perplexity column
    - perplexity_bins_old: List of perplexity values for Old data
    - perplexity_bins_new: List of perplexity values for New data
    - figsize: Size of the figure
    - font_size: Font size for labels and titles
    - label_offset: Vertical offset for sample size labels
    - title_offset: Vertical position of the title
    """
    if perplexity_bins_old is None:
        perplexity_bins_old = [0, 50, 100]
    if perplexity_bins_new is None:
        perplexity_bins_new = [0, 50, 100]
    
    # Get the exact colors from seaborn's Set2 palette
    colors = sns.color_palette('Set2')
    old_color = colors[1]
    new_color = colors[0]
    
    # Calculate number of rows and columns for subplots
    n_vars = len(y_list)
    n_cols = 2  # One column for Old, one for New
    n_rows = n_vars
    
    # Create figure with subplots and adjust spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    fig.suptitle(t=f'{x}', fontsize=font_size+2, y=title_offset)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Store results for printing
    results = []
    
    # Process each y variable
    for idx, y in enumerate(y_list):
        # Process Old data
        ax_old = axes[idx, 0]
        old_data = data[data['Source'] == 'Old']
        
        # Calculate correlations for Old data
        correlations_old = []
        p_values_old = []
        bin_means_old = []
        counts_old = []
        
        for i in range(len(perplexity_bins_old)-1):
            lower = perplexity_bins_old[i]
            upper = perplexity_bins_old[i+1]
            
            mask = (old_data[perplexity_col] >= lower) & (old_data[perplexity_col] < upper)
            subset = old_data[mask]
            
            if len(subset) > 0:
                corr, p_val = stats.spearmanr(subset[x], subset[y])
                correlations_old.append(corr)
                p_values_old.append(p_val)
                bin_means_old.append(subset[perplexity_col].mean())
                counts_old.append(len(subset))
                
                # Store result for printing
                results.append({
                    'source': 'Old',
                    'variable': y,
                    'bin_range': f"{lower:.0f}-{upper:.0f}",
                    'mean_perplexity': subset[perplexity_col].mean(),
                    'correlation': corr,
                    'p_value': p_val,
                    'sample_size': len(subset)
                })
        
        # Plot Old data
        ax_old.plot(bin_means_old, correlations_old, 'o-', color=old_color, alpha=0.7)
        ax_old.grid(True, alpha=0.3)
        
        # Add significance markers for Old data
        for i, (corr, p_val, mean) in enumerate(zip(correlations_old, p_values_old, bin_means_old)):
            if p_val < 0.05:
                ax_old.plot(mean, corr, '*', color='red', markersize=15)
        
        # Process New data
        ax_new = axes[idx, 1]
        new_data = data[data['Source'] == 'New']
        
        # Calculate correlations for New data
        correlations_new = []
        p_values_new = []
        bin_means_new = []
        counts_new = []
        
        for i in range(len(perplexity_bins_new)-1):
            lower = perplexity_bins_new[i]
            upper = perplexity_bins_new[i+1]
            
            mask = (new_data[perplexity_col] >= lower) & (new_data[perplexity_col] < upper)
            subset = new_data[mask]
            
            if len(subset) > 0:
                corr, p_val = stats.spearmanr(subset[x], subset[y])
                correlations_new.append(corr)
                p_values_new.append(p_val)
                bin_means_new.append(subset[perplexity_col].mean())
                counts_new.append(len(subset))
                
                # Store result for printing
                results.append({
                    'source': 'New',
                    'variable': y,
                    'bin_range': f"{lower:.0f}-{upper:.0f}",
                    'mean_perplexity': subset[perplexity_col].mean(),
                    'correlation': corr,
                    'p_value': p_val,
                    'sample_size': len(subset)
                })
        
        # Plot New data
        ax_new.plot(bin_means_new, correlations_new, 'o-', color=new_color, alpha=0.7)
        ax_new.grid(True, alpha=0.3)
        
        # Add significance markers for New data
        for i, (corr, p_val, mean) in enumerate(zip(correlations_new, p_values_new, bin_means_new)):
            if p_val < 0.05:
                ax_new.plot(mean, corr, '*', color='red', markersize=15)
        
        # Set labels and titles
        if idx == len(y_list) - 1:  # Last row
            # Add sample sizes between x-tick labels and x-axis title
            for i, count in enumerate(counts_old):
                ax_old.text(i, label_offset, f'(n={count})', 
                          ha='center', va='top', transform=ax_old.transAxes,
                          fontsize=font_size-2)
            
            for i, count in enumerate(counts_new):
                ax_new.text(i, label_offset, f'(n={count})', 
                          ha='center', va='top', transform=ax_new.transAxes,
                          fontsize=font_size-2)
            
            ax_old.set_xlabel(f'Mean {perplexity_col} in bin', fontsize=font_size, labelpad=20)
            ax_new.set_xlabel(f'Mean {perplexity_col} in bin', fontsize=font_size, labelpad=20)
        else:
            ax_old.set_xticks([])
            ax_new.set_xticks([])
        
        # Set y-label only for leftmost plots
        ax_old.set_ylabel(f'{y}', fontsize=font_size)
        ax_new.set_ylabel('')
        
        # Set y-ticks only for leftmost plots
        ax_new.set_yticks([])
        
        # Set tick parameters
        ax_old.tick_params(axis='y', labelsize=font_size-2)
        
        # Add horizontal line at 0
        ax_old.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_new.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Set y-axis limits to be symmetric around 0
        max_corr = max(max(abs(c) for c in correlations_old), max(abs(c) for c in correlations_new))
        ax_old.set_ylim(-max_corr*1.1, max_corr*1.1)
        ax_new.set_ylim(-max_corr*1.1, max_corr*1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print("Source\tVariable\tPerplexity range\tMean Perplexity\tCorrelation\tp-value\tSample size")
    print("-" * 90)
    
    # Sort results by source, variable, and bin range for consistent output
    results.sort(key=lambda x: (x['source'], x['variable'], x['bin_range']))
    
    for r in results:
        print(f"{r['source']}\t{r['variable']}\t{r['bin_range']}\t\t{r['mean_perplexity']:.1f}\t\t{r['correlation']:.3f}\t{r['p_value']:.3f}\t{r['sample_size']}")



def plot_variance_by_perplexity(data, variables, perplexity_col='Perplexity', 
                              perplexity_bins_old=None, perplexity_bins_new=None, 
                              figsize=(20, 15), font_size=15, label_offset=-0.12,
                              show_sample_sizes=True):
    """
    Create bar plots showing variance of multiple variables across different Perplexity levels,
    with Old and New data side by side using consistent colors.
    
    Parameters:
    - data: DataFrame containing the data
    - variables: List of variables to analyze (e.g., ['T(30)', 'T(300)', 'Stress'])
    - perplexity_col: Name of the Perplexity column
    - perplexity_bins_old: List of perplexity values for Old data
    - perplexity_bins_new: List of perplexity values for New data
    - figsize: Size of the figure
    - font_size: Font size for labels and titles
    - label_offset: Vertical offset for sample size labels (default: -0.12)
    - show_sample_sizes: Whether to display sample sizes below x-axis labels (default: True)
    """
    if perplexity_bins_old is None:
        perplexity_bins_old = [0, 50, 100]
    if perplexity_bins_new is None:
        perplexity_bins_new = [0, 50, 100]
    
    # Get the exact colors from seaborn's Set2 palette
    colors = sns.color_palette('Set2')
    old_color = colors[1]  #
    new_color = colors[0]  
    
    # Calculate number of rows and columns for subplots
    n_vars = len(variables)
    n_cols = 2  # One column for Old, one for New
    n_rows = n_vars
    
    # Create figure with subplots and adjust spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    # First pass: calculate all variances to determine y-axis limits
    y_limits = {}
    for var in variables:
        max_var = 0
        # Check Old data
        old_data = data[data['Source'] == 'Old']
        for i in range(len(perplexity_bins_old)-1):
            lower = perplexity_bins_old[i]
            upper = perplexity_bins_old[i+1]
            mask = (old_data[perplexity_col] >= lower) & (old_data[perplexity_col] < upper)
            subset = old_data[mask]
            if len(subset) > 0:
                max_var = max(max_var, subset[var].var())
        
        # Check New data
        new_data = data[data['Source'] == 'New']
        for i in range(len(perplexity_bins_new)-1):
            lower = perplexity_bins_new[i]
            upper = perplexity_bins_new[i+1]
            mask = (new_data[perplexity_col] >= lower) & (new_data[perplexity_col] < upper)
            subset = new_data[mask]
            if len(subset) > 0:
                max_var = max(max_var, subset[var].var())
        
        y_limits[var] = max_var * 1.1  # Add 10% padding
    
    # Process each variable
    for idx, var in enumerate(variables):
        # Process Old data
        ax_old = axes[idx, 0]
        old_data = data[data['Source'] == 'Old']
        
        # Calculate statistics for Old data
        variances_old = []
        bin_labels_old = []
        counts_old = []
        
        for i in range(len(perplexity_bins_old)-1):
            lower = perplexity_bins_old[i]
            upper = perplexity_bins_old[i+1]
            
            mask = (old_data[perplexity_col] >= lower) & (old_data[perplexity_col] < upper)
            subset = old_data[mask]
            
            if len(subset) > 0:
                variances_old.append(subset[var].var())
                # Format bin labels to show decimal places when needed
                if any(x % 1 != 0 for x in [lower, upper]):
                    bin_labels_old.append(f'{lower:.2f}-{upper:.2f}')
                else:
                    bin_labels_old.append(f'{lower:.0f}-{upper:.0f}')
                counts_old.append(len(subset))
        
        # Plot Old data
        bars_old = ax_old.bar(bin_labels_old, variances_old, alpha=0.7, color=old_color, width=0.6)
        ax_old.grid(True, alpha=0.3)
        
        # Process New data
        ax_new = axes[idx, 1]
        new_data = data[data['Source'] == 'New']
        
        # Calculate statistics for New data
        variances_new = []
        bin_labels_new = []
        counts_new = []
        
        for i in range(len(perplexity_bins_new)-1):
            lower = perplexity_bins_new[i]
            upper = perplexity_bins_new[i+1]
            
            mask = (new_data[perplexity_col] >= lower) & (new_data[perplexity_col] < upper)
            subset = new_data[mask]
            
            if len(subset) > 0:
                variances_new.append(subset[var].var())
                # Format bin labels to show decimal places when needed
                if any(x % 1 != 0 for x in [lower, upper]):
                    bin_labels_new.append(f'{lower:.2f}-{upper:.2f}')
                else:
                    bin_labels_new.append(f'{lower:.0f}-{upper:.0f}')
                counts_new.append(len(subset))
        
        # Plot New data
        bars_new = ax_new.bar(bin_labels_new, variances_new, alpha=0.7, color=new_color, width=0.6)
        ax_new.grid(True, alpha=0.3)
        
        # Set labels only for leftmost and bottom plots
        if idx == len(variables) - 1:  # Last row
            # Set x-tick positions and labels
            ax_old.set_xticks(range(len(bin_labels_old)))
            ax_old.set_xticklabels(bin_labels_old, rotation=0, fontsize=font_size-2)
            
            ax_new.set_xticks(range(len(bin_labels_new)))
            ax_new.set_xticklabels(bin_labels_new, rotation=0, fontsize=font_size-2)
            
            # Add sample sizes between x-tick labels and x-axis title only if show_sample_sizes is True
            if show_sample_sizes:
                for i, count in enumerate(counts_old):
                    ax_old.text(i, label_offset, f'(n={count})', 
                              ha='center', va='top', transform=ax_old.transAxes,
                              fontsize=font_size-2)
                
                for i, count in enumerate(counts_new):
                    ax_new.text(i, label_offset, f'(n={count})', 
                              ha='center', va='top', transform=ax_new.transAxes,
                              fontsize=font_size-2)
            
            # Set x-axis labels
            ax_old.set_xlabel(f'{perplexity_col} Range', fontsize=font_size, labelpad=20)
            ax_new.set_xlabel(f'{perplexity_col} Range', fontsize=font_size, labelpad=20)
        else:
            ax_old.set_xticks([])
            ax_new.set_xticks([])
        
        # Set y-label only for leftmost plots
        ax_old.set_ylabel(f'{var}', fontsize=font_size)
        ax_new.set_ylabel('')
        
        # Set y-ticks only for leftmost plots
        ax_new.set_yticks([])
        
        # Set tick parameters
        ax_old.tick_params(axis='y', labelsize=font_size-2)
        
        # Set the same y-axis limits for both plots
        ax_old.set_ylim(0, y_limits[var])
        ax_new.set_ylim(0, y_limits[var])
    
    # Adjust layout to ensure everything fits
    plt.tight_layout()
    plt.show()

def interaction_plot(data, x1, x2, y, source=None, figsize=(12, 5), fontsize=14, rotation=0,
                    x1_bins=None, x2_bins=None):
    """
    Create an interaction plot and scatter plot to visualize the relationship between three variables.
    
    Parameters:
    - data: DataFrame containing the data
    - x1: First variable name for x-axis
    - x2: Second variable name for grouping/coloring
    - y: Response variable name
    - source: Optional filter for 'Source' column
    - figsize: Figure size tuple
    - fontsize: Font size for labels
    - rotation: Rotation angle for x-axis labels
    - x1_bins: Optional list of bin edges for x1. If None, uses qcut with 3 groups
    - x2_bins: Optional list of bin edges for x2. If None, uses qcut with 2 groups
    """
    if source is not None:
        data = data[data['Source'] == source]
    
    plt.figure(figsize=figsize)

    # Create bins for x1
    if x1_bins is not None:
        x1_groups = pd.cut(data[x1], bins=x1_bins)
    else:
        x1_groups = pd.qcut(data[x1], 3)

    # Create bins for x2
    if x2_bins is not None:
        x2_groups = pd.cut(data[x2], bins=x2_bins)
    else:
        x2_groups = pd.qcut(data[x2], 2)

    # Custom color palette with blue and red tones
    custom_palette = ['#1f77b4', '#d62728']  # Blue and red from seaborn's default palette

    # Plot 1: Interaction plot (categorical bins)
    plt.subplot(1, 2, 1)
    sns.pointplot(data=data, 
                x=x1_groups,
                y=y, 
                hue=x2_groups,
                palette=custom_palette)
    plt.xticks(rotation=rotation)
    plt.xlabel(x1, fontsize=fontsize)
    plt.ylabel(y, fontsize=fontsize)
    plt.legend(title=x2, fontsize=fontsize-2)

    # Plot 2: Scatterplot with coloring
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=data, x=x1, y=y, hue=x2_groups, palette=custom_palette)
    plt.xlabel(x1, fontsize=fontsize)
    plt.ylabel(y, fontsize=fontsize)
    plt.legend(title=x2, fontsize=fontsize-2)

    plt.tight_layout()
    plt.show()




df_metrics = load('output/df_metric_final_wresults_stressTrust.joblib')
df_metrics['Source'] = 'New'
df_metrics.shape
df_metrics.describe()
sns.kdeplot(data=df_metrics, x='Runtime (sec)', fill=True)
plt.show()

df_metrics_old = pd.read_csv("output/df_metric_momentum_wresults.csv")
df_metrics_old.rename(
    columns={'Early_exaggeration': 'Early exaggeration',
             'Initial_momentum': 'Initial momentum',
             'Final_momentum': 'Final momentum',
             'tSNE_runtime_min': 'Runtime (min)',
             'KL_divergence': 'KL',
             'trust_k30': 'T(30)',
             'trust_k300': 'T(300)',
             'stress': 'Stress'
}, inplace=True)
df_metrics_old['Runtime (sec)'] = df_metrics_old['Runtime (min)'] * 60
df_metrics_old['Source'] = 'Old'
df_metrics_old.shape
sns.kdeplot(data=df_metrics_old, x='Runtime (sec)', fill=True)
plt.show()

df_metrics_old.describe()

(df_metrics_old['Runtime (sec)'] >= 5000).sum()
(df_metrics_old['Runtime (sec)'] >= 1500).sum()
(df_metrics_old['Runtime (sec)'] >= 1300).sum()

sns.kdeplot(data=df_metrics_old[df_metrics_old['Runtime (sec)'] <= 1500], x='Runtime (sec)', fill=True)
plt.show()


df_merged_original = pd.concat([df_metrics, df_metrics_old], axis=0)
df_merged_original.shape

parameters = ['Perplexity', 'Early exaggeration', 'Initial momentum', 'Final momentum', 'Theta']
outcomes = ['KL', 'T(30)', 'T(300)', 'Stress', 'Runtime (sec)']
sub_outcomes = ['KL', 'T(30)', 'T(300)']
plot_scatter_with_regression(df_merged_original, parameters, sub_outcomes, color_by='Source', legend=False, fm_thresholds=0.95, figsize=(15, 8))
plot_scatter_with_regression(
    df_merged_original, parameters,
    ['Stress', 'Runtime (sec)'], color_by='Source',
    legend=False, fm_thresholds=[0.95, 0.96],
    figsize=(15, 8))

plot_scatter_with_regression(
    df_merged_original, parameters, outcomes,
    color_by='Source', legend=False,
    fm_thresholds=[0.95],
    figsize=(12, 8),
    label_offset=-0.7,
    font_size=16,
    font_size_ticks=14
    )


df_merged_original[df_merged_original['Stress'] > 1]['Final momentum'].describe()
df_merged_original[df_merged_original['Final momentum'] > 0.96]['Stress'].describe()

_, upper_limit_time = identify_outliers(df_merged_original, 'Runtime (sec)')
_, upper_limit_time_new = identify_outliers(df_merged_original[df_merged_original['Source'] == 'New'], 'Runtime (sec)')
_, upper_limit_time_old= identify_outliers(df_merged_original[df_merged_original['Source'] == 'Old'], 'Runtime (sec)')  


pairscatter_colorcoded(
    df_merged_original[df_merged_original['Source'] == 'New'],
    'Theta', 'Runtime (sec)', 'Perplexity'
)

pairscatter_colorcoded(
    df_merged_original[df_merged_original['Source'] == 'Old'],
    'Theta', 'Runtime (sec)', 'Final momentum'
)

pairscatter_colorcoded(
    df_merged_original[(df_merged_original['Source'] == 'Old') & (df_merged_original['Runtime (sec)'] < 5000)],
    'Theta', 'Runtime (sec)', 'Perplexity'
)

df_merged_original[df_merged_original['Stress'] > 1]['Final momentum'].describe()
df_merged_original[df_merged_original['Final momentum'] > 0.95]['Stress'].describe()

identify_outliers(df_merged_original[df_merged_original['Source'] == 'New'], 'Stress')
identify_outliers(df_merged_original[df_merged_original['Source'] == 'New'], 'T(300)')
identify_outliers(df_merged_original[df_merged_original['Source'] == 'New'], 'T(30)')
identify_outliers(df_merged_original[df_merged_original['Source'] == 'Old'], 'Stress')
identify_outliers(df_merged_original[df_merged_original['Source'] == 'Old'], 'T(300)')
identify_outliers(df_merged_original[df_merged_original['Source'] == 'Old'], 'T(30)')


fm_limit = 0.95
runtime_limit = 2000
df_merged_original[df_merged_original['Runtime (sec)'] >= runtime_limit]['Theta'].describe() # rm 112, with theta in [0.1, 0.45]

df_merged_clean = df_merged_original[(df_merged_original['Runtime (sec)'] < runtime_limit) & (df_merged_original['Final momentum'] < fm_limit)]
df_merged_clean.describe()
# dump(df_merged_clean, 'output/df_merged_clean.joblib')

df_merged_original[df_merged_original['Runtime (sec)'] >= runtime_limit][['Theta', 'Perplexity']].describe()
df_merged_original[df_merged_original['Theta'] <= 0.45]['Runtime (sec)'].describe()


plot_scatter_with_regression(
    df_merged_clean, parameters, outcomes,
    color_by='Source', legend=False,
    fm_thresholds=[0.95],
    figsize=(12, 8),
    label_offset=-0.7,
    font_size=16,
    font_size_ticks=14
    )

pairscatter_colorcoded(
    df_merged_clean[df_merged_clean['Source'] == 'Old'],
    'Perplexity', 'Runtime (sec)', 'Theta'
)

pairscatter_colorcoded(
    df_merged_clean,
    'Perplexity', 'Runtime (sec)', 'Theta'
)

plot_scatter_with_regression(
    df_merged_clean[df_merged_clean['Theta'] > 0.5], ['Perplexity'], ['Runtime (sec)'],
    color_by='Source', legend=False,
    fm_thresholds=None,
    figsize=(12, 8),
    label_offset=-0.4)

plot_scatter_with_regression(
    df_merged_clean[df_merged_clean['Theta'] > 0.5], parameters, outcomes,
    color_by='Source', legend=False,
    fm_thresholds=None,
    figsize=(12, 8),
    label_offset=-0.4)

cor_df_new, significant_new, p_value_new = corr_with_pvalues(df_merged_clean[df_merged_clean['Source'] == 'New'].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))
cor_df_old, significant_old, p_value_old = corr_with_pvalues(df_merged_clean[df_merged_clean['Source'] == 'Old'].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))

limit = 0.45
cor_df_new_theta, significant_new_theta, p_values_new_theta = corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Theta'] >= limit)].drop(columns=['Source', 'Combination'], inplace=False))
cor_df_old_theta, significant_old_theta, p_values_old_theta = corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Theta'] >= limit)].drop(columns=['Source', 'Combination'], inplace=False))

cor_df_new_theta_below, significant_new_theta_below, _ = corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Theta'] < limit)].drop(columns=['Source', 'Combination'], inplace=False))
cor_df_old_theta_below, significant_old_theta_below, _ = corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Theta'] < limit)].drop(columns=['Source', 'Combination'], inplace=False))


plot_scatter_with_regression(
    df_merged_clean,
    parameters, outcomes,
    color_by='Source', legend=False,
    fm_thresholds=None,
    figsize=(12, 8),
    label_offset=-0.6,
    theta_threshold=0.45,
    font_size=16)


perplexity_limit = 47
theta_new_limit = 0.25

corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Perplexity'] < 40)].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))
corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Perplexity'] < 40)].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))
corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Perplexity'] >= 126)].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))

corr_with_pvalues(df_merged_clean.drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))

plot_scatter_with_regression(
    df_merged_clean[(df_merged_clean['Source'] == 'New')],
    parameters,
    ['T(30)'],
    color_by='Source',
    figsize=(10,5))

plot_scatter_with_regression(
    df_merged_clean,
    ['Perplexity'],
    ['KL'],
    color_by='Source',
    label_offset=-0.1,
    figsize=(10,6),
    perplexity_threshold=50)

pairscatter_colorcoded(
    df_merged_clean[df_merged_clean['Source'] == 'New'],
    'Final momentum', 'T(300)',
    color_col='Perplexity'
)

pairscatter_colorcoded(
    df_merged_clean[df_merged_clean['Source'] == 'Old'],
    'Perplexity', 'T(300)',
    color_col='Theta'
)




corr_plot(corr_data=cor_df_old, significant=significant_old, figsize=(12, 8), fontsize=12, rotation_x_tick=0)
rows = [0,4]
corr_plot(corr_data=cor_df_old, significant=significant_old, rows = rows,
          figsize=(8, 4), fontsize=10, rotation_x_tick=0,
          top_margin=0.99, bottom_margin=0.01)
rows = [0,3,4]
corr_plot(corr_data=cor_df_new, significant=significant_new, rows = rows,
          figsize=(8, 4), fontsize=10, rotation_x_tick=0,
          top_margin=0.99, bottom_margin=0.01)



min_perplexity = df_merged_clean['Perplexity'].min()
max_perplexity = df_merged_clean['Perplexity'].max()
plot_variance_by_perplexity(
    df_merged_clean,
    variables=outcomes,
    perplexity_col='Perplexity',
    perplexity_bins_old=[min_perplexity, 50, max_perplexity+5],
    perplexity_bins_new=[min_perplexity, 50, max_perplexity+5],
    figsize=(12, 8),
    font_size=15,
    label_offset=-0.12,
    show_sample_sizes=True
)


df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Perplexity'] <50)][outcomes].agg(['var', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Perplexity'] <50)][outcomes].agg(['var', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Perplexity'] >=50)][outcomes].agg(['var', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Perplexity'] >=50)][outcomes].agg(['var', 'count'])


df_merged_clean.groupby('Source')[outcomes].std()

df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Theta'] <0.45)]['Theta'].agg(['mean', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Theta'] <0.45)]['Theta'].agg(['mean', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Theta'] >=0.45)]['Theta'].agg(['mean', 'count'])
df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Theta'] >=0.45)]['Theta'].agg(['mean', 'count'])


analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Final momentum', y_list=['T(300)', 'Stress', 'T(30)'],
                                  perplexity_col='Perplexity',
                                  perplexity_bins_old=[0,50,155],
                                  perplexity_bins_new=[0,50,155],
                                  figsize=(20,10),
                                  font_size=17)

analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Final momentum', y_list=['Stress'],
                                  perplexity_col='Final momentum',
                                  perplexity_bins_old=[0.8,0.85,0.9,1],
                                  perplexity_bins_new=[0.8,0.85,0.9,1],
                                  figsize=(15,10))

analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Final momentum', y_list=['KL'],
                                  perplexity_col='Perplexity',
                                  perplexity_bins_old=[0,50,155],
                                  perplexity_bins_new=[0,50,155],
                                  figsize=(12,8))


analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Theta', y_list=['T(30)', 'T(300)'],
                                  perplexity_col='Perplexity',
                                  perplexity_bins_old=[0,50,155],
                                  perplexity_bins_new=[0,50,155],
                                  figsize=(12,8),
                                  font_size=14)

analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Theta', y_list=['Stress', 'Runtime (sec)'],
                                  perplexity_col='Perplexity',
                                  perplexity_bins_old=[0,50,155],
                                  perplexity_bins_new=[0,50,155],
                                  figsize=(12,8),
                                  font_size=14)



analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Early exaggeration', y_list=['T(30)', 'T(300)'],
                                  perplexity_col='Perplexity',
                                  perplexity_bins_old=[0,50,155],
                                  perplexity_bins_new=[0,50,155],
                                  figsize=(12,8),
                                  font_size=15)                                  


analyze_correlation_by_perplexity(data=df_merged_clean, x = 'Perplexity', y_list=['Runtime (sec)'],
                                  perplexity_col='Theta',
                                  perplexity_bins_old=[0,0.45,1.5],
                                  perplexity_bins_new=[0,0.45,1.5],
                                  figsize=(10,5),
                                  font_size=14)


plot_variance_by_perplexity(
    df_merged_clean,
    variables=['Runtime (sec)'],
    perplexity_col='Theta',
    perplexity_bins_old=[0,0.45,1.001],
    perplexity_bins_new=[0,0.45,1.001],
    figsize=(10, 5),
    font_size=14,
    label_offset=-0.09,
    show_sample_sizes=True
)

theta_quantiles = df_merged_clean['Theta'].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
plot_variance_by_perplexity(
    df_merged_clean,
    variables=['T(30)'],
    perplexity_col='Theta',
    perplexity_bins_old=theta_quantiles,
    perplexity_bins_new=theta_quantiles,
    figsize=(16, 8),
    font_size=15,
    label_offset=-0.09,
    show_sample_sizes=False
)


plot_variance_by_perplexity(
    df_merged_clean,
    variables=['KL'],
    perplexity_col='Perplexity',
    perplexity_bins_old=[0,50,155],
    perplexity_bins_new=[0,50,155],
    figsize=(10, 5),
    font_size=14,
    label_offset=-0.09,
    show_sample_sizes=True
)




interaction_plot(
    data=df_merged_clean,
    source='Old',
    x1='Theta',
    x2 = 'Perplexity',
    y = 'Runtime (sec)')
    #x2_bins=[0, 50, 155])

interaction_plot(
    data=df_merged_clean,
    source='New',
    x1='Perplexity',
    x2 = 'Theta',
    y = 'Runtime (sec)')
    #x2_bins=[0, 50, 155])


sns.kdeplot(df_merged_clean['T(30)'])
plt.show()

sns.kdeplot(df_merged_clean[df_merged_clean['Source'] == 'Old']['T(30)'])
plt.show()

sns.kdeplot(df_merged_clean[df_merged_clean['Source'] == 'New']['T(30)'])
plt.show()


df_merged_clean[df_merged_clean['Source'] == 'Old'].drop(columns=['Runtime (min)', 'Source', 'Combination']).corr()























# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

x = 'Theta'
y = 'Runtime (sec)'
color_col = 'Perplexity'

# Left plot (Old data)
sns.scatterplot(
    data=df_merged_clean[df_merged_clean['Source'] == 'Old'],
    x=x,
    y=y,
    hue=color_col,
    palette='viridis',
    hue_norm=(df_merged_clean[color_col].min(), df_merged_clean[color_col].max()),
    ax=ax1
)

ax1.set_title('Pretreatment 1', fontsize=14, pad=20)
ax1.set_xlabel(f'{x}', fontsize=12)
ax1.set_ylabel(f'{y}', fontsize=12)
ax1.legend(title=f'{color_col}', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Right plot (New data)
sns.scatterplot(
    data=df_merged_clean[df_merged_clean['Source'] == 'New'],
    x=x,
    y=y,
    hue=color_col,
    palette='viridis',
    hue_norm=(df_merged_clean[color_col].min(), df_merged_clean[color_col].max()),
    ax=ax2
)

ax2.set_title('Pretreatment 2', fontsize=14, pad=20)
ax2.set_xlabel(f'{x}', fontsize=12)
ax2.set_ylabel('', fontsize=12)  # Empty y-label for right plot
ax2.grid(True, linestyle='--', alpha=0.6)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()