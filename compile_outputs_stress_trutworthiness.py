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
def plot_scatter_with_regression(data, parameters, outcomes, regression_type=None, alpha=0.5, degree=1, figsize=(12, 8), font_size=12, font_size_ticks=12, label_offset=-0.4, save_path=None, legend=False, color_by=None, fm_thresholds=None, theta_threshold=None):
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
def pairscatter_colorcoded(data, x, y, color_col, title=None, figsize=(10, 6), save_path=None):
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=color_col,  # Color points by 'Final momentum'
        palette='viridis',  # Use the viridis colormap
        hue_norm=(data[color_col].min(), data[color_col].max()),  # Sync scale
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel(x)
    plt.ylabel(y)

    # Add a legend
    plt.legend(title=color_col)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_col)  # Move legend outside
    # Show the plot
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


def analyze_correlation_by_perplexity(data, x='Theta', y='T(30)', perplexity_col='Perplexity', perplexity_bins=None):
    """
    Calculate and plot Spearman's correlation between two variables for different Perplexity levels.
    
    Parameters:
    - data: DataFrame containing the data
    - x: First variable name (default: 'Theta')
    - y: Second variable name (default: 'T(30)')
    - perplexity_col: Name of the Perplexity column
    - perplexity_bins: List of perplexity values to use as bin edges. If None, will use quantiles
    """
    if perplexity_bins is None:
        # Use quantiles to create bins
        perplexity_bins = data[perplexity_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
    
    # Calculate correlations for each bin
    correlations = []
    p_values = []
    bin_means = []
    
    for i in range(len(perplexity_bins)-1):
        lower = perplexity_bins[i]
        upper = perplexity_bins[i+1]
        
        # Get data for this perplexity range
        mask = (data[perplexity_col] >= lower) & (data[perplexity_col] < upper)
        subset = data[mask]
        
        if len(subset) > 0:
            # Calculate Spearman correlation
            corr, p_val = stats.spearmanr(subset[x], subset[y])
            correlations.append(corr)
            p_values.append(p_val)
            bin_means.append(subset[perplexity_col].mean())
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot correlation coefficients
    plt.plot(bin_means, correlations, 'bo-', label='Spearman correlation')
    
    # Add significance markers
    for i, (corr, p_val, mean) in enumerate(zip(correlations, p_values, bin_means)):
        if p_val < 0.05:
            plt.plot(mean, corr, 'r*', markersize=15)
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Mean Perplexity in bin')
    plt.ylabel('Spearman correlation')
    plt.title(f'Correlation between {x} and {y} by Perplexity level')
    
    # Add legend
    plt.legend(['Correlation', 'Significant (p < 0.05)'])
    
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print("Perplexity range\tMean Perplexity\tCorrelation\tp-value")
    print("-" * 70)
    for i in range(len(perplexity_bins)-1):
        print(f"{perplexity_bins[i]:.1f}-{perplexity_bins[i+1]:.1f}\t\t{bin_means[i]:.1f}\t\t{correlations[i]:.3f}\t\t{p_values[i]:.3f}")


def plot_variance_by_perplexity(data, variables, perplexity_col='Perplexity', perplexity_bins=None, figsize=(12, 8), font_size=12):
    """
    Create bar plots showing variance of multiple variables across different Perplexity levels.
    
    Parameters:
    - data: DataFrame containing the data
    - variables: List of variables to analyze (e.g., ['T(30)', 'T(300)', 'Stress'])
    - perplexity_col: Name of the Perplexity column
    - perplexity_bins: List of perplexity values to use as bin edges. If None, will use quantiles
    - figsize: Size of the figure
    - font_size: Font size for labels and titles
    """
    if perplexity_bins is None:
        # Use quantiles to create bins
        perplexity_bins = data[perplexity_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
    
    # Calculate number of rows and columns for subplots
    n_vars = len(variables)
    n_cols = min(2, n_vars)  # Maximum 2 columns
    n_rows = (n_vars + 1) // 2  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    # Process each variable
    for idx, var in enumerate(variables):
        ax = axes[idx]
        
        # Calculate statistics for each bin
        variances = []
        bin_labels = []
        counts = []
        
        for i in range(len(perplexity_bins)-1):
            lower = perplexity_bins[i]
            upper = perplexity_bins[i+1]
            
            # Get data for this perplexity range
            mask = (data[perplexity_col] >= lower) & (data[perplexity_col] < upper)
            subset = data[mask]
            
            if len(subset) > 0:
                variances.append(subset[var].var())
                bin_labels.append(f'{lower:.0f}-{upper:.0f}')
                counts.append(len(subset))
        
        # Plot variances
        bars = ax.bar(bin_labels, variances, alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Perplexity Range', fontsize=font_size)
        ax.set_ylabel(f'Variance of {var}', fontsize=font_size)
        ax.set_title(f'Variance of {var} by Perplexity Level', fontsize=font_size)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', labelsize=font_size-2)
        
        # Add sample sizes as text above bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}',
                    ha='center', va='bottom', fontsize=font_size-2)
    
    # Remove any empty subplots
    for idx in range(len(variables), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print("Variable\tPerplexity range\tSample size\tVariance")
    print("-" * 60)
    for var in variables:
        for i in range(len(perplexity_bins)-1):
            lower = perplexity_bins[i]
            upper = perplexity_bins[i+1]
            mask = (data[perplexity_col] >= lower) & (data[perplexity_col] < upper)
            subset = data[mask]
            if len(subset) > 0:
                print(f"{var}\t{lower:.0f}-{upper:.0f}\t\t{len(subset)}\t\t{subset[var].var():.3f}")

# Example usage:



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
    label_offset=-0.4,
    theta_threshold=0.45)


perplexity_limit = 47
theta_new_limit = 0.25

corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'New') & (df_merged_clean['Perplexity'] < 40)].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))
corr_with_pvalues(df_merged_clean[(df_merged_clean['Source'] == 'Old') & (df_merged_clean['Perplexity'] < 40)].drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))
corr_with_pvalues(df_merged_clean.drop(columns=['Source', 'Combination', 'Runtime (min)'], inplace=False))

plot_scatter_with_regression(
    df_merged_clean,
    parameters,
    ['Runtime (sec)'],
    color_by='Source')

pairscatter_colorcoded(
    df_merged_clean[(df_merged_clean['Source'] == 'Old')],
    'Early exaggeration', 'T(300)',
    color_col='Perplexity'
)

pairscatter_colorcoded(
    df_merged_clean,
    'Theta', 'T(30)',
    color_col='Perplexity'
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



analyze_correlation_by_perplexity(
    df_merged_clean[df_merged_clean['Source'] == 'Old'],
    x='Theta',
    y='T(30)',
    perplexity_col='Perplexity',
    perplexity_bins=[0, 20, 40, 60, 80, 100]  # You can adjust these bins
)

analyze_correlation_by_perplexity(
    df_merged_clean[df_merged_clean['Source'] == 'Old'],
    x='Perplexity',
    y='T(30)',
    perplexity_col='Theta',
    perplexity_bins=[0.2, 0.5, 0.7, 0.9, 1]  # You can adjust these bins
)


plot_variance_by_perplexity(
    df_merged_clean[df_merged_clean['Source'] == 'Old'],
    variables=['T(30)', 'T(300)', 'Stress'],
    perplexity_col='Perplexity',
    perplexity_bins=[0, 50, 100],
    figsize=(15, 10),
    font_size=12
)


df_merged_clean[df_merged_clean['Source'] == 'Old'][outcomes].var()
df_merged_clean[df_merged_clean['Source'] == 'New'][outcomes].var()


























df = df_merged_clean.copy()
theta_bins = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
perp_bins = [0, 75, 150]

# Create binned columns
df['Theta_bin'] = pd.cut(df['Theta'], bins=theta_bins, labels=[f'{theta_bins[i]:.1f}-{theta_bins[i+1]:.1f}' for i in range(len(theta_bins)-1)])
df['Perp_bin'] = pd.cut(df['Perplexity'], bins=perp_bins, labels=[f'{perp_bins[i]}-{perp_bins[i+1]}' for i in range(len(perp_bins)-1)])

# Group by both bins and compute variance
variance_by_bins = df[(df['Source'] == 'Old')].groupby(['Theta_bin', 'Perp_bin'])['T(30)'].agg(['var', 'count']).reset_index()

# Print results
print("\nVariance of T(30) by Theta and Perplexity bins:")
print(variance_by_bins)

# Optional: Create a pivot table for easier viewing
pivot_var = variance_by_bins.pivot(index='Theta_bin', columns='Perp_bin', values='var')
pivot_count = variance_by_bins.pivot(index='Theta_bin', columns='Perp_bin', values='count')

print("\nVariance table:")
print(pivot_var)
print("\nSample sizes:")
print(pivot_count)


df[(df['Source'] == 'Old') & (df['Perplexity'] <= 75)]['T(30)'].agg(['var', 'count'])
df[(df['Source'] == 'Old') & (df['Perplexity'] > 75) & (df['Perplexity'] <= 150)]['T(30)'].agg(['var', 'count'])
