import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats  # Fixed: Import stats for t-distribution
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
import shap
from joblib import load, dump
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder


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
# dump(df_metrics, 'output/df_metric_final_wresults.joblib')

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


# Correlation
cormat = df_metrics.drop(columns=['Combination'], inplace=False).corr()
cor_sliced = cormat.iloc[:5, 6:]
cor_sliced

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
    return cor_df, significant
    
cols_remove = ['Combination']
if 'Source' in df_metrics.columns:
    cols_remove = cols_remove + ['Source']

cor_df, significant = corr_with_pvalues(df_metrics.drop(columns=cols_remove, inplace=False))
sig_correlations = cor_df[significant].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations)


# Scatter plot
def plot_scatter_with_regression(data, parameters, outcomes, regression_type=None, alpha=0.5, degree=1, figsize=(12, 8), font_size=12, font_size_ticks=12, label_offset=-0.4, save_path=None, legend=False, color_by=None):
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

    # Create a list to store scatter objects for continuous color coding
    scatter_objects = []

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
                                  hue=color_by, palette='Set2', alpha=alpha, ax=ax)
                    if legend and j == len(parameters) - 1 and i == 0:
                        ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Use matplotlib's scatter for continuous variables
                    scatter = ax.scatter(data[parameter], data[outcome], 
                                       c=data[color_by], cmap='viridis', alpha=alpha)
                    if i == 0 and j == 0:  # Store scatter object only from first plot
                        scatter_objects.append(scatter)
            else:
                ax.scatter(data[parameter], data[outcome], alpha=alpha)
            
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

    # Add a single colorbar for continuous color coding
    if not is_categorical and color_by is not None and scatter_objects:
        # Create a new axes for the colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(scatter_objects[0], cax=cbar_ax, label=color_by)

    # Adjust the layout to ensure everything fits nicely
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)  # Save with high resolution
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()

sub_outcomes = ['KL', 'T(30)', 'T(300)']
plot_scatter_with_regression(
    df_metrics, parameters, sub_outcomes,
    label_offset=-0.5, save_path = None, font_size=11,
    font_size_ticks = 11, figsize=(10,5))


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

out = 'Stress'
param = 'Final momentum'
color_col = 'Perplexity'
pairscatter_colorcoded(df_metrics, param, out, color_col) 
pairscatter_colorcoded(df_metrics[df_metrics['Final momentum'] < 0.96], param, out, color_col)
pairscatter_colorcoded(df_metrics[df_metrics['Final momentum'] <
                                     0.96], color_col, out, param)


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

boxplots(df_metrics, outcomes, figsize=(12, 8))
boxplots(df_metrics[df_metrics[df_metrics['Final momentum'] <= 0.96]], outcomes, figsize=(12, 8))


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

identify_outliers(df_metrics, 'Stress')

# Scatter plot with and without distortion from high final momentum
sub_outcomes = ['KL', 'T(30)', 'T(300)']
plot_scatter_with_regression(
    df_metrics, parameters, sub_outcomes,
    label_offset=-0.5, save_path = None, font_size=11,
    font_size_ticks = 11, figsize=(10,5)
)

df_metrics_no_distortion = df_metrics[df_metrics['Final momentum'] < 0.96].drop(columns=['Combination'], inplace=False)
df_metrics_no_distortion['Final momentum'].max()
plot_scatter_with_regression(
    df_metrics_no_distortion, parameters, sub_outcomes,
    label_offset=-0.5, save_path = None, font_size=11,
    font_size_ticks = 11, figsize=(10,5)
)

cor_df_momemtum, significant_momentum = corr_with_pvalues(df_metrics_no_distortion)
sig_correlations_momentum = cor_df_momemtum[significant_momentum].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_momentum)



out = 'T(300)'
param = 'Perplexity'
color_col = 'Final momentum'
pairscatter_colorcoded(df_metrics_no_distortion[df_metrics_no_distortion['Final momentum'] < 0.90], param, out, color_col)

data = df_metrics_no_distortion[df_metrics_no_distortion['Final momentum'] < 0.94]
plot_scatter_with_regression(data, parameters, sub_outcomes)

cor_df_momemtum_090, significant_momentum_090 = corr_with_pvalues(data.drop(columns=['Source'], inplace=False))
sig_correlations_momentum_90 = cor_df_momemtum_090[significant_momentum_090].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_momentum_90)


lower_time, upper_time = identify_outliers(df_metrics, 'Runtime (sec)')
df_metrics_no_outliers_time = df_metrics[(df_metrics['Runtime (sec)'] > lower_time) & (df_metrics['Runtime (sec)'] < upper_time)]
df_metrics_no_outliers_time = df_metrics_no_outliers_time.drop(columns=['Combination'])
data_2 = df_metrics_no_outliers_time[df_metrics_no_outliers_time['Final momentum'] < 0.92]
plot_scatter_with_regression(data_2, parameters, sub_outcomes)
plot_scatter_with_regression(df_metrics, parameters, sub_outcomes)


# Previous results
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
lower_bound_time, upper_bound_time = identify_outliers(df_metrics_old, 'Runtime (sec)')
df_metrics_old['Runtime (sec)'].min(), df_metrics_old['Runtime (sec)'].max()
df_metrics_old['Combination'].isin(df_metrics['Combination']).sum() # 0
df_metrics_old['Source'] = 'Old'
df_metrics['Source'] = 'New'

df_metrics_old_no_outliers = df_metrics_old[(df_metrics_old['Runtime (sec)'] > lower_bound_time) & (df_metrics_old['Runtime (sec)'] < upper_bound_time)]
df_metrics_merged = pd.concat([df_metrics, df_metrics_old_no_outliers], axis=0)
len(df_metrics_merged['Combination'].unique())
df_metrics_merged.shape

plot_scatter_with_regression(df_metrics_merged, parameters, sub_outcomes, color_by='Source')
plot_scatter_with_regression(df_metrics_merged, parameters, ['Stress', 'Runtime (sec)'])

df_metrics_no_distortion['Source'] = 'New'
df_metrics_merged_no_dist = pd.concat([df_metrics_no_distortion, df_metrics_old_no_outliers], axis=0)
plot_scatter_with_regression(df_metrics_merged_no_dist, parameters, sub_outcomes, color_by='Source')
plot_scatter_with_regression(df_metrics_merged_no_dist, parameters, ['Stress', 'Runtime (sec)'], color_by='Source')


df_metrics_merged2 = pd.concat([df_metrics, df_metrics_old], axis=0)
plot_scatter_with_regression(df_metrics_merged2, parameters, sub_outcomes, color_by='Source', legend=False)


sns.histplot(data=df_metrics_old[df_metrics_old['Runtime (sec)'] < 1000], x='Runtime (sec)', hue='Perplexity', bins=30, legend=False)
plt.show()


sns.kdeplot(data=df_metrics_merged, x='Perplexity', hue='Source', fill=True)
plt.show()

sns.kdeplot(data=df_metrics_merged, x='Final momentum', hue='Source', fill=True)
plt.show()

sns.kdeplot(data=df_metrics_merged, x='T(30)', hue='Source', fill=True)
plt.show()


sns.scatterplot(data=df_metrics_merged, x='Perplexity', y='T(30)', hue='Final momentum', style='Source')
plt.show()

sns.lmplot(data=df_metrics_merged, x='Perplexity', y='T(30)', hue='Source', ci=None)
plt.show()



comb_old = df_metrics_old_no_outliers['Combination'].to_list()
comb_old = [ast.literal_eval(x) for x in comb_old]
# dump(comb_old, 'output/comb_reran_momentum.joblib')

combinations = load('output/parameter_combinations.joblib')
len(combinations) #600
len(set(combinations).intersection(set(comb_old))) #441
len(np.setdiff1d(comb_old, combinations))
len(set(comb_old) - set(combinations)) # 0

combinations_new = df_metrics['Combination'].to_list()
len(set(combinations).intersection(set(combinations_new)))
len(np.setdiff1d(combinations_new, combinations))
len(set(combinations_new) - set(combinations)) # 0


len(np.setdiff1d(comb_old, combinations_new)) # 0



data_clean_2 = df_metrics_merged_no_dist[df_metrics_merged_no_dist['Final momentum'] < 0.90]
plot_scatter_with_regression(data_clean_2, parameters, sub_outcomes, color_by='Source', legend=False)

cor_df_clean1, significant_clean1 = corr_with_pvalues(data_clean_2[data_clean_2['Source'] == 'Old'].drop(columns=['Source', 'Combination'], inplace=False))
sig_correlations_clean1 = cor_df_clean1[significant_clean1].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_clean1)

cor_df_clean2, significant_clean2 = corr_with_pvalues(data_clean_2[data_clean_2['Source'] == 'New'].drop(columns=['Source', 'Combination'], inplace=False))
sig_correlations_clean2 = cor_df_clean2[significant_clean2].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_clean2)

pairscatter_colorcoded(data_clean_2[data_clean_2['Source'] == 'Old'], 'Final momentum', 'T(300)', 'Perplexity', figsize=(10, 6))

df = data_clean_2[data_clean_2['Source'] == 'Old']
df = df[df['Perplexity'] > 60]
cor_df_clean12, significant_clean12 = corr_with_pvalues(df.drop(columns=['Source', 'Combination'], inplace=False))
sig_correlations_clean12 = cor_df_clean12[significant_clean12].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_clean12)

pairscatter_colorcoded(df, 'Final momentum', 'T(300)', 'Perplexity', figsize=(10, 6))



identify_outliers(df_metrics_old, 'Runtime (sec)')
df_metrics_old['Runtime (sec)'].min(), df_metrics_old['Runtime (sec)'].max()
(df_metrics_old['Runtime (sec)'] > 1000).sum()
df_metrics_old_rm_time = df_metrics_old[df_metrics_old['Runtime (sec)'] <= 1000]
df_metrics_old_rm_time.shape
df_metrics_old_rm_time.columns
np.unique(df_metrics_old_rm_time['Source'])

df_merge_with_FMoutliers = pd.concat([df_metrics, df_metrics_old_rm_time], axis=0)
df_merge_with_FMoutliers.shape
df_merge_with_FMoutliers_momentum = df_merge_with_FMoutliers[df_merge_with_FMoutliers['Final momentum'] < 0.90] 
df_merge_with_FMoutliers_momentum.shape

df1 = df_merge_with_FMoutliers_momentum[df_merge_with_FMoutliers_momentum['Source'] == 'Old']
cor_df_clean13, significant_clean13 = corr_with_pvalues(df1.drop(columns=['Source', 'Combination'], inplace=False))
sig_correlations_clean13 = cor_df_clean13[significant_clean13].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_clean13)

df2 = df_merge_with_FMoutliers_momentum[df_merge_with_FMoutliers_momentum['Source'] == 'New']
identify_outliers(df2, 'Runtime (sec)')
print(df2['Runtime (sec)'].min(), df2['Runtime (sec)'].max())
(df2['Runtime (sec)'] > 1000).sum()

cor_df_clean23, significant_clean23 = corr_with_pvalues(df2.drop(columns=['Source', 'Combination'], inplace=False))
sig_correlations_clean23 = cor_df_clean23[significant_clean23].stack().dropna()
print("Significant correlations (Bonferroni-adjusted):")
print(sig_correlations_clean23)


pairscatter_colorcoded(df1, 'Final momentum', 'T(300)', 'Perplexity', figsize=(10, 6))
pairscatter_colorcoded(df1, 'Theta', 'Stress', 'Perplexity', figsize=(10, 6))
pairscatter_colorcoded(df1, 'Perplexity', 'Stress', 'Theta', figsize=(10, 6))
pairscatter_colorcoded(df2, 'Perplexity', 'Stress', 'Theta', figsize=(10, 6))
corr_with_pvalues(df1[df1['Perplexity'] > 50].drop(columns=['Source', 'Combination'], inplace=False))


plot_scatter_with_regression(df_merge_with_FMoutliers_momentum, parameters, sub_outcomes, color_by='Source', legend=False)
plot_scatter_with_regression(df_merge_with_FMoutliers_momentum, parameters, ['Stress', 'Runtime (sec)'], color_by='Source', legend=False)


from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


features = parameters + ['Source']
X = df_metrics_merged_no_dist[features]
y = df_metrics_merged_no_dist['T(30)']

# One-hot encode 'Source' and combine with numerical features
X_encoded = pd.concat([X[parameters], pd.get_dummies(X['Source'], drop_first=True)], axis=1)


# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Low-FM regime (FM < 0.96)
train_low = X_train['Final momentum'] < 0.96
X_train_low = X_train[train_low]
y_train_low = y_train[train_low]

# High-FM regime (FM >= 0.96)
train_high = X_train['Final momentum'] >= 0.96
X_train_high = X_train[train_high]
y_train_high = y_train[train_high]

# Train Random Forest for each regime
rf_low = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, max_leaf_nodes=3)
rf_low.fit(X_train_low, y_train_low)

rf_high = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, max_leaf_nodes=3)
rf_high.fit(X_train_high, y_train_high)



# Reference = Mean of Old in low-FM
ref_low = X_train_low[X_train_low['Old'] == True].mean().values.reshape(1, -1)

# Explain model
explainer_low = shap.TreeExplainer(rf_low, data=ref_low)
shap_values_low = explainer_low.shap_values(X_test[X_test['Final momentum'] < 0.96])

# Summary plot
shap.summary_plot(shap_values_low, X_test[X_test['Final momentum'] < 0.96], feature_names=features)
plt.title("SHAP Summary (Low FM, Reference: Batch1)")
plt.show()

# Dependence plot: Perplexity vs. FM interaction
shap.dependence_plot(
    "Perplexity", 
    shap_values_low, 
    X_test[X_test['Final momentum'] < 0.96], 
    interaction_index="Final momentum",
    show=False
)
plt.title("Perplexity Dependence (Low FM, Colored by FM)")
plt.gcf().axes[0].axhline(y=0, color='red', linestyle='--')  # Reference line
plt.show()


# Reference = Mean of Batch1 in high-FM (if Batch1 exists in high-FM; else use low-FM)
ref_high = X_train_high[X_train_high['Old'] == True].mean().values.reshape(1, -1)

# Explain model
explainer_high = shap.TreeExplainer(rf_high, data=ref_high)
shap_values_high = explainer_high.shap_values(X_test[X_test['Final momentum'] >= 0.96])

# Summary plot
shap.summary_plot(shap_values_high, X_test[X_test['Final momentum'] >= 0.96], feature_names=features)
plt.title("SHAP Summary (High FM, Reference: Batch1)")
plt.show()

# Dependence plot: FM vs. Source interaction
shap.dependence_plot(
    "Final Momentum", 
    shap_values_high, 
    X_test[X_test['Final momentum'] >= 0.96], 
    interaction_index="Old",
    show=False
)
plt.title("FM Dependence (High FM, Colored by Source)")
plt.show()

# Compare, SHAP values for 'Source' across regimes
source_effect_low = pd.DataFrame({
    'SHAP': shap_values_low[:, features.index('Source')],
    'Regime': 'Low FM',
    'Source': X_test[X_test['Final momentum'] < 0.96]['Old']
})

source_effect_high = pd.DataFrame({
    'SHAP': shap_values_high[:, features.index('Source')],
    'Regime': 'High FM',
    'Source': X_test[X_test['Final momentum'] >= 0.96]['Old']
})

source_effect = pd.concat([source_effect_low, source_effect_high])

# Plot batch effects
sns.boxplot(data=source_effect, x='Regime', y='SHAP', hue='Source')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("SHAP Values for 'Source' (Reference: Batch1)")
plt.ylabel("Impact on T30 Relative to Batch1")
plt.show()























# Compare with previous results
df_metrics_old = pd.read_csv("output/df_metric_momentum_wresults.csv")
identify_outliers(df_metrics_old, 'Runtime (sec)')
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
lower_bound_time, upper_bound_time = identify_outliers(df_metrics_old, 'Runtime (sec)')
identify_outliers(df_metrics_old, 'Runtime (min)')


plot_scatter_with_regression(df_metrics_old, parameters, sub_outcomes) #same plot as in the paper
plot_scatter_with_regression(df_metrics_old, parameters, ['Runtime (sec)']) #same plot as in the paper
plot_scatter_with_regression(df_metrics, parameters, ['Runtime (sec)']) #same plot as in the paper

df_metrics_old['Runtime (sec)'].min(), df_metrics_old['Runtime (sec)'].max() 
df_metrics['Runtime (sec)'].min(), df_metrics['Runtime (sec)'].max()

plot_scatter_with_regression(df_metrics_old[df_metrics_old['Runtime (sec)'] <= 1050], parameters, ['Runtime (sec)'])
plot_scatter_with_regression(df_metrics_old[df_metrics_old['Runtime (sec)'] <= 1050], parameters, sub_outcomes)


df_metrics_old_no_outliers = df_metrics_old[(df_metrics_old['Runtime (sec)'] > lower_bound_time) & (df_metrics_old['Runtime (sec)'] < upper_bound_time)]

comb_old = df_metrics_old_no_outliers['Combination']
comb_new = df_metrics['Combination']

difference_combinations = np.setdiff1d(comb_old, comb_new) # not a single overlap in combinations
difference_combinations_original = np.setdiff1d(df_metrics_old['Combination'].unique(), comb_new) # same

removed_combinations_old = df_metrics_old[~df_metrics_old['Combination'].isin(comb_old)]['Combination']
#np.setdiff1d(df_metrics_old['Combination'].unique(), comb_old)

import ast 
tuple_list = removed_combinations_old.apply(ast.literal_eval).tolist()

# Step 2: Convert to 2D NumPy array
comb_2d = np.vstack(tuple_list)  # Shape: (54, 5)

# Optional: Convert to DataFrame with column names
df_2d_old_ = pd.DataFrame(comb_2d, columns=['Perplexity', 'Early', 'Initial', 'Final', 'Theta'])
df_2d_old_['Theta'].min(), df_2d_old_['Theta'].max() # removed theta in range (0.1, 0.45)
df_2d_old_['Final'].min(), df_2d_old_['Final'].max() # removed final momentum in range (0.8, 1.0)

df_metrics_old_no_outliers['Theta'].min(), df_metrics_old_no_outliers['Theta'].max() # kept theta in range (0.19, 1.0)
df_metrics_old_no_outliers['Final momentum'].min(), df_metrics_old_no_outliers['Final momentum'].max() # kept final momentum in range (0.8, 1.0)


