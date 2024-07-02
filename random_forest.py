import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance,partial_dependence
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import os

# Load data
df_parameters_results = pd.read_csv('output/df_parameters_results.csv')
df_parameters_results.columns
parameters = ['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta']
outcomes = ['KL_divergence', 'trust_k30', 'trust_k300', 'stress']

# Correlation matrix
cormat = df_parameters_results.drop(df_parameters_results.columns[[0,1,7,8]],axis=1).corr()
cormat = cormat.loc[parameters, outcomes]
round(cormat,2)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(cormat, cmap=cmap, vmin=cormat.min().min(), vmax=cormat.max().max(), center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True, fmt=".2f")
plt.show()

# Creating pairplot with seaborn
#sns.pairplot(df_parameters_results.loc[:,parameters + outcomes])
#plt.show()
# Plotting multiple scatter plots
def plot_scatter_with_regression(data, parameters, outcomes, regression_type=None, alpha = 0.5, degree = 1):
    # Function for LOESS smoothing using scipy
    def lowess_smooth(x, y, frac=0.67):
        f = interp1d(x, y, kind='linear')
        x_new = np.linspace(min(x), max(x), num=len(x))
        y_smooth = f(x_new)
        return x_new, y_smooth

    # Plotting multiple scatter plots with optional regression lines
    fig, axes = plt.subplots(nrows=len(outcomes), ncols=len(parameters), figsize=(12, 8), 
                             sharex='col', sharey='row')  # Share axes appropriately

    # Loop through each subplot
    for i, outcome in enumerate(outcomes):
        for j, parameter in enumerate(parameters):
            ax = axes[i, j]
            
            # Scatter plot
            ax.scatter(data[parameter], data[outcome], label='Data', alpha=alpha)
            
            # Fit regression line if specified
            if regression_type == 'polyfit':
                coeffs = np.polyfit(data[parameter], data[outcome], deg=degree)
                #ax.plot(data[parameter], np.polyval(coeffs, data[parameter]), color='red', label='Polyfit')
                p = np.poly1d(coeffs)
                x_sorted = np.sort(data[parameter])
                ax.plot(x_sorted, p(x_sorted), color='red', label=f'Polyfit (deg={degree})')
            elif regression_type == 'loess':
                #x_smooth, y_smooth = lowess_smooth(data[parameter], data[outcome])
                #ax.plot(x_smooth, y_smooth, color='red', label='LOESS')
                smoothed = lowess(data[outcome], data[parameter], frac=0.67)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOESS')
            
            # Set labels only for leftmost plots and bottom plots
            if j == 0:
                ax.set_ylabel(outcome)
            if i == len(outcomes) - 1:
                ax.set_xlabel(parameter)
            
            # Remove ticks and labels for other plots
            if i < len(outcomes) - 1:
                ax.set_xticks([])
                ax.set_xlabel('')
            if j > 0:
                ax.set_yticks([])
                ax.set_ylabel('')
            
            # Add legend to the first plot of the last column
            if j == len(parameters) - 1 and i == 0:
                ax.legend()

    plt.tight_layout()
    plt.show()

plot_scatter_with_regression(df_parameters_results, parameters, outcomes, regression_type='polyfit', alpha=1, degree=2)


# Group by outcomes and summarize
summary = df_parameters_results.groupby(outcomes).size().reset_index(name='Count')

# Crosstabulation of outcomes with parameters
crosstab = pd.crosstab(index=df_parameters_results['KL_divergence'], 
                       columns=df_parameters_results['Perplexity'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  print(crosstab)

# Random Forest Regressor - multivariate
parameters_df = df_parameters_results[parameters]
outcomes_df = df_parameters_results[outcomes]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(parameters_df, outcomes_df, test_size=0.2, random_state=42)

# Build a predictive model (using Random Forest as an example)
modelRF = RandomForestRegressor(n_estimators=100, random_state=42)
modelRF.fit(X_train, y_train)

# The impurity-based feature importances.
#The higher, the more important the feature. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.
#Warning: impurity-based feature importances can be misleading for high cardinality features (many unique values). See sklearn.inspection.permutation_importance as an alternative.
#The values of this array sum to 1, unless all trees are single node trees consisting of only the root node, in which case it will be an array of zeros.
importance = modelRF.feature_importances_ #array([9.89389402e-01, 2.12487041e-03, 1.96343629e-05, 1.30454082e-05, 8.45304779e-03])
feature_importance_df = pd.DataFrame({'Parameter': parameters, 'Importance': np.round(importance, 2)})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Parameter', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
# The score
#The coefficient of determination is defined as, where is the residual sum of squares ((y_true - y_pred)** 2).sum() and is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
# The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a score of 0.0.
modelRF.score(X_train, y_train) # 0.9997842695432688

# Partial dependence plots for key parameters
features = [i for i in range(len(parameters))]
PartialDependenceDisplay.from_estimator(modelRF, X_train, features=features, feature_names=parameters, grid_resolution=50)
plt.show()


# RF for each of the outcomes
def RF_outcome(outcome, df, parameters_df):
  outcomes_df = df[outcome]
  parameters = parameters_df.columns
  
  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(parameters_df, outcomes_df, test_size=0.2, random_state=42)

  # Build a predictive model (using Random Forest as an example)
  modelRF = RandomForestRegressor(n_estimators=100, random_state=42)
  modelRF.fit(X_train, y_train)

  print(f'Feature importance for quality metric {outcome}')
  importance = modelRF.feature_importances_
  feature_importance_df = pd.DataFrame({'Parameter': parameters, 'Importance': np.round(importance, 2)})
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
  print(feature_importance_df)

  # Plot feature importance
  plt.figure(figsize=(10, 6))
  sns.barplot(x='Importance', y='Parameter', data=feature_importance_df)
  plt.title(f'{outcome}')
  plt.show()

  print(f'Model score of Random Forest [{outcome}]: {modelRF.score(X_train, y_train)}')

  # Permutation importance
  perm_importance = permutation_importance(modelRF, X_test, y_test, n_repeats=10, random_state=42)
  sorted_idx = perm_importance.importances_mean.argsort()

  # Partial dependence plots for key parameters
  features = [i for i in range(len(parameters))]
  PartialDependenceDisplay.from_estimator(modelRF, X_train, features=features, feature_names=parameters, grid_resolution=50)
  plt.suptitle(f'{outcome}') #, y=1.05) 
  plt.show()

  


RF_outcome(outcomes[0], df_parameters_results, parameters_df)
RF_outcome(outcomes[1], df_parameters_results, parameters_df)

# Now including interactions between parameters
import shap
def plot_shap_interactions(modelRF, X_train, feature_names):
    explainer = shap.TreeExplainer(modelRF)
    shap_values = explainer.shap_values(X_train)
    
    # Plot SHAP summary plot
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar")
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)
    
    # Plotting interaction values
    for feature in range(X_train.shape[1]):
        shap.dependence_plot(feature, shap_values, X_train, interaction_index='auto', feature_names=feature_names)

# Example usage:
# plot_shap_interactions(modelRF, X_train, parameters)




from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assume df_parameters_results is your DataFrame
#parameters = ['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta']
#outcomes = ['Shepard_stress', 'Trustworthiness_k1', 'Trustworthiness_k2', 'KL_divergence']

X = df_parameters_results[parameters]
y = df_parameters_results[outcomes]

# PolynomialFeatures to include interactions and quadratic terms
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
poly_feature_names = poly.get_feature_names_out(parameters)

# Update the column names in the DataFrame
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

# Print the updated feature names
print(poly_feature_names)




pd_result_pair = partial_dependence(modelRF, features=[0,1], X=parameters_df,grid_resolution=50)

# Extract partial dependence values for the pair
pd_values_pair = pd_result_pair['average']

# Compute partial dependence for individual features
pd_result_perplexity = partial_dependence(modelRF, features=[0], X=parameters_df, grid_resolution=50)
pd_result_early_exaggeration = partial_dependence(modelRF, features=[1], X=parameters_df, grid_resolution=50)

pd_values_perplexity = pd_result_perplexity['average'][0]
pd_values_early_exaggeration = pd_result_early_exaggeration['average'][0]

# Compute the range (variation) in the joint partial dependence
pd_variation_pair = np.max(pd_values_pair) - np.min(pd_values_pair)

# Compute the ranges (variations) in the individual partial dependencies
pd_variation_perplexity = np.max(pd_values_perplexity) - np.min(pd_values_perplexity)
pd_variation_early_exaggeration = np.max(pd_values_early_exaggeration) - np.min(pd_values_early_exaggeration)

# Compute the pair-wise interaction measure
interaction_measure_pair = pd_variation_pair / (pd_variation_perplexity + pd_variation_early_exaggeration)

print(f"Pair-wise interaction measure for 'Perplexity' and 'Early_exaggeration': {interaction_measure_pair}")


# Compute partial dependence for Perplexity (feature j)
PD_j = partial_dependence(modelRF, features=[0], X=parameters_df, grid_resolution=50)
PD_j = PD_j['average'][0]

# Compute partial dependence for Early_exaggeration (feature k)
PD_k = partial_dependence(modelRF, features=[1], X=parameters_df, grid_resolution=50)
PD_k = PD_k['average'][0]

# Compute 2-way partial dependence function PD_jk
pd_result_pair = partial_dependence(modelRF, features=[0,1], X=parameters_df,grid_resolution=50)
pd_values_pair = pd_result_pair['average'][0]
  