import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


## data = load('output/df_merged_old_runtimeSec_below2000.joblib')
data = load('output/df_merged_clean.joblib') # FM outliers > 0.9 and runtime > 2000 removed

sns.kdeplot(data=data, x='Final momentum', hue='Source', fill=True)
plt.show()

print(data['Final momentum'].describe()) # < 0.9
print(data['Runtime (sec)'].describe()) # < 2000

data['Pretreatment'] = data['Source'].map({'Old': 1, 'New': 2})

# Standardize numerical features
# numerical_features = ['Perplexity', 'Early exaggeration', 'Initial momentum', 'Final momentum', 'Theta']
# scaler = StandardScaler()
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Print standardized feature statistics
# print("\nStandardized feature statistics:")
# print(data[numerical_features].describe())

def train_decision_tree(data, source = 'New', features = None, target = None, seed = 42, figsize = (20, 10), max_depth_input = None, min_samples_leaf_input = None, savePath_model = None):
    """
    Train a decision tree model on the provided data and save the model if savePath is provided.
    Uses train-test split and cross-validation for proper model evaluation.
    
    Parameters:
    - data: DataFrame containing the training data.
    - source: Source of the data to filter on (default is 'New').
    - savePath: Path to save the trained model (default is None).
    
    Returns:
    - model: Trained decision tree model.
    """
    if 'Source' not in data.columns:
        raise ValueError("Data must contain 'Source' column for filtering.")
    if target not in data.columns:
        raise ValueError(f"Target '{target}' not found in data.")
    if not all(feature in data.columns for feature in features):
        raise ValueError(f"Not all features {features} found in data.")
    
    if source is not None:
        if source == 'Old':
            source_value = 0
        elif source == 'New':
            source_value = 1
        else:
            raise ValueError("Source must be 'Old', 'New' or None")
        data = data[data['Source_numeric'] == source_value]  # Filter

    # Define features and target
    X = data[features]
    y = data[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # 1. Find optimal hyperparameters using GridSearchCV on training data
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],  # Shallow trees for interpretability
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [5, 10, 15, 20, 25, 30],  # Ensure enough samples per leaf for stability
        'max_leaf_nodes': [10, 15, 20, None]  # Control complexity
    }

    model = DecisionTreeRegressor(random_state=seed)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_params['max_depth'] = max_depth_input if max_depth_input else best_params['max_depth']
    best_params['min_samples_leaf'] = min_samples_leaf_input  if min_samples_leaf_input else best_params['min_samples_leaf']
    
    print(f"Training decision tree for {target} with source {source}...")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"Source: {source}")
    print(f"Best parameters: {best_params}")

    # Train final model with best parameters on training data
    best_model = DecisionTreeRegressor(random_state=seed, **best_params)
    
    # Perform cross-validation on the training data
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='r2')
    print(f"Cross-validation R² scores (on training data): {cv_scores}")
    print(f"Mean CV R² (on training data): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Calculate RMSE using cross_val_predict on training data
    y_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=kf)
    rmse_cv = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    print(f"Cross-validated RMSE (on training data): {rmse_cv:.4f}")

    # Train final model on full training set
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"\nTest set performance:")
    print(f"R² Score: {r2_test:.4f}")
    print(f"RMSE: {rmse_test:.4f}")

    if savePath_model:
        dump(best_model, savePath_model)
        print(f"Model saved to {savePath_model}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    return best_model


def plot_decision_tree(model, feature_names, figsize=(20, 10), max_depth_plot = 4, savePath=None, dpi=300):
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True, 
        fontsize=10,
        max_depth=max_depth_plot)
    # plt.title("Decision Tree")
    plt.tight_layout()
    if savePath:
        # Save as PDF with high quality
        plt.savefig(savePath, dpi=dpi, format='pdf', bbox_inches='tight')
        print(f"Decision tree plot saved to {savePath}")
    plt.show()


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
    plt.legend(title=color_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_feature_importance(model, feature_names, figsize=(10, 6), fontsize = 14, savePath=None):
    """
    Plot feature importances as a horizontal bar plot.
    
    Parameters:
    - model: Trained decision tree model
    - feature_names: List of feature names
    - figsize: Figure size tuple
    - fontsize: Font size for labels and text
    - savePath: Path to save the plot (optional)
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    # plt.xlabel('Feature Importance')
    # plt.title('Feature Importance in Decision Tree')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set fontsize for y-axis labels
    plt.yticks(fontsize=fontsize)
    
    # Add value labels on the bars
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Feature importance plot saved to {savePath}")
    
    plt.show()


features = ['Perplexity', 'Early exaggeration', 'Initial momentum', 'Final momentum', 'Theta', 'Pretreatment']
features_split = ['Perplexity', 'Early exaggeration', 'Initial momentum', 'Final momentum', 'Theta']


model_KL_full = train_decision_tree(data, source=None, features=features, target='KL', max_depth_input=4, min_samples_leaf_input=20)
plot_decision_tree(model_KL_full, feature_names=features, max_depth_plot=4)
plot_feature_importance(model_KL_full, feature_names=features, fontsize=12)

model_Stress_full = train_decision_tree(data, source=None, features=features, target='Stress', max_depth_input=7, min_samples_leaf_input=30)
plot_decision_tree(model_Stress_full, feature_names=features, max_depth_plot=7)

# Full does not capture difference between Old and Neww  well
model_T30_full = train_decision_tree(data, source=None, features=features, target='T(30)', max_depth_input=7, min_samples_leaf_input=30)
plot_decision_tree(model_T30_full, feature_names=features, max_depth_plot=7)

# model_T30_old = train_decision_tree(data, source='Old', features=features_split, target='T(30)', max_depth_input=7, min_samples_leaf_input=20)
# plot_decision_tree(model_T30_old, feature_names=features_split, max_depth_plot=7)
# pairscatter_colorcoded(data[data['Source'] == 'Old'], x='Perplexity', y='T(30)', color_col='Theta')
# sns.kdeplot(data=data[data['Source'] == 'Old'], x='T(30)', fill=True)
# plt.show()

# model_T30_new = train_decision_tree(data, source='New', features=features_split, target='T(30)', max_depth_input=7, min_samples_leaf_input=25)
# plot_decision_tree(model_T30_new, feature_names=features_split, max_depth_plot=7)
# pairscatter_colorcoded(data[data['Source'] == 'New'], x='Perplexity', y='T(30)', color_col='Theta')
# sns.kdeplot(data=data[data['Source'] == 'New'], x='T(30)', fill=True)
# plt.show()


model_T300_full = train_decision_tree(data, source=None, features=features, target='T(300)', max_depth_input=7, min_samples_leaf_input=7)
plot_decision_tree(model_T300_full, feature_names=features, max_depth_plot=7)

model_runtime_full = train_decision_tree(data, source=None, features=features, target='Runtime (sec)', max_depth_input=5, min_samples_leaf_input=30)
plot_decision_tree(model_runtime_full, feature_names=features, max_depth_plot=7)






data[data['Source']=='New']['KL'].describe()