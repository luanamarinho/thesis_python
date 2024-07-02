import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Load data
df_parameters_results = pd.read_csv('output/df_parameters_results.csv')
df_parameters_results.columns
parameters = ['Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta']
outcomes = ['KL_divergence', 'trust_k30', 'trust_k300', 'stress']

# Assuming df is your dataframe and 'KL_divergence' is the target variable
# Add a unique identifier for each unique KL divergence value
df_parameters_results['KL_id'] = df_parameters_results.groupby('KL_divergence').ngroup()

# Define the formula
formula = ("KL_divergence ~ Perplexity + I(Perplexity**2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +"
           "Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta +"
           "Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + Early_exaggeration:Theta +"
           "Initial_momentum:Final_momentum + Initial_momentum:Theta +"
           "Final_momentum:Theta")

# Fit the mixed-effects model
model = mixedlm(formula, df_parameters_results, groups=df_parameters_results['KL_id'], re_formula="~1")
result = model.fit()
print(result.summary())
