df_parameters_results = readr::read_csv('output/df_parameters_results.csv')

# Perform regression for each outcome
outcomes <- names(df_parameters_results)[10:13]
parameters <- names(df_parameters_results)[3:7]
parameters_df <- df_parameters_results[,parameters]

# Set seed for reproducibility
set.seed(123)

# Split data into training and test sets (80% training, 20% test)
train_indices <- sample(1:nrow(df_parameters_results), 0.8 * nrow(df_parameters_results))
train_data <- df_parameters_results[train_indices, ]
test_data <- df_parameters_results[-train_indices, ]

# Perform regression for Shepard_stress with quadratic term and interactions
model_stress <- lm(
    stress ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +
        Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta,
    data = train_data
)

# Perform regression for Trustworthiness with quadratic term and interactions
model_trust30 <- lm(
    trust_k30 ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +
        Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta,
    data = train_data
)

# Perform regression for Trustworthiness with quadratic term and interactions
model_trust300 <- lm(
    trust_k300 ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +
        Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta,
    data = train_data
)

# Perform regression for KL_divergence with quadratic term and interactions
model_KL <- lm(KL_divergence ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +
                  Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta,
                data = train_data)

# Predictions on test set
pred_stress <- predict(model_stress, newdata = test_data)
pred_trust30 <- predict(model_trust30, newdata = test_data)
pred_trust300 <- predict(model_trust300, newdata = test_data)
pred_KL <- predict(model_KL, newdata = test_data)

# Assess model performance (e.g., R-squared) on test set
r_squared_stress <- 1 - sum((test_data$stress - pred_stress)^2) / sum((test_data$stress - mean(test_data$stress))^2)
r_squared_trust30 <- 1 - sum((test_data$trust_k30 - pred_trust30)^2) / sum((test_data$trust_k30 - mean(test_data$trust_k30))^2)
r_squared_trust300 <- 1 - sum((test_data$trust_k300 - pred_trust300)^2) / sum((test_data$trust_k300 - mean(test_data$trust_k300))^2)
r_squared_KL <- 1 - sum((test_data$KL_divergence - pred_KL)^2) / sum((test_data$KL_divergence - mean(test_data$KL_divergence))^2)

cat("R-squared for Shepard Stress:", r_squared_stress, "\n")
cat("R-squared for Trustworthiness:", r_squared_trust30, "\n")
cat("R-squared for Trustworthiness:", r_squared_trust300, "\n")
cat("R-squared for KL Divergence:", r_squared_KL, "\n")



# Define the model formula with quadratic term and interactions
model_formula <- as.formula(paste(
  "stress ~ poly(Perplexity, 2, raw = TRUE) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +",
  "Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta +",
  "Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + Early_exaggeration:Theta +",
  "Initial_momentum:Final_momentum + Initial_momentum:Theta +",
  "Final_momentum:Theta"
))


# Load necessary libraries
library(caret)
outcome <- outcomes[1]
formula <- paste0(
  outcome,
  " ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +",
  "Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta +",
  "Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + Early_exaggeration:Theta +",
  "Initial_momentum:Final_momentum + Initial_momentum:Theta +",
  "Final_momentum:Theta"
)

# Convert the string to a formula object
formula <- as.formula(formula)

# Perform cross-validation (5-fold cross-validation)
cv_results <- train(
  form = formula,
  data = df_parameters_results,
  method = "lm",     # Linear regression method
  trControl = trainControl(method = "cv", number = 10)  # 5-fold cross-validation
)

# View cross-validation results
print(cv_results)


# Function to check residuals for each outcome
check_residuals <- function(outcome, data, parameters) {
  # Construct the formula string
  formula <- paste(
    outcome,
    "~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +",
    "Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta +",
    "Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + Early_exaggeration:Theta +",
    "Initial_momentum:Final_momentum + Initial_momentum:Theta +",
    "Final_momentum:Theta"
  )
  
  # Convert the string to a formula object
  formula <- as.formula(formula)
  
  # Fit the model
  model <- lm(formula, data)
  
  # Plot residuals
  par(mfrow=c(2, 2))
  plot(model, which=1:4)
}

# Apply the function to each outcome
lapply(outcomes, check_residuals, data = df_parameters_results, parameters = parameters)


# Load necessary libraries for regularization
library(glmnet)

# Function to perform Ridge regression cross-validation for each outcome
perform_ridge_cv <- function(outcome, data, parameters) {
  # Extract predictor matrix and outcome vector
  X <- model.matrix(as.formula(paste("~", paste(parameters, collapse="+"))), data)
  y <- data[[outcome]]
  
  # Perform Ridge regression with cross-validation
  cv_ridge <- cv.glmnet(X, y, alpha=0, nfolds=10)
  
  # Print the cross-validation results
  cat("Cross-validation results for Ridge regression on", outcome, ":\n")
  print(cv_ridge)
}

# Apply the function to each outcome
lapply(outcomes, perform_ridge_cv, data = df_parameters_results, parameters = parameters)

# Function to check residuals for each outcome
check_residuals <- function(outcome, data, parameters) {
  # Construct the formula string
  formula <- paste(
    outcome,
    "~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta +",
    "Perplexity:Early_exaggeration + Perplexity:Initial_momentum + Perplexity:Final_momentum + Perplexity:Theta +",
    "Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + Early_exaggeration:Theta +",
    "Initial_momentum:Final_momentum + Initial_momentum:Theta +",
    "Final_momentum:Theta"
  )
  
  # Convert the string to a formula object
  formula <- as.formula(formula)
  
  # Fit the model
  model <- lm(formula, data)
  
  # Plot residuals
  par(mfrow=c(2, 2))  # Arrange plots in 2x2 grid
  plot(model)  # Default plot function for lm objects creates the 4 residual plots
}

# Apply the function to each outcome
lapply(outcomes, check_residuals, data = df_parameters_results, parameters = parameters)





#################
# Load the lme4 package
# Load the lme4 package
library(lme4)

# Assuming df_parameters_results is your dataset in R

# Add a unique identifier for each unique KL divergence value
df_parameters_results = readr::read_csv('output/df_parameters_results.csv')
df_parameters_results[,c(parameters,outcomes)] |> dplyr::group_by(KL_divergence) |> View()

xtabs(KL_divergence ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta, df_parameters_results)
#ftable(mytable)

data_scaled = data.frame(scale(df_parameters_results[,c(parameters,outcomes)]))
data_scaled$KL_id <- as.factor(as.numeric(as.factor(data_scaled$KL_divergence)))

# Define the formula for the mixed-effects model
formula <- "KL_divergence ~ Perplexity + I(Perplexity^2) + Early_exaggeration + 
            Initial_momentum + Final_momentum + Theta + 
            Perplexity:Early_exaggeration + Perplexity:Initial_momentum + 
            Perplexity:Final_momentum + Perplexity:Theta + 
            Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + 
            Early_exaggeration:Theta + 
            Initial_momentum:Final_momentum + Initial_momentum:Theta + 
            Final_momentum:Theta + (1 | KL_id)"

# Fit the mixed-effects model in R using lme4
model <- lmer(formula, data = data_scaled )

# Print the summary of the model
summary(model)

# Check the sample size per group
samples_grouped = df_parameters_results |>
  dplyr::group_by(
    KL_divergence, Perplexity, Early_exaggeration, Initial_momentum,
    Final_momentum, Theta
  ) |>
  dplyr::tally()

range(samples_grouped$n)
View(samples_grouped)

###
model_simple <- lm(KL_divergence ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta, data = data_scaled)
summary(model_simple)

model_interaction <- lm(KL_divergence ~ (Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta)^2, data = data_scaled)
summary(model_interaction)

model_quadratic <- lm(KL_divergence ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta, data = data_scaled)
summary(model_quadratic)

formula <- "KL_divergence ~ Perplexity + I(Perplexity^2) + Early_exaggeration + 
            Initial_momentum + Final_momentum + Theta + 
            Perplexity:Early_exaggeration + Perplexity:Initial_momentum + 
            Perplexity:Final_momentum + Perplexity:Theta + 
            Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + 
            Early_exaggeration:Theta + 
            Initial_momentum:Final_momentum + Initial_momentum:Theta + 
            Final_momentum:Theta + I(Perplexity^2):Early_exaggeration + I(Perplexity^2):Initial_momentum + 
            I(Perplexity^2):Final_momentum + I(Perplexity^2):Theta"
model_interactions <- lm(formula = formula, data = data_scaled)
summary(model_interactions)


# Compare models using ANOVA
anova(model_simple, model_quadratic, model_interactions)

par(mfrow = c(2, 2))
plot(model_interactions)


hist(data_scaled$KL_divergence, breaks = 20, main = "Histogram of KL Divergence", xlab = "KL Divergence")
#Decide on the number of categories and the cutoffs based on the histogram. For instance, you can create three categories: low, medium, and high.

# Define cutoffs
quantiles <- quantile(data_scaled$KL_divergence, probs = c(0.33, 0.66))
data_scaled$KL_divergence_cat <- cut(data_scaled$KL_divergence, breaks = c(-Inf, quantiles, Inf), labels = c("Low", "Medium", "High"))


table(data_scaled$KL_divergence_cat, data_scaled$Perplexity)



# GLM with all variables categorical
data_scaled$Perplexity <- as.factor(data_scaled$Perplexity)
data_scaled$Early_exaggeration <- as.factor(data_scaled$Early_exaggeration)
data_scaled$Initial_momentum <- as.factor(data_scaled$Initial_momentum)
data_scaled$Final_momentum <- as.factor(data_scaled$Final_momentum)
data_scaled$Theta <- as.factor(data_scaled$Theta)
data_scaled$KL_divergence_cat <- as.factor(data_scaled$KL_divergence_cat)



install.packages("ordinal")
library(ordinal)
data_scaled = data.frame(scale(df_parameters_results[,c(parameters,outcomes)]))
quantiles <- quantile(data_scaled$KL_divergence, probs = c(0.33, 0.66))
data_scaled$KL_divergence_cat <- cut(data_scaled$KL_divergence, breaks = c(-Inf, quantiles, Inf), labels = c("Low", "Medium", "High"))

data_scaled$Perplexity <- as.ordered(data_scaled$Perplexity)
data_scaled$Early_exaggeration <- as.ordered(data_scaled$Early_exaggeration)
data_scaled$Initial_momentum <- as.ordered(data_scaled$Initial_momentum)
data_scaled$Final_momentum <- as.ordered(data_scaled$Final_momentum)
data_scaled$Theta <- as.ordered(data_scaled$Theta)
data_scaled$KL_divergence_cat <- as.ordered(data_scaled$KL_divergence_cat)

#Ordinal Logistic Regression if the categories are ordered (low, medium, high).
library(MASS)

model_simple <- polr(KL_divergence_cat ~ Perplexity, data = data_scaled)
summary(model_simple)

model_additive <- polr(KL_divergence_cat ~ Perplexity + Early_exaggeration + Initial_momentum, data = data_scaled)
summary(model_additive)

model_full <- polr(KL_divergence_cat ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta, data = data_scaled)
summary(model_full)


library(VGAM)
model_vglm <- vglm(KL_divergence_cat ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta, family = cumulative(parallel = TRUE), data = data_scaled)
summary(model_vglm)


# Ridge Regularization
library(ordinal)

# Fit the ordinal logistic regression with ridge regularization
model_ridge <- clm(KL_divergence_cat ~ Perplexity + I(Perplexity^2) + Early_exaggeration + Initial_momentum + Final_momentum + Theta,
                   data = data_scaled,
                   link = "logit",
                   Hess = TRUE,
                   lambda = 1)  # Specify the regularization parameter (lambda)

summary(model_ridge)

formula <- "KL_divergence ~ Perplexity + I(Perplexity^2) + Early_exaggeration + 
            Initial_momentum + Final_momentum + Theta + 
            Perplexity:Early_exaggeration + Perplexity:Initial_momentum + 
            Perplexity:Final_momentum + Perplexity:Theta + 
            Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + 
            Early_exaggeration:Theta + 
            Initial_momentum:Final_momentum + Initial_momentum:Theta + 
            Final_momentum:Theta + I(Perplexity^2):Early_exaggeration + I(Perplexity^2):Initial_momentum + 
            I(Perplexity^2):Final_momentum + I(Perplexity^2):Theta"

clm(formula = formula, data = data_scaled, link = "logit", Hess = TRUE, lambda = 100)

model_glm <- glm(KL_divergence ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta, 
                 family = binomial, 
                 data = data_scaled2)
summary(model_glm)

coef_estimates <- coef(summary(model_interactions))[, "Estimate"]
coef_se <- coef(summary(model_interactions))[, "Std. Error"]

# Fit Bayesian logistic regression model
library(rstanarm)

# Specify priors based on the initial frequentist model if available
priors <- normal(location = coef_estimates, scale = coef_se, autoscale = FALSE)

# Fit Bayesian ordinal logistic regression model
model_bayesian_ordinal <- stan_polr(
  KL_divergence_cat ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta,
  data = data_scaled,
  method = "logistic",  # Specify the link function here, can be logistic, probit, loglog, cloglog, or cauchit
  prior = priors,  # Specify your priors here
  prior_intercept = normal(0, 10),
  chains = 4,  # Number of Markov chains
  iter = 2000  # Number of iterations per chain
)
summary(model_bayesian_ordinal)



# Load necessary libraries
library(MASS)   # For fitting ordinal logistic regression (polr)
library(brms)   # For Bayesian ordinal logistic regression (stan_polr)

# Assuming df_ordered contains your ordered categorical variables
# Replace with your actual dataset name if different
data_ordered <- data_scaled

# Function to calculate prior probabilities from observed frequencies
calculate_prior_probs <- function(var_name, data) {
  table_var <- table(data[[var_name]])
  prior_probs <- as.vector(table_var / sum(table_var))
  return(prior_probs)
}

# Calculate prior probabilities for each ordinal variable
prior_probs_perplexity <- calculate_prior_probs("Perplexity", data_ordered)
prior_probs_early_exaggeration <- calculate_prior_probs("Early_exaggeration", data_ordered)
prior_probs_initial_momentum <- calculate_prior_probs("Initial_momentum", data_ordered)
prior_probs_final_momentum <- calculate_prior_probs("Final_momentum", data_ordered)
prior_probs_theta <- calculate_prior_probs("Theta", data_ordered)

# Specify priors for each ordinal variable
priors <- list(
  prior("categorical", class = "b", coef = "Perplexity", location = 1:length(levels(data_ordered$Perplexity)), prob = prior_probs_perplexity),
  prior("categorical", class = "b", coef = "Early_exaggeration", location = 1:length(levels(data_ordered$Early_exaggeration)), prob = prior_probs_early_exaggeration),
  prior("categorical", class = "b", coef = "Initial_momentum", location = 1:length(levels(data_ordered$Initial_momentum)), prob = prior_probs_initial_momentum),
  prior("categorical", class = "b", coef = "Final_momentum", location = 1:length(levels(data_ordered$Final_momentum)), prob = prior_probs_final_momentum),
  prior("categorical", class = "b", coef = "Theta", location = 1:length(levels(data_ordered$Theta)), prob = prior_probs_theta)
)

# Fit Bayesian ordinal logistic regression model using stan_polr
model_bayesian_ordinal <- stan_polr(
  formula = KL_divergence_cat ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta,
  data = data_ordered,
  method = "logistic",    # Specify the link function (logistic for ordinal regression)
  prior = priors,         # Use defined priors
  prior_intercept = normal(0, 10),  # Specify prior for intercepts
  chains = 4,             # Number of Markov chains
  iter = 2000             # Number of iterations per chain
)

# Summarize the model
summary(model_bayesian_ordinal)



# Fit Bayesian ordinal logistic regression model using stan_polr
model_bayesian_ordinal <- stan_polr(
  KL_divergence_cat ~ Perplexity + Early_exaggeration + Initial_momentum + Final_momentum + Theta,
  data = data_ordered,
  method = "logistic",    # Specify the link function (logistic for ordinal regression)
  prior = priors,         # Use defined priors
  prior_intercept = normal(0, 10),  # Specify prior for intercepts
  chains = 4,             # Number of Markov chains
  iter = 2000             # Number of iterations per chain
)

# Summarize the model
summary(model_bayesian_ordinal)




# Load required library
library(randomForest)

df_parameters_results = readr::read_csv('output/df_parameters_results.csv')

# Perform regression for each outcome
outcomes <- names(df_parameters_results)[10:13]
parameters <- names(df_parameters_results)[3:7]
parameters_df <- df_parameters_results[,parameters]

# Example data (replace with your actual data)
set.seed(123)  # for reproducibility

# Split data into predictors (X) and target (y)

# Train the random forest regressor
formula <- "KL_divergence ~ Perplexity + Early_exaggeration + 
            Initial_momentum + Final_momentum + Theta + 
            Perplexity:Early_exaggeration + Perplexity:Initial_momentum + 
            Perplexity:Final_momentum + Perplexity:Theta + 
            Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + 
            Early_exaggeration:Theta + 
            Initial_momentum:Final_momentum + Initial_momentum:Theta + 
            Final_momentum:Theta"
rf_model <- randomForest(formula = KL_divergence ~ Perplexity + Early_exaggeration + 
            Initial_momentum + Final_momentum + Theta + 
            Perplexity:Early_exaggeration + Perplexity:Initial_momentum + 
            Perplexity:Final_momentum + Perplexity:Theta + 
            Early_exaggeration:Initial_momentum + Early_exaggeration:Final_momentum + 
            Early_exaggeration:Theta + 
            Initial_momentum:Final_momentum + Initial_momentum:Theta + 
            Final_momentum:Theta, data = df_parameters_results, ntree = 500, importance = TRUE)

# Print the model summary
print(rf_model)

# Feature importance plot (optional)
varImpPlot(rf_model)

# Predictions (optional)
# Suppose you have new data 'new_data' with the same structure as 'my_data'
# new_data <- ...  # prepare your new data
# predictions <- predict(rf_model, new_data)

# Compute partial dependence plot for two-way interaction (e.g., feature 1 and feature 2)
rf_model_KL <- randomForest(x=df_parameters_results[,parameters],y=df_parameters_results$KL_divergence, importance = TRUE, ntree = 100)

# Compute partial dependence plot for two-way interaction (e.g feature 1 and feature 2)
partialPlot(rf_model_KL, data.frame(df_parameters_results), x.var = c(parameters[1]))

# Compute Friedman's H-statistic (total interaction) for feature 1
pd <- partialPlot(rf_model_KL, data.frame(df_parameters_results), x.var = c(parameters[1]), plot = FALSE)
interaction_measure <- diff(range(pd$y))

cat("Friedman's H-statistic (total interaction) for feature 1:", interaction_measure, "\n")


###########
#https://cran.r-project.org/web/packages/iml/vignettes/intro.html
#install.packages("iml")

library("iml")
library("randomForest")
data("Boston", package = "MASS")
rf = randomForest(medv ~ ., data = Boston, ntree = 50)
X = Boston[which(names(Boston) != "medv")]
model = Predictor$new(rf, data = X, y = Boston$medv)
effect = FeatureEffects$new(model)
effect$plot(features = c("lstat", "age", "rm"))

X <- Boston[which(names(Boston) != "medv")]
predictor <- Predictor$new(rf, data = X, y = Boston$medv)

rf <- randomForest(x=df_parameters_results[,parameters],y=df_parameters_results$KL_divergence, importance = TRUE, ntree = 100)
X <- parameters_df
predictor <- Predictor$new(rf, data = X, y = df_parameters_results$KL_divergence)

imp <- FeatureImp$new(predictor, loss = "mae")
plot(imp)


imp$results
#Besides knowing which features were important, we are interested in how the features influence the predicted outcome. The FeatureEffect class implements accumulated local effect plots, partial dependence plots and individual conditional expectation curves. The following plot shows the accumulated local effects (ALE) for the feature ‘lstat’.
ale <- FeatureEffect$new(predictor, feature = "lstat", grid.size = 10)
ale$plot()

ale$set.feature("rm")
ale$plot()

#Measure interactions

#We can also measure how strongly features interact with each other. The interaction measure regards how much of the variance of f(x)
#is explained by the interaction. The measure is between 0 (no interaction) and 1 (= 100% of variance of f(x) due to interactions). For each feature, we measure how much they interact with any other feature:
interact <- Interaction$new(predictor, grid.size = 15)
plot(interact)

#We can also specify a feature and measure all it’s 2-way interactions with all other features:

interact <- Interaction$new(predictor, feature = "Early_exaggeration", grid.size = 15)
plot(interact)