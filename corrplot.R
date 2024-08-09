

# Load data
df <- read.csv('output/df_metric_momentum_wresults.csv')
colnames(df)
parameters = c('Perplexity', 'Early_exaggeration', 'Initial_momentum', 'Final_momentum', 'Theta')
outcomes = c('KL_divergence', 'trust_k30', 'trust_k300', 'stress', 'tSNE_runtime_min')


# Compute correlation matrix and p-values
corr_results <- psych::corr.test(df[, -1], adjust = 'BH')

# Extract correlation matrix and p-values
cor_matrix <- corr_results$r
p_values <- corr_results$p
p_values_adj <- corr_results$p.adj

# Create a significance matrix (1 for significant, 0 for not significant)
sig_matrix <- ifelse(p_values_adj < 0.05, 1, 0)

# Set significance for non-significant correlations to NA
cor_matrix[!sig_matrix] <- NA

# Plot the correlation matrix using corrplot
corrplot::corrplot(cor_matrix, 
         method = "color",           # Use colors to show the correlations
         type = "upper",             # Show only the upper triangle
         col = colorRampPalette(c("blue", "white", "red"))(200),  # Color scale
         addCoef.col = "black",      # Show correlation coefficients
         tl.col = "black",           # Color of text labels
         tl.srt = 45,                # Text label rotation
         na.label = "n",             # Label for non-significant values
         diag = FALSE,               # Hide the diagonal
         title = "Significant Correlations Only")

# Save the plot
ggsave("significant_correlations.png", width = 12, height = 8)