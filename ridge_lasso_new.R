# remove object from my environment
rm(list = ls())

# Load necessary libraries
library(dplyr)
library(readr)
library(readxl)
library(caret)
library(glmnet)
library(ggplot2)

# Set working directory
setwd("C:/Users/zhwja/OneDrive - The University of Chicago/Big Data/FINAL PROJECT")

# Load the variable description file
var_desc <- read_excel("PPHA_30546_MP03-Variable_Description.xlsx")

# Select variables from Opportunity Insights and PM COVID sources
selected_vars <- var_desc %>%
  filter(Source %in% c("Opportunity Insights", "PM COVID")) %>%
  pull(Variable)

# Ensure county, state, and deathspc are included
selected_vars <- c(selected_vars, "county", "state", "deathspc")

# Load the main dataset
data <- read_csv("Data-Covid002.csv")

# Filter the dataset
filtered_data <- data %>% select(all_of(selected_vars))

# Remove rows with missing values
cleaned_data <- na.omit(filtered_data)

# Create dummy variables for each state
cleaned_data_with_dummies <- cleaned_data %>%
  mutate(across(state, as.factor)) %>% # Convert state to a factor variable
  model.matrix(~ state - 1, data = .) %>% # Create dummy variables
  as.data.frame() %>% 
  bind_cols(cleaned_data, .) %>% 
  select(-state)  # Drop the original state column

# Set seed for reproducibility
set.seed(11)

# Split the data into training (80%) and test (20%) sets
train_index <- createDataPartition(cleaned_data_with_dummies$deathspc, p = 0.8, list = FALSE)
train_data <- cleaned_data_with_dummies[train_index, ]
test_data <- cleaned_data_with_dummies[-train_index, ]

# Define the dependent variable
y_train <- train_data$deathspc
y_test <- test_data$deathspc

# Define the independent variables (excluding non-numeric variables)
X_train <- train_data %>% select(-deathspc, -county)
X_test <- test_data %>% select(-deathspc, -county)

# Fit OLS model
ols_model <- lm(y_train ~ ., data = X_train)

# Compute MSE for training and test sets
train_predictions <- predict(ols_model, X_train)
test_predictions <- predict(ols_model, X_test)

train_mse <- mean((train_predictions - y_train)^2)
test_mse <- mean((test_predictions - y_test)^2)

# Print MSE results
cat("Training Set MSE:", train_mse, "\n")
cat("Test Set MSE:", test_mse, "\n")

# Standardize predictors (mean=0, variance=1)
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test)

# Define lambda (α) values for tuning
lambda_grid <- 10^seq(-2, 2, length.out = 100)

# Ridge Regression with cross-validation
set.seed(25)
ridge_cv <- cv.glmnet(as.matrix(X_train_scaled), y_train, alpha = 0, lambda = lambda_grid, nfolds = 10)

# Lasso Regression with cross-validation
set.seed(25)
lasso_cv <- cv.glmnet(as.matrix(X_train_scaled), y_train, alpha = 1, lambda = lambda_grid, nfolds = 10)

# Retrieve optimal lambda values
ridge_lambda_opt <- ridge_cv$lambda.min
lasso_lambda_opt <- lasso_cv$lambda.min

# Print optimal lambda values
cat("Optimal Lambda for Ridge Regression:", ridge_lambda_opt, "\n")
cat("Optimal Lambda for Lasso Regression:", lasso_lambda_opt, "\n")

# Convert lambda values to a dataframe for plotting
ridge_plot_data <- data.frame(
  lambda = ridge_cv$lambda,
  cv_error = ridge_cv$cvm
)

lasso_plot_data <- data.frame(
  lambda = lasso_cv$lambda,
  cv_error = lasso_cv$cvm
)

# Plot Ridge Regression: CV Error vs. Lambda
ggplot(ridge_plot_data, aes(x = lambda, y = cv_error)) +
  geom_line(color = "blue") +
  geom_vline(xintercept = ridge_lambda_opt, color = "red", linetype = "dashed") +
  scale_x_log10() +
  labs(title = "Ridge Regression: CV Error vs. Lambda",
       x = "Lambda (α)", y = "Cross-Validation Error") +
  theme_minimal()

# Plot Lasso Regression: CV Error vs. Lambda
ggplot(lasso_plot_data, aes(x = lambda, y = cv_error)) +
  geom_line(color = "blue") +
  geom_vline(xintercept = lasso_lambda_opt, color = "red", linetype = "dashed") +
  scale_x_log10() +
  labs(title = "Lasso Regression: CV Error vs. Lambda",
       x = "Lambda (α)", y = "Cross-Validation Error") +
  theme_minimal()

# Re-estimate Ridge Regression using the optimal lambda
ridge_final <- glmnet(as.matrix(X_train_scaled), y_train, alpha = 0, lambda = ridge_lambda_opt)

# Re-estimate Lasso Regression using the optimal lambda
lasso_final <- glmnet(as.matrix(X_train_scaled), y_train, alpha = 1, lambda = lasso_lambda_opt)

# Print confirmation message
cat("Ridge Regression re-estimated with λ =", ridge_lambda_opt, "\n")
cat("Lasso Regression re-estimated with λ =", lasso_lambda_opt, "\n")

# Predict using Ridge Regression
ridge_train_pred <- predict(ridge_final, as.matrix(X_train_scaled))
ridge_test_pred <- predict(ridge_final, as.matrix(X_test_scaled))

# Compute MSE for Ridge Regression
ridge_train_mse <- mean((ridge_train_pred - y_train)^2)
ridge_test_mse <- mean((ridge_test_pred - y_test)^2)

# Predict using Lasso Regression
lasso_train_pred <- predict(lasso_final, as.matrix(X_train_scaled))
lasso_test_pred <- predict(lasso_final, as.matrix(X_test_scaled))

# Compute MSE for Lasso Regression
lasso_train_mse <- mean((lasso_train_pred - y_train)^2)
lasso_test_mse <- mean((lasso_test_pred - y_test)^2)

# Print results
cat("Ridge Regression - Training MSE:", ridge_train_mse, "\n")
cat("Ridge Regression - Test MSE:", ridge_test_mse, "\n")
cat("Lasso Regression - Training MSE:", lasso_train_mse, "\n")
cat("Lasso Regression - Test MSE:", lasso_test_mse, "\n")