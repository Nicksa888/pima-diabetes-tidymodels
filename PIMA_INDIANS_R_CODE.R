# =====================================================
# Full loop: Tune, finalize, and evaluate all models
# =====================================================

library(tidymodels)
library(mlbench)
library(janitor)
library(future)
plan(multisession)

# Load data
data(PimaIndiansDiabetes2)
pima <- PimaIndiansDiabetes2 |>
  clean_names() |>
  drop_na()

# Recipe
pima_recipe <- recipe(diabetes ~ ., data = pima) |>
  step_normalize(all_numeric_predictors())

# Cross-validation
set.seed(123)
cv_folds <- vfold_cv(pima, v = 5, strata = diabetes)

# 80/20 final test split
data_split <- initial_split(pima, prop = 0.8, strata = diabetes)

# Model specs
models <- list(
  LASSO = logistic_reg(penalty = tune(), mixture = 1) |> set_engine("glmnet") |> set_mode("classification"),
  Ridge = logistic_reg(penalty = tune(), mixture = 0) |> set_engine("glmnet") |> set_mode("classification"),
  ElasticNet = logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet") |> set_mode("classification"),
  RandomForest = rand_forest(mtry = tune(), min_n = tune(), trees = 500) |> set_engine("ranger") |> set_mode("classification"),
  XGBoost = boost_tree(trees = 500, learn_rate = tune(), tree_depth = tune()) |> set_engine("xgboost") |> set_mode("classification")
)

# Metric set
my_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity)

# Result containers
tuned_models <- list()
final_models <- list()
final_fits <- list()
final_metrics <- list()

# Loop through each model
for (model_name in names(models)) {
  cat(glue::glue("\n🔄 Tuning {model_name}...\n"))
  
  # Create workflow
  wf <- workflow() |> add_model(models[[model_name]]) |> add_recipe(pima_recipe)
  
  # Tune model
  tuned <- tune_grid(
    wf,
    resamples = cv_folds,
    grid = 20,
    metrics = my_metrics
  )
  
  # Select best
  best_params <- select_best(tuned, metric = "roc_auc")
  final_wf <- finalize_workflow(wf, best_params)
  fit <- last_fit(final_wf, split = data_split)
  metrics <- collect_metrics(fit) |> mutate(model = model_name)
  
  # Store
  tuned_models[[model_name]] <- tuned
  final_models[[model_name]] <- final_wf
  final_fits[[model_name]] <- fit
  final_metrics[[model_name]] <- metrics
}

# Combine all metrics
all_results <- bind_rows(final_metrics)
print(all_results)

# Key Performance Metrics Explained
# Metric	Meaning
# accuracy	Proportion of correct predictions overall
# roc_auc	Area under the ROC curve (discrimination ability; higher = better)
# brier_class	Brier score (calibration; lower = better)
# 
# Model Comparison Summary
# Model	Accuracy	ROC AUC	Brier Score	Interpretation
# LASSO	0.747	0.832	0.163	Solid discrimination, good calibration
# Ridge	0.734	0.832	0.166	Similar to LASSO but slightly worse overall
# Elastic Net	0.734	0.829	0.163	Comparable to Ridge, no clear gain from mixing L1/L2
# Random Forest	0.772	0.843	0.154	Best accuracy and calibration; likely captures non-linear patterns
# XGBoost	0.747	0.843	0.160	Excellent ROC AUC, better than LASSO but slightly behind RF in accuracy
# 
# Best Models by Metric
# 
# Highest ROC AUC: Random Forest and Xgboost: (0.843)
# Highest Accuracy: Random Forest (0.772)
# Lowest Brier Score: Random Forest (0.154)
# 
# Takeaways
# Random Forest is your best all-around performer.
# 
# XGBoost offers similarly strong discrimination, slightly lower calibration.
# 
# LASSO and Ridge perform well but are outpaced by tree-based models, especially on accuracy
# 
# Elastic Net doesn’t improve over LASSO/Ridge here — likely due to the nature of your features (not strongly correlated or redundant).
