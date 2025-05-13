# Predicting Diabetes with tidymodels

A comparative analysis of binary classifiers applied to the **Pima Indians Diabetes dataset**, using R's [`tidymodels`](https://www.tidymodels.org/) framework.

---

## Dataset Source and Context

This dataset originates from the **National Institute of Diabetes and Digestive and Kidney Diseases** and is made available via the `mlbench::PimaIndiansDiabetes2` dataset in R.

- Goal: Predict whether a patient has diabetes based on clinical measurements.
- Observations: 768 women (age ≥21)
- Target variable: `diabetes` (binary: "pos" or "neg")
- Predictors: glucose, BMI, insulin, age, blood pressure, etc.

> Note: Missing values are handled by filtering rows with NAs (standard practice for this dataset).

---

## Purpose of the Project

The objective is to build, tune, and evaluate several classification models to:
- Predict diabetes status
- Compare model performance using consistent cross-validation
- Interpret feature importance and decision patterns

---

## Modeling Steps

1. Data Cleaning
   - Loaded from `mlbench`
   - Cleaned with `janitor::clean_names()`
   - Removed rows with missing values

2. Preprocessing
   - All numeric features normalized
   - One-hot encoding applied if needed
   - Built using `recipes`

3. Models Compared
   - LASSO (GLMNet with `mixture = 1`)
   - Ridge Regression (`mixture = 0`)
   - Elastic Net (`mixture ∈ [0, 1]`)
   - Random Forest (`ranger`)
   - XGBoost (`xgboost`)

4. Evaluation
   - Stratified 5-fold cross-validation
   - Primary metric: ROC AUC
   - Secondary metrics: accuracy, sensitivity, specificity
   - ROC curves plotted for all models
---

## Reproducibility Instructions

> Requires R (≥ 4.2) and RStudio

### 1. Clone the repo

```bash
git clone https://github.com/your-username/pima_tidymodels.git
cd pima_tidymodels
