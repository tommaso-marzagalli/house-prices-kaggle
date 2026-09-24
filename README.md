# House Prices — Advanced Regression Techniques

Solution notebook for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), combining feature engineering, systematic model comparison, hyperparameter tuning, and a custom hybrid trend/seasonality model.

**Leaderboard score: 0.13268 (RMSLE)**

## Overview

The goal is to predict residential sale prices (`SalePrice`) from ~80 raw features describing each property. Rather than fitting a single model, this notebook builds an end-to-end pipeline and compares several modeling strategies under the same cross-validation scheme, letting the data decide which one wins.

## Approach

1. **EDA** — missing value analysis, target distribution.
2. **Feature engineering** — five hand-built features (area ratios, outdoor space, room spaciousness, a neighborhood-level grouped statistic computed leak-safely from training data only).
3. **Mutual Information analysis** — scores every feature, including the engineered ones, by how informative it is about `SalePrice` on its own.
4. **Preprocessing pipeline** — median imputation for numerical columns, One-Hot Encoding for low-cardinality categoricals, target encoding (`MEstimateEncoder`) for the high-cardinality `Neighborhood` column, a K-Means cluster feature, and PCA components from correlated area features. Every step is wrapped as a scikit-learn transformer so it refits correctly on every cross-validation fold.
5. **Model comparison** — Random Forest, Ridge, and XGBoost compared via 5-fold cross-validation.
6. **Hyperparameter tuning** — the best-performing model (XGBoost) tuned for more trees and a lower learning rate.
7. **Hybrid model** — a custom estimator where a linear model captures the market trend and sale-month seasonality, and XGBoost learns the residual from the house's own characteristics.
8. **Final training and submission** — the model with the lowest cross-validation error is automatically selected and retrained on the full training set.

## Results

| Model                                   | CV MAE  |
|------------------------------------------|--------:|
| Random Forest (100 trees)                | 17,585  |
| Random Forest (300 trees, max_depth=15)  | 17,608  |
| Ridge (linear baseline)                  | 18,589  |
| XGBoost (default)                        | 17,516  |
| **XGBoost (tuned)**                      | **15,585** |
| Hybrid (trend + seasonality + XGBoost)   | 15,847  |

Tuned XGBoost was selected as the final model. The public leaderboard score was **0.13268 (RMSLE)**.

## Dataset

The competition data is **not included** in this repository, per Kaggle's competition rules on data redistribution. To run the notebook:

1. Join the [competition on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) and accept the rules.
2. Download `train.csv` and `test.csv`, either from the competition page or via the Kaggle API:
```bash
   kaggle competitions download -c house-prices-advanced-regression-techniques
```
3. Place both files in a local `data/` folder (already excluded via `.gitignore`).

## How to run

```bash
pip install pandas numpy scikit-learn xgboost category_encoders matplotlib jupyter
jupyter notebook house-prices-competition-notebook.ipynb
```

Update the `TRAIN_PATH` / `TEST_PATH` variables in the first code cell if your data folder is named differently.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · category_encoders · Matplotlib

## Author

**Tommaso Marzagalli** — Mathematical Engineering student, Politecnico di Milano
[LinkedIn](https://www.linkedin.com/in/tommaso-marzagalli-aa487b270)
