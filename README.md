## Assignment 8: Ensemble Learning for Complex Regression (Bike Sharing Demand)

**Student Name:** NAVEEN US  
**Roll Number:** DA25M020  
**Course:** DA5401 - Data Analytics Laboratory

## Notebook Overview
This assignment applies ensemble learning techniques to the Bike Sharing Demand dataset to build accurate regression models. We compare Bagging, Boosting, and Stacking against single-model baselines, add cross-validation, perform hyperparameter tuning, and produce comprehensive visualizations and error analyses.

### Objectives
- Build baselines (Decision Tree, Linear Regression) and ensemble models (Bagging, Gradient Boosting, Stacking).
- Add K-Fold cross-validation and hyperparameter tuning (GridSearchCV).
- Evaluate with RMSE and R²; analyze residuals and feature importance.
- Create visualizations to explain performance, learning behavior, and errors.

## What’s Implemented

### Part A: Data & Baselines
- Data loading, cleaning, one-hot encoding of categorical variables.
- Train/test split (80/20).
- Baseline models: Decision Tree and Linear Regression.

### Part B: Bagging (Variance Reduction)
- `BaggingRegressor` with Decision Tree base learner.
- Comparison vs single Decision Tree (RMSE and residuals).

### Part C: Boosting (Bias Reduction)
- `GradientBoostingRegressor` with analysis of learning behavior and feature importance.

### Part D: Stacking (Optimal Combination)
- Base learners: KNN, Bagging, Gradient Boosting.
- Meta-learner: Ridge.
- Visual comparison of base and meta predictions.

### Part E: Cross-Validation & Hyperparameter Tuning
- 5-fold CV using `'neg_root_mean_squared_error'` for scoring.
- GridSearchCV for: Decision Tree, Bagging, Gradient Boosting, KNN, and Stacking (Ridge α).
- Final comparisons prioritize tuned models when available.

## Key Results

### Best Performing Model (tuned-preferred)
- **Stacking Regressor (Tuned)** — RMSE ≈ **42.59**, R² ≈ **0.9427**

### Tuned Models (summary)
- Gradient Boosting (Tuned): RMSE ≈ 44.56, R² ≈ 0.9373  
- Bagging (Tuned): RMSE ≈ 47.87, R² ≈ 0.9276  
- Decision Tree (Tuned): RMSE ≈ 61.15, R² ≈ 0.8819  
- KNN (Tuned): RMSE ≈ 81.99, R² ≈ 0.7877  
- Baselines for reference — Linear Regression: RMSE ≈ 100.45; Decision Tree: RMSE ≈ 118.56

### Bias–Variance Insights
- Bagging reduces variance vs a single tree but underperforms tuned Boosting/Stacking.
- Gradient Boosting produces large bias reduction; strongest individual ensemble.
- Stacking combines complementary signals (KNN locality, tree interactions, boosting corrections) to generalize best.

### Feature Insights
- Important drivers: hour-of-day (`hr_*`), `atemp`, `temp`, `hum`, `workingday`, `yr`.
- Errors increase at extreme demand ranges; tuned Stacking remains well-calibrated broadly.

## Files in This Assignment

### Notebook
- `Assignment8.ipynb` — complete code, cross-validation and tuning, visualizations, and discussion.

### Output Visualizations (saved at run time)
1. `eda_visualizations.png` — EDA: targets, correlations, hourly trends  
2. `baseline_models_comparison.png` — baseline performance and residuals  
3. `bagging_analysis.png` — Bagging vs single tree analysis  
4. `gradient_boosting_analysis.png` — Boosting learning curve, importance, performance  
5. `stacking_analysis.png` — Base and meta predictions, gains vs others  
6. `performance_progression.png` — performance evolution (tuned-preferred)  
7. `ensemble_models_comparison.png` — final models RMSE and R² comparison  
8. `comprehensive_analysis.png` — error and residual distributions, improvements  
9. `prediction_comparison.png` — actual vs predicted across selected models  
10. `feature_importance.png` — GB feature importance (top)  
11. `feature_and_error_analysis.png` — feature overlap and error by demand range

Note: The final comparison and plots prefer tuned models where available and fall back to untuned baselines if not.

## How to Run
1. Open the notebook:
   - `Assignment8.ipynb`
2. Ensure the dataset file `hour.csv` is in the same directory (or update the path in the data loading cell).  
   Dataset source: `https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset`
3. Run all cells sequentially:
   - Data loading and preprocessing  
   - Baseline models  
   - Bagging, Gradient Boosting, Stacking  
   - Cross-validation and hyperparameter tuning  
   - Final comparisons and visualizations

## Requirements
Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Reproducibility Notes
- Fixed `random_state=42` for data splits and models where applicable.
- CV scoring uses `'neg_root_mean_squared_error'` (version-safe across sklearn).
- Visualizations saved with `dpi=300` and tight layout.
- Train/test split is 80/20; one-hot encoding applied consistently.

## Troubleshooting
- If you see a message like “Exception ignored in: ResourceTracker… ChildProcessError,” it’s a harmless multiprocessing shutdown warning from background parallel jobs. To avoid it, run with fewer nested parallel jobs or use joblib’s threading backend for GridSearchCV.

## Citation (Dataset)
Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113–127. Bike Sharing Dataset from UCI Machine Learning Repository.


