# CV Model Comparison (Training-Only CV)
Source: experiments\final_cohort_2026_10_24
Output: experiments\cv_model_comparison_2026_10_24

## Feature pooling
Hidden states (486,29,2048) mean-pooled over layers (axis=1) -> (486,2048). Prompt features (5 dims). Fusion = prompt + PCA(50) of pooled hidden fit per fold on train only.

## Split
Prompt-level GroupKFold by prompt (5 folds), 340 train prompts / 146 test prompts (seed 12345). Test set untouched during CV/tuning.

## Selected
Best by CV MAE: rf_hidden -> {'model': 'rf', 'config': 'hidden', 'params': {'n_estimators': 100, 'pca_components': 50}, 'pipe': 'pipe_h+rf'}

## Metrics
Baseline (train mean): MAE=0.31402095084609183, RMSE=0.3830361960661166, R2=-0.0019779841735334536
Test (rf/hidden): MAE=0.27487671232876715, RMSE=0.40127818042125313, R2=-0.09968838666683766

## Bootstrap (paired)
MAE_diff (baseline-model), 95% CI: [0.016852790088638214, 0.05994144842868656], P(diff>0)=1.0

## Notes
Negative R2 indicates worse than train-mean baseline on test set. Differences small/uncertain; no test data used for selection.