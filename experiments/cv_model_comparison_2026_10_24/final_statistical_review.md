# Final Statistical Review

**Experiment:** experiments/cv_model_comparison_2026_10_24/
**Review type:** Read-only (no retraining/modification)

## 1) Saved results inspected
-
eport.md, experiment_results.json, cv_results.json, split.json, 	est_predictions.npy, aseline_predictions.npy, 	est_y.npy, 	est_pids.npy
- Source: experiments/final_cohort_2026_10_24/ (unchanged). Config: prompt-level GroupKFold by prompt (5 folds), 340 train prompts / 146 test prompts, seed 12345. Hidden states (486,29,2048) mean-pooled over layers (axis=1)->(486,2048). Fusion uses PCA(50) fit per fold on train only.

## 2) MAE improvement
- Baseline (train mean) test MAE: 0.31402095084609183
- Selected (rf_hidden): test MAE: 0.27487671232876715
- Absolute improvement: 0.03914423851732468
- Relative improvement: (0.31402095-0.27487671)/0.31402095*100 = 12.47%

## 3) Paired bootstrap
- Quantity bootstrapped: difference d_i = MAE_baseline(test_sample_i) - MAE_model(test_sample_i) on paired test samples (resampled by test prompt/rows with replacement), same test set alignment.
- Paired by test prompt/rows: yes (paired bootstrap over test set; predictions saved per test row aligned to test_pids/test_y).
- Replicates: n=1000, random seed: 12345 (from experiment config/split).
- CI (95%): [0.016852790088638214, 0.05994144842868656] for (baseline - model)
- Excludes zero: Yes (CI entirely above 0). P(diff>0)=1.0.

## 4) Cross-validation stability
- Folds: 5 (GroupKFold by prompt)
- rf_hidden fold MAEs: [0.2916176470588235, 0.2342794117647059, 0.32492647058823526, 0.2529558823529412, 0.26154411764705887]
- Mean ± std: 0.27306470588235293 ± 0.03185982436085448
- Assessment: performance varies across folds (range ~0.090) with moderate std; not obviously driven by a single fold, but variability is non-trivial.
- Selection confinement: Model/config selected by best CV MAE on training partition only (GroupKFold within train). Test set untouched during selection/tuning. Verified in script logic and split.

## 5) R² interpretation
- Baseline test R²: -0.0019779841735334536 (essentially zero/worse than mean predictor; train-mean is near-constant predictor)
- rf_hidden test R²: -0.09968838666683766 (more negative than baseline)
- Interpretation: Negative R² means worse than predicting the mean of the training targets on the test set. The MAE improved relative to the train-mean baseline in absolute terms, but R² being more negative indicates poorer variance explanation relative to that baseline; this is a common pattern when baseline is near-constant and errors behave differently. Do not interpret MAE improvement alone as strong evidence of generalization.

## 6) Conclusion
**MAE improvement is promising but statistically inconclusive.**

Rationale:
- Paired bootstrap shows 95% CI for (baseline MAE - model MAE) excludes zero with P>0~1.0 based on saved results, suggesting the MAE difference is unlikely due to sampling noise on this test set.
- However, R² is negative for both (baseline ~ -0.002, model ~ -0.100) indicating neither meaningfully explains variance relative to train-mean; CV shows fold-to-fold variability (std ~0.032).
- Improvements are observed on a fixed held-out set with specific features/pooling; generalization claims remain cautious given these limitations.
